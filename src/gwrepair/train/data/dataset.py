import os
from pathlib import Path
import logging
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset

from ml4gw import gw
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.dataloading import Hdf5TimeSeriesDataset

from gwrepair.train.data.signals import WaveformSampler, GlitchSampler
from gwrepair.train.data.clusterglitch import MultiClusterGlitchSampler
    
class RepairDataset:

    def __init__(
        self,
        data_dir: Path | str,
        waveform_args: dict,
        glitch_args: dict,
        sample_rate: float = 2048,
        duration: float = 5,
        ifos: List[str] = ["H1", "L1", "V1"],
        fduration: float = 2,
        psd_length: float = 132,
        fftlength: float = 5,
        f_min: float = 20,
        right_pad: float = 1,
        batch_size: int = 10,
        batches_per_epoch: int = 10,
        num_val_samples: int = 3,
        num_test_samples: int = 200,
        logger: logging.Logger | None = None,
        device: str = 'cpu',
        waveform_snr: List[float] | None = None,
        glitch_snr: List[float] | None = None,
        whiten: bool = True,
        mute_waveform: bool = False,
        mute_glitch: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.ifos = ifos
        self.fduration = fduration
        self.psd_length = psd_length
        self.fftlength = fftlength
        self.f_min = f_min
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.logger = logger
        self.device = device
        self.waveform_snr = waveform_snr
        self.glitch_snr = glitch_snr
        self.whiten = whiten
        self.mute_waveform = mute_waveform
        self.mute_glitch = mute_glitch

        ## signal generators
        right_pad = right_pad + fduration / 2
        waveform_args["sample_rate"] = sample_rate
        waveform_args["duration"] = duration
        waveform_args["ifos"] = ifos
        waveform_args["f_min"] = f_min
        waveform_args["right_pad"] = right_pad
        
        glitch_args["sample_rate"] = sample_rate
        glitch_args["duration"] = duration
        glitch_args["fduration"] = fduration
        glitch_args["right_pad"] = right_pad
        glitch_args["device"] = device

        num = torch.distributions.Geometric(probs=0.5, )
        
        self.waveform_gen = WaveformSampler(**waveform_args)
        self.glitch_gen = MultiClusterGlitchSampler(num=num, **glitch_args)
            
    def log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def train_val_fnames(self):
        fnames = sorted(os.listdir(self.data_dir))
        fnames = [os.path.join(self.data_dir, f) for f in fnames]
        num_files = len(fnames)
        len_valid_files = num_files // 2
        self.train_fnames = fnames[len_valid_files:]
        self.val_fnames = fnames[:len_valid_files]

    def transforms(self):
        self.signal_kernel = int(self.duration*self.sample_rate)
        self.psd_kernel = int(self.psd_length*self.sample_rate)
        self.background_kernel = int(self.signal_kernel + self.psd_kernel)
        self.spectral_density = SpectralDensity(
            sample_rate = self.sample_rate, 
            fftlength = self.fftlength,
            overlap = None,
            average = "mean",
            window = None,
            fast = True,
        )
        self.psd_splits = [self.psd_kernel, self.signal_kernel]
        
        self.whitener_tool = Whiten(
            fduration = self.fduration,
            sample_rate = self.sample_rate,
            highpass = self.f_min,
        ).to(self.device)   ## tool to whiten data even if self.whiten = False

        if self.whiten:
            self.whitener = self.whitener_tool
        else:
            self.whitener = torch.nn.Identity()

    def background(self, fnames, batches, batches_per_epoch):
        dataset = Hdf5TimeSeriesDataset(
            fnames,
            channels=self.ifos,
            kernel_size=self.background_kernel,
            batch_size=batches,
            batches_per_epoch=batches_per_epoch,
            coincident=False,
        )
        return dataset

    def inject(self, X, step='train', injection="both"):

        if step == 'train':
            X = X.to(self.device)

        ## make psd
        psd_X, X = torch.split(X, self.psd_splits, dim=-1)
        psds = self.spectral_density(psd_X.double().to(self.device))

        ## make injections
        if self.mute_waveform:
            waveform = torch.zeros_like(X)
            params = {"distance":  torch.zeros_like(X)}
            scale = torch.ones_like(X).unsqueeze(0).unsqueeze(0)
            jitters = 0
        else:
            waveform, params, jitters = self.waveform_gen(X)
            waveform, scale = self.rescale_snr(waveform, psds, self.waveform_snr)

        
        glitch = self.glitch_gen(X)
        glitch, _ = self.rescale_snr(glitch, psds, self.glitch_snr, ifos=1)

        if step == "train":
            
            ## inject and whiten
            X = X + waveform + glitch
            X = self.whitener(X, psds)
            psd_X = self.whitener(psd_X[:, :1], psds[:, :1])

            if injection == "both":
                waveform = self.whitener(waveform[:, :1], psds[:, :1])
                glitch = self.whitener(glitch[:, :1], psds[:, :1])
                return X, waveform, glitch, psd_X

            if injection == "waveform":
                waveform = self.whitener(waveform[:, :1], psds[:, :1])
                return X, waveform, psd_X

            if injection == "glitch":
                glitch = self.whitener(glitch[:, :1], psds[:, :1])
                return X, glitch, psd_X
                    
        else:
            ##parameter handling
            parameters = {}
            params["distance"] = params["distance"]/scale[:, 0, 0]
            parameters["injection"] = params

            #meta_data
            waveform_snr = self.compute_snr(waveform, psds)
            glitch_snr = self.compute_snr(glitch, psds)
            parameters["snrs"] = {"waveform":waveform_snr, "glitch": glitch_snr}
            parameters["jitters"] = jitters

            ## optional mmuting:
            if self.mute_waveform:
                waveform = torch.zeros_like(X)
            if self.mute_glitch:
                glitch = torch.zeros_like(X)

            ##inject and whiten
            X = X + waveform + glitch
            X_ = self.whitener(X, psds)
            waveform_ = self.whitener(waveform, psds)
            glitch_ = self.whitener(glitch, psds)
            psd_X_ = torch.zeros_like(X_)

            ## prepare unwhitened data
            crop = int(self.fduration / 2 * self.sample_rate)
            X = X
            waveform = waveform
            glitch = glitch
            psd_X = psd_X

            strain = (X, waveform, glitch, psd_X)
            whitened = (X_, waveform_, glitch_, psd_X_)
            others = (psds, parameters)

            return strain, whitened, others


    def compute_snr(self, S, psds, net=True, ifos=None):

        ifos = ifos or len(self.ifos)

        snrs = gw.compute_ifo_snr(
            S, psds, 
            sample_rate = self.sample_rate,
            highpass = self.f_min
        )

        if net:
            snrs = (snrs**2).sum(axis=-1) ** 0.5
        else:
            snrs = (snrs[:, 0])
        return snrs
        
    def rescale_snr(self, S, psds, snr_range, ifos=None):

        if snr_range is None:
            scale = torch.ones(S.shape[0]).view(1, 1, -1)
            return S, scale 

        snr = self.compute_snr(S, psds, net=False, ifos=ifos)
        
        target_snr = torch.empty_like(snr).uniform_(snr_range[0], snr_range[1])
        scale = target_snr / (snr + 1e-8)
        scale = scale[:, None, None]
        S = S * scale   

        return S, scale
        

    def setup(self, step="train", injection="waveform"):

        self.train_val_fnames()
        self.transforms()

        if step == "train":

            ## Generate Validation data
            X = self.background(
                self.val_fnames,
                batches = self.num_val_samples,
                batches_per_epoch = 1,
            )
            X = next(iter(X))
            X, injection, B = self.inject(X, step='train', injection=injection)

            self.val_batch = (
                X.to(self.device),
                injection.to(self.device),
                B.to(self.device)   
            )
            self.log(f"Generated {X.shape[0]} validation signals")

            self.waveform_gen = self.waveform_gen.to(self.device)
            self.glitch_gen = self.glitch_gen.to(self.device)
            self.spectral_density = self.spectral_density.to(self.device)

        if step == "test":

            X = self.background(
                self.val_fnames,
                batches = self.num_test_samples,
                batches_per_epoch = 1,
            )
            X = next(iter(X))

            (X, waveform, glitch, psd_X), (X_, waveform_, glitch_, psd_X_), (psds, params) = self.inject(X, step='test')

            self.test_batch = (X, waveform, glitch, psd_X), (X_, waveform_, glitch_, psd_X_), (psds, params)

            #self.test_X = X
            #self.test_waveform = waveform
            #self.test_glitch = glitch
            #self.test_psd = psds
            #self.test_background = psd_X
            #self.test_parameters = params
            #self.test_X_ = X_
            #self.test_waveform_ = waveform_
            #self.test_glitch_ = glitch_
            #self.log(f"Generated {X.shape[0]} testing signals")

    def train_dataloader(self):
        dataset = self.background(
            fnames = self.train_fnames,
            batches = self.batch_size,
            batches_per_epoch = self.batches_per_epoch,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=0, pin_memory=False
        )
        return dataloader

    def val_loader(self):
        val_dataset = TensorDataset(
            self.val_X,
            self.val_waveform,
            self.val_glitch,
            self.val_psd_X,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.num_val_samples,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.log("Validation dataset generated")
        return val_loader

    def test_loader(self):
        test_dataset = TensorDataset(
            self.test_X,
            self.test_waveform,
            self.test_glitch,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.num_test_samples,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.log("Testing dataset generated")
        return test_loader