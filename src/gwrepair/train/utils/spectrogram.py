import torch
import matplotlib.pyplot as plt

from gwrepair.train.utils.masking import mag_mask
from gwrepair.train.data.dataset import RepairDataset

class stft_module(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop: int,
        win: int,
        mask_expand: int = 3,
        data_module: RepairDataset | None = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.mask_expand = mask_expand
        self.register_buffer('window', torch.hann_window(win))
        self.data_module = data_module

    def stft(self, x: torch.tensor, return_phase=False, ret_complex=False):

        flattened = False
        if x.dim() == 3:
            N, C, T = x.shape
            x = x.reshape(N * C, T)
            flattened = True

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )


        if flattened:
            F, TT = X.shape[-2], X.shape[-1]
            X = X.reshape(N, C, F, TT)

        if ret_complex:
            return X

        if return_phase:
            return X.abs(), torch.angle(X)
        else:
            return X.abs()

    def istft(
        self, 
        X: torch.tensor,
        phase: torch.tensor,
    ):
        phase = phase[:, 0:X.shape[1]]
        X = X * torch.exp(1j * phase)

        flattened=False
        if X.dim() == 4:
            N, C, F, TT = X.shape
            X = X.reshape(N*C, F, TT)
            phase = phase.reshape(N*C, F, TT)
            flattened = True
    
        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=self.window,
            length=None,
            return_complex=False,
        )

        if flattened:
            T = x.shape[-1]
            x = x.reshape(N, C, T)
        return x

    @torch.no_grad()
    def forward(self, batch, step="train", injection="both", expand=None):

        if step == "train" or step == "validate":

            if self.data_module is None or step == "validate":
                ts_X, ts_injection, ts_B = batch
            else:
                if injection == "both":
                    assert "Can't use [both] injections for training"
                    
                ts_X, ts_injection, ts_B = self.data_module.inject(
                    X = batch,
                    step = "train",
                    injection = injection
                )

            ts_B = ts_B[:, :1]
            ts_injection = ts_injection[:, :1]
            
            stft_X = self.stft(ts_X)
            stft_B = self.stft(ts_B)
            stft_injection = self.stft(ts_injection)

            stft_mask = mag_mask(stft_injection, stft_B, expand=expand)

            return stft_X, stft_mask


        if step == "test":

            strain, whitened, _ = batch

            ##strain stfts
            ts_signal, ts_waveform, ts_glitch, ts_background = strain

            ts_signal = ts_signal
            
            stft_signal, phase = self.stft(ts_signal, return_phase=True)
            stft_waveform = self.stft(ts_waveform)
            stft_glitch = self.stft(ts_glitch)
            stft_background = self.stft(ts_background)

            signal = (stft_signal, stft_background, phase)
            injections = (stft_waveform, stft_glitch)
            Strain = (signal, injections)

            ## whitened_stfts
            ##strain stfts
            ts_signal_, ts_waveform_, ts_glitch_, ts_background_ = whitened
            
            stft_signal_ = self.stft(ts_signal_)
            stft_waveform_ = self.stft(ts_waveform_)
            stft_glitch_ = self.stft(ts_glitch_)
            stft_background_ = self.stft(ts_background_)

            stft_waveform_mask_ = mag_mask(stft_waveform_, stft_background_, self.mask_expand)
            stft_glitch_mask_ = mag_mask(stft_glitch_, stft_background_, self.mask_expand)

            signal_ = (stft_signal_,)
            injections_ = (stft_waveform_, stft_glitch_, stft_waveform_mask_, stft_glitch_mask_)
            Strain_ = (signal_, injections_)

            return Strain, Strain_, _

        if step == "predict":

            ts_signal, ts_background, psds = batch
            
            stft_signal, phase = self.stft(ts_signal, return_phase=True)
            stft_background = self.stft(ts_background).abs()
    
            return stft_signal, stft_background, psds, phase

    

    

    

    


