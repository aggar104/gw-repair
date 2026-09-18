import torch
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
import matplotlib.pyplot as plt

from gwrepair.train.models import AttentionUNet2D
from gwrepair.utils import ReadConfigs
from gwrepair.train.data.dataset import RepairDataset
from gwrepair.train.utils import stft_module
from gwrepair.utils.MaskReplacement import ReplaceSTFT

@dataclass
class TrainConfig:
    lr: float = 2e-4
    epochs: int = 50
    grad_clip: float = 1.0
    amp: bool = True
    ckpt_save_epoch: int = 5
    model: AttentionUNet2D = AttentionUNet2D(num_attention_gates=3)
    ckpts_path: Path = "../train/checkpoints"


def load_models(cfg=TrainConfig):
    
    glitch_model = deepcopy(cfg.model)
    glitch_checkpoint = torch.load(
        Path(cfg.ckpts_path) / "glitch.ckpt",
        map_location='cpu',
    )
    glitch_model.load_state_dict(glitch_checkpoint["model_state_dict"])
    _ = glitch_model.eval()
    
    waveform_model = deepcopy(cfg.model)
    waveform_checkpoint = torch.load(
        Path(cfg.ckpts_path) / "waveform.ckpt",
        map_location='cpu',
    )
    waveform_model.load_state_dict(waveform_checkpoint["model_state_dict"])
    _ = waveform_model.eval()

    return glitch_model, waveform_model

def data_modules(
    dataset: RepairDataset | None = None,
    spectogram: stft_module | None = None,
):
    dataset = dataset or ReadConfigs()
    dataset.transforms()

    if spectogram is None:
        spectogram = stft_module(n_fft = 256, hop = 16, win = 256)

    return dataset, spectogram

def repair(
    X: torch.tensor,
    psd_X: torch.tensor,
    ckpts_path: Path | None = None,
    cfg: TrainConfig = TrainConfig,
    dataset: RepairDataset | None = None,
    spectogram: stft_module | None = None,
    f_min: float = 20,
    sample_rate: float = 2048,
    return_whitned: bool = True,
    context=20,
    max_search: int = 250,
    split_complex: bool = False,
    plot = None,
):
    ## modules ##
    dataset, spectogram = data_modules(dataset, spectogram)
    if ckpts_path is not None:
        cfg.ckpts_path = ckpts_path
    glitch_model, waveform_model = load_models(cfg)
    
    ## transorm data ##
    psds = dataset.spectral_density(psd_X)
    B = psd_X[..., -X.shape[-1]:]
    stft_B = spectogram.stft(B, ret_complex=True)
    
    X_ = dataset.whitener(X, psds)
    stft_X = spectogram.stft(X, ret_complex=True)
    stft_X_ = spectogram.stft(X_).abs()

    ## predict mask ##
    device = torch.device('cpu')
    crop = (stft_X.shape[-1] - stft_X_.shape[-1]) // 2
    
    with torch.no_grad():
        with torch.amp.autocast(
            "cuda",
            enabled=(True and device.type == "cuda")
        ):
    
            glitch_mask_ = glitch_model(stft_X_)
            waveform_mask_ = waveform_model(stft_X_)

    threshold = 0.5
    glitch_mask_ = (glitch_mask_ >= threshold).float()
    waveform_mask_ = (waveform_mask_ >= threshold).float()

    ## remove > f_min mask and pad
    df = sample_rate // spectogram.n_fft
    f_low = int(f_min // df)
    glitch_mask_[:, :, :f_low] = torch.zeros_like(glitch_mask_[:, :, :f_low])
    waveform_mask_[:, :, :f_low] = torch.zeros_like(glitch_mask_[:, :, :f_low])


    glitch_mask = torch.zeros_like(stft_X)[:, :1]
    waveform_mask = torch.zeros_like(stft_X)[:, :1]
    glitch_mask[..., crop:-crop] = glitch_mask_
    waveform_mask[..., crop:-crop] = waveform_mask_

    ## replace mask
    pred_stft = ReplaceSTFT(
        stft_X,
        stft_B,
        glitch_mask,
        waveform_mask,
        context = context,
        max_search = max_search,
        split_complex = split_complex,
    )

    ## transform prediction
    ts_pred = spectogram.istft(pred_stft.abs(), torch.angle(pred_stft))
    
    if return_whitned:
        ts_pred_ = dataset.whitener(ts_pred, psds)
        stft_pred_ = spectogram.stft(ts_pred_)
        return (ts_pred, ts_pred_, stft_pred_), (glitch_mask, waveform_mask)
        
    else:
        return ts_pred, (glitch_mask, waveform_mask)

def repair_whitened(
    X: torch.tensor,
    psd_X: torch.tensor,
    ckpts_path: Path | None = None,
    cfg: TrainConfig = TrainConfig,
    dataset: RepairDataset | None = None,
    spectogram: stft_module | None = None,
    f_min: float = 20,
    sample_rate: float = 2048,
    return_whitned: bool = True,
    context=20,
    max_search: int = 250,
    split_complex: bool = False,
    plot = None,
):
    ## modules ##
    dataset, spectogram = data_modules(dataset, spectogram)
    if ckpts_path is not None:
        cfg.ckpts_path = ckpts_path
    glitch_model, waveform_model = load_models(cfg)
    
    ## transorm data ##
    B = psd_X[..., -X.shape[-1]:]
    stft_B = spectogram.stft(B, ret_complex=True)
    stft_X = spectogram.stft(X, ret_complex=True)

    ## predict mask ##
    device = torch.device('cpu')
    
    with torch.no_grad():
        with torch.amp.autocast(
            "cuda",
            enabled=(True and device.type == "cuda")
        ):
    
            glitch_mask_ = glitch_model(stft_X.abs())
            waveform_mask_ = waveform_model(stft_X.abs())

    threshold = 0.5
    glitch_mask_ = (glitch_mask_ >= threshold).float()
    waveform_mask_ = (waveform_mask_ >= threshold).float()

    ## remove > f_min mask and pad
    df = sample_rate // spectogram.n_fft
    f_low = int(f_min // df)
    glitch_mask_[:, :, :f_low] = torch.zeros_like(glitch_mask_[:, :, :f_low])
    waveform_mask_[:, :, :f_low] = torch.zeros_like(glitch_mask_[:, :, :f_low])

    ## replace mask
    pred_stft = ReplaceSTFT(
        stft_X,
        stft_B,
        glitch_mask_,
        waveform_mask_,
        context = context,
        max_search = max_search,
        split_complex = split_complex,
    )

    ## transform prediction
    ts_pred = spectogram.istft(pred_stft.abs(), torch.angle(pred_stft))
    
        
    return ts_pred, (glitch_mask_, waveform_mask_)