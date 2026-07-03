import matplotlib.pyplot as plt
import torch

from gwrepair.train.utils.masking import mag_mask

def plot_stft_mag(ax, X, sample_rate=2048, duration=3, v = [None, None], labels='both'):

    if ax is None:
        fig, ax = plt.subplots()

    if X.dim() == 4:
        mag = X[0, 0]
    else:
        mag = (X).abs()

    
    F, TT = mag.shape[-2:]
    p = ax.imshow(
        mag,
        origin="lower",
        aspect="auto",
        vmin = v[0],
        vmax = v[1],
        extent=[0, duration, 0, sample_rate/2]
    )

    if labels == "both":
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
    elif labels == "x":
        ax.set_xlabel("Time [s]")
    elif labels == 'y':
        ax.set_ylabel("Frequency [Hz]")
    return p


def plot_stft_mask(
    ax,
    mask, 
    sample_rate=2048, 
    duration=3, 
    levels=[0.5], 
    labels='both', 
    c='C0', 
    linewidth=2.5, 
    linestyles="solid"
):

    if ax is None:
        fig, ax = plt.subplots()

    if mask.dim() == 4:
        mag = mask[0, 0].float()
    else:
        mag = (mask).float()

    
    F, TT = mag.shape[-2:]
    p = ax.contour(
        mag,
        levels=levels,
        colors=c,
        linewidths=2.5,
        linestyles="solid",
        extent=[0, duration, 0, sample_rate/2]
    )

    if labels == "both":
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
    elif labels == "x":
        ax.set_xlabel("Time [s]")
    elif labels == 'y':
        ax.set_ylabel("Frequency [Hz]")
    return p

