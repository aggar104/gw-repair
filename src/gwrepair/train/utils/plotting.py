import matplotlib.pyplot as plt
import torch
from scipy.signal.windows import tukey

from gwrepair.train.utils.masking import mag_mask

def plot_stft_mag(ax, X, sample_rate=2048, duration=3, v = None, labels='both', f_min=20,):

    df =  (sample_rate/2) / (X.shape[-2] - 1)
    f_low = int(f_min // df)

    if ax is None:
        fig, ax = plt.subplots()

    if X.dim() == 4:
        mag = X[0, 0, f_low:]
    else:
        mag = (X[f_low:]).abs()

    if v is None:
        v = [None, None]

    
    F, TT = mag.shape[-2:]
    p = ax.imshow(
        mag,
        origin="lower",
        aspect="auto",
        vmin = v[0],
        vmax = v[1],
        extent=[0, duration, f_min, sample_rate/2]
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
    linestyles="solid",
    f_min=20
):

    df =  (sample_rate/2) / (mask.shape[-2] - 1)
    f_low = int(f_min // df)

    if ax is None:
        fig, ax = plt.subplots()

    if mask.dim() == 4:
        mag = (mask[0, 0, f_low:]).float()
    else:
        mag = (mask[f_low:]).float()

    
    F, TT = mag.shape
    p = ax.contour(
        mag,
        levels=levels,
        colors=c,
        linewidths=2.5,
        linestyles="solid",
        extent=[0, duration, f_min, sample_rate/2]
    )

    if labels == "both":
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
    elif labels == "x":
        ax.set_xlabel("Time [s]")
    elif labels == 'y':
        ax.set_ylabel("Frequency [Hz]")
    return p

def plot_timeseries(strains, ax=None, c=None, alpha=None, labels=None, sample_rate=2048, i=0):

    t = torch.arange(0, strains[0].shape[-1]/sample_rate, 1/sample_rate)

    num_strains = len(strains)

    if c is None:
        c = [c]*num_strains
    if alpha is None:
        alpha = [alpha]*num_strains
    if labels is None:
        label = [labels]*num_strains
    else:
        label = labels

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ret_ax = True
    else:
        ret_ax = False

    for j in range(num_strains):

        ax.plot(t, strains[j][i, 0], c=c[j], alpha=alpha[j], label=label[j])

    if labels is not None:
        ax.legend()

    if ret_ax:
        return ax

def plot_stfts(stft_, masks=None, ax=None, c=None, lw=None, styles=None, levels=None, duration=3, sample_rate=2048, v=None, axis_labels="both", i=0, overlap=True, f_min=20):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ret_ax = True
    else:
        ret_ax = False

    if stft_.dim() == 4:
        stft = stft_.clone()[i, 0]
    else:
        stft = stft_.clone()

    if masks is not None:
        if len(masks) == 2 and overlap:
            masks.append(masks[0].float()*masks[1].float())
            
        if c is None:
            c1, c2, c3 = "r", "g", "C1"
            c = [c1, c2, c3]
        if lw is None:
            lw = [3, 3, 4]
        if styles is None:
            styles = ["dotted", "dotted", "solids"]
        if levels is None:
            levels = [[0.5]]*3

    else:
        masks = []

    plot_stft_mag(ax, stft, v=v, labels=axis_labels, duration=duration, sample_rate=sample_rate, f_min=f_min)

    for j in range(len(masks)):

        mask = masks[j].clone()
        if mask.dim() == 4:
            mask = mask[i, 0]

        if mask.shape[-1] != stft.shape[-1]:

            crop = (mask.shape[-1] - stft.shape[-1]) // 2
            mask = mask[..., crop:-crop]
            
        plot_stft_mask(
            ax=ax, 
            mask = mask,
            sample_rate=sample_rate,
            duration=duration,
            levels=levels[j],
            labels='n',
            c=c[j],
            linewidth=lw[j],
            linestyles=styles[j],
            f_min=f_min,
            
        )

def morlet_cwt(x, fs=2048, fmin=20, fmax=512, nfreq=128, n_cycles=7, win=None):
    """
    Torch CWT using complex Morlet wavelets.

    x: [T] real tensor
    returns:
        cwt: [F, T] complex tensor
        freqs: [F]
        t: [T]
    """

    device = x.device
    x = x.float()
    T = x.numel()

    freqs = torch.linspace(fmin, fmax, nfreq, device=device)
    t = torch.arange(T, device=device) / fs

    if win is None:
        win = torch.from_numpy(tukey(x.shape[-1], 0.2))

    X = torch.fft.fft(x*win)

    cwt = []

    for f in freqs:
        sigma_t = n_cycles / (2 * torch.pi * f)

        half_width = int((5 * sigma_t * fs).item())
        tau = torch.arange(-half_width, half_width + 1, device=device) / fs

        wavelet = torch.exp(2j * torch.pi * f * tau)
        wavelet *= torch.exp(-0.5 * (tau / sigma_t) ** 2)

        wavelet = wavelet / torch.sqrt(torch.sum(torch.abs(wavelet) ** 2))

        pad = torch.zeros(T, dtype=torch.complex64, device=device)
        L = min(wavelet.numel(), T)
        pad[:L] = wavelet[:L]

        W = torch.fft.fft(torch.roll(pad, -L // 2))
        coef = torch.fft.ifft(X * torch.conj(W))

        cwt.append(coef)

    cwt = torch.stack(cwt, dim=0)

    return cwt, freqs, t


def plot_torch_cwt(x, fs=2048, fmin=20, fmax=512, nfreq=128, n_cycles=7, win=None):
    cwt, freqs, t = morlet_cwt(
        x, fs=fs, fmin=fmin, fmax=fmax, nfreq=nfreq, n_cycles=n_cycles, win=win
    )

    power = torch.abs(cwt).detach().cpu()
    freqs_cpu = freqs.detach().cpu()
    t_cpu = t.detach().cpu()

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t_cpu, freqs_cpu, (power), shading="auto")
    plt.yscale("log")
    plt.ylim(fmin, fmax)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="log10 |CWT|")
    plt.tight_layout()
    plt.show()

    return cwt, freqs, t

