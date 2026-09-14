import numpy as np
import torch
import torch.nn.functional as F

def dilate_mask_1d(mask: torch.Tensor, expand: int = 100):
    """
    Dilate a 1D boolean mask by `expand` samples on both sides.

    Args:
        mask: [T] bool tensor
        expand: number of samples to expand on each side

    Returns:
        [T] bool tensor
    """
    if expand <= 0:
        return mask.bool()

    x = mask.float()[None, None, :]  # [1, 1, T]

    x = F.max_pool1d(
        x,
        kernel_size=2 * expand + 1,
        stride=1,
        padding=expand,
    )

    return x[0, 0].bool()

def find_best_t0(x, xline, t0, search_width):

    loss = []
    context = x.shape[-1]
    for i in range(search_width):
    
        x_repl = xline[t0 + i : int(t0 + context + i)]
    
        loss.append(
            ((x.real - x_repl.real)**2).sum().item() + 
            ((x.imag - x_repl.imag)**2).sum().item()
        )
    
    i = np.argmin(loss)
    return i

def choose_valid_t0(
    t_end: int,
    dt: int,
    gs: int,
    ge: int,
    ws: int,
    we: int,
    context: int,
    search_width: int,
    n_tries: int = 10_000,
    generator: torch.Generator | None = None,
):
    """
    Choose t0 such that [t0, t0 + dt] is inside [t_start, t_end]
    and does not overlap glitch [gt, gt+dgt] or waveform [wt, wt+dwt].
    """

    gs, ws = gs - context, ws - context
    ge, se = ge + context, we + context

    for _ in range(n_tries):
        t0 = torch.randint(t_end, (1, )).item()

        seg_start = t0
        seg_end = t0 + dt

        overlaps_glitch = seg_start < ge and seg_end > gs
        overlaps_wave = seg_start < we and seg_end > ws

        if not overlaps_glitch and not overlaps_wave and t0 + 2*context + search_width + (ge-gs+10) < t_end:
            return t0

    raise RuntimeError("Could not find a valid t0. Try smaller dt or check regions.")

def smootherstep(s):
    return 6*s**5 - 15*s**4 + 10*s**3

def roll_replace_complex_bin(xline, gline, wline, context=30, search_width=150, smooth=None):
    
    if smooth is None:
        smooth = smootherstep(torch.linspace(0, 1, context))
    
    #gline = dilate_mask_1d(gline.float(), 3).float()
    #wline = dilate_mask_1d(wline.float(), 3).float()

    gline = gline.abs()
    wline = wline.abs()
    line = xline.clone()
    
    
    gs = int(gline.argmax().item())
    ws = int(wline.argmax().item())
    ge = int(gs + gline.sum().item())
    we = int(ws + wline.sum().item())
    gw = ge-gs

    loss = []
    t0 = choose_valid_t0(gline.shape[-1], search_width, gs, ge, ws, we, context+1, search_width)

    x_left = xline[gs-context: gs]
    x_right = xline[ge+1: ge+context+1]
    
    if ws == 0 or ws > gs:
        t0 += find_best_t0(x_left, xline, t0, search_width)
    elif ws < gs: 
        t0 += find_best_t0(x_right, xline, t0 + context + gw + 1, search_width)# - (context + gw + 1)

    x_repl_left = xline[t0: t0+context]
    x_repl_center = xline[t0+context: t0+context+gw+1]
    x_repl_right = xline[t0+context + gw+1: t0+2*context+gw+1]

    left = x_left * (1-smooth) + x_repl_left*(smooth)
    right = x_right * (smooth) + x_repl_right*(1-smooth)

    line[gs-context: gs] = left
    line[gs:ge+1] = x_repl_center
    line[ge+1:ge+context+1] = right

    #check if no overlap
    glitch_start = gs - context
    glitch_end   = ge + context
    
    overlap_ = max(ws, glitch_start) <= min(we, glitch_end)
    
    if not overlap_:
        return line, torch.zeros_like(xline)

    elif gs - context < ws and ge + context > ws and gs < ws:
        # left

        xline_ = xline.clone()
        
        keep_region = ws - context
        merge = torch.zeros_like(xline)
        merge[keep_region: ws] = smooth
        merge[ws:] = torch.ones_like(merge[ws:])

        out = xline_*merge + line*(1-merge)

        overlap = torch.zeros_like(xline)
        overlap[keep_region: ge] = torch.ones_like(overlap[keep_region: ge])

        return out, overlap

    elif gs - context < we and ge + context> we and ge > we:
        # right

        xline_ = xline.clone()
        
        keep_region = we + context
        merge = torch.zeros_like(xline)
        merge[we+1: keep_region+1] = 1 - smooth
        merge[:we+1] = torch.ones_like(merge[:we+1])

        out = xline_*merge + line*(1-merge)

        overlap = torch.zeros_like(xline)
        overlap[gs+1: keep_region+1] = torch.ones_like(overlap[gs+1: keep_region+1])
        
        return out, overlap

    elif gs - context < ws and ge+context > we and gs < ws and ge > we:
        # both

        xline_ = xline.clone()
        
        keep_region_l = ws - context
        merge_l = torch.zeros_like(xline)
        merge_l[keep_region_l: ws] = smooth
        merge_l[ws:] = torch.ones_like(merge_l[ws:])
        
        keep_region_r = we + context
        merge_r = torch.zeros_like(xline)
        merge_r[we+1: keep_region_r+1] = 1 - smooth
        merge_r[:we+1] = torch.ones_like(merge_r[:we+1])

        merge = merge_l + merge_r

        out = xline_*merge + line*(1-merge)

        overlap = torch.zeros_like(xline)
        overlap[keep_region_l: keep_region_r] = torch.ones_like(overlap[keep_region_l: keep_region_r])
        return out, overlap

    else:
        print("Couldn't find mask replacement")
        return xline, torch.zeros_like(xline)

def roll_replace_complex_stft(
    stft_X: torch.Tensor,
    glitch_mask: torch.Tensor,
    waveform_mask: torch.Tensor | None = None,
    context: int = 10,
):
    """
    stft_X: [B, C, F, T] complex
    glitch_mask: [B, 1 or C, F, T] bool
    waveform_mask: [B, 1 or C, F, T] bool or None
    """

    y = stft_X.clone()
    overlap = torch.zeros_like(stft_X)
    B, C, FF, T = stft_X.shape

    for b in range(B):
        for f in range(FF):
            gm = glitch_mask[b, 0, f]
            wm = None
            if waveform_mask is not None:
                wm = waveform_mask[b, 0, f]

            if gm.any():
                yf, overf = roll_replace_complex_bin(stft_X[b, 0, f], gm, wm, context=context)
                y[b, 0, f] = yf
                overlap[b, 0, f] = overf

    return y, overlap