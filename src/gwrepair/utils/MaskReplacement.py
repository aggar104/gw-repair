import numpy as np
import torch
from scipy.ndimage import label

## 2d mask tools

def merge_nearby_glitch_masks(mask: torch.Tensor, max_time_gap: int = 15):
    """
    Merge mask regions separated by <= max_time_gap along time.

    Each frequency bin is treated independently, so glitches separated
    in frequency are NOT merged.

    Args:
        mask: bool tensor [..., F, T], e.g. [B, C, F, T]
        max_time_gap: maximum temporal gap to fill

    Returns:
        merged mask with same shape
    """
    mask = mask.bool()
    T = mask.shape[-1]

    t = torch.arange(T, device=mask.device)

    # Most recent masked point to the left
    left = torch.where(mask, t, -T)
    left = torch.cummax(left, dim=-1).values

    # Nearest masked point to the right
    right = torch.where(mask, t, 2 * T)
    right = torch.flip(
        torch.cummin(torch.flip(right, dims=[-1]), dim=-1).values,
        dims=[-1],
    )

    gap = right - left - 1

    # Fill only points lying between two masks with a small enough gap
    fill = (
        ~mask
        & (left >= 0)
        & (right < T)
        & (gap <= max_time_gap)
    )

    return mask | fill

def cluster_glitches(mask: torch.Tensor):
    """
    Separate a 2D [F, T] glitch mask into connected components.

    Returns:
        List of [F, T] boolean masks, one per glitch.
    """
    device = mask.device

    labels, n = label(
        mask.bool().cpu().numpy(),
        structure=np.ones((3, 3)),
    )

    components = [
        torch.from_numpy(labels == i).to(device)
        for i in range(1, n + 1)
    ]

    return components

## 1D mask tools

class Slice:
    def __init__(
        self,
        mask: torch.Tensor,
        context_len: int = 15,
        context: bool = True,
    ):
        self.context_len = context_len
        self.size = len(mask)

        idx = mask.bool().nonzero(as_tuple=True)[0]

        if len(idx) == 0:
            self.exist = False
            self.width = 0
            return

        self.exist = True

        # stop is exclusive
        self.index = slice(
            idx[0].item(),
            idx[-1].item() + 1,
        )

        self.width = self.index.stop - self.index.start

        if context:
            self.make_context()
            self.start = self.context.start
            self.stop = self.context.stop
        else:
            self.start = self.index.start
            self.stop = self.index.stop

    def make_context(self):
        self.context = self.check_edge(
            slice(
                self.index.start - self.context_len,
                self.index.stop + self.context_len,
            )
        )

        self.context_width = (
            self.context.stop - self.context.start
        )

        self.left = slice(
            self.context.start,
            self.index.start,
        )
        self.left_width = self.left.stop - self.left.start
        self.left_exist = self.left_width >= self.context_len / 3

        self.right = slice(
            self.index.stop,
            self.context.stop,
        )
        self.right_width = self.right.stop - self.right.start
        self.right_exist = self.right_width >= self.context_len / 3

    def check_edge(self, index):
        return slice(
            max(index.start, 0),
            min(index.stop, self.size),
        )


class MaskHandler1D:
    def __init__(
        self,
        context_len: int = 15,
        max_search: int = 150,
        event_id: int = None,
        f_bin: int = None,
        plot: bool = False,
        smooth: torch.Tensor | None = None,
    ):
        self.context_len = context_len
        self.max_search = max_search
        self.event_id = event_id
        self.f_bin = f_bin
        self.plot = plot

        self.smooth = (
            smooth
            if smooth is not None
            else smoothstep(context_len)
        )

    def handle(
        self,
        mask_lines: tuple[torch.Tensor, torch.Tensor],
        data_lines: tuple[torch.Tensor, torch.Tensor],
    ):
        x, b = data_lines

        glitch = Slice(
            mask=mask_lines[0],
            context_len=self.context_len,
            context=True,
        )

        if not glitch.exist:
            return x

        waveform = Slice(
            mask=mask_lines[1],
            context_len=self.context_len,
            context=False,
        )

        sides = self.determine_replacement_side(
            glitch,
            waveform,
        )

        if sides == [False, False]:
            return x

        # enough background to generate max_search candidates
        b = b[-(self.max_search + glitch.context_width):]

        replacement = self.find_best_replacement(
            x,
            b,
            glitch,
            sides,
        )

        replaced = self.soft_replace(
            x,
            replacement,
            glitch,
        )

        # Put waveform region back after glitch replacement
        if self.preserve:
            waveform.make_context()

            replaced = self.soft_replace(
                replaced,
                x[waveform.context],
                waveform,
            )

        return replaced

    def determine_replacement_side(
        self,
        glitch,
        waveform,
        print_status=False,
    ):
        status = []
        sides = [True, True]
        preserve = False

        if not waveform.exist:
            status.append("No waveform")

        else:
            if (
                glitch.start <= waveform.start
                and glitch.stop >= waveform.stop
            ):
                preserve = True
                status.append("glitch covers waveform completely")

            elif (
                glitch.start >= waveform.start
                and glitch.stop <= waveform.stop
            ):
                sides = [False, False]
                status.append("waveform covers glitch completely")

            elif (
                glitch.start <= waveform.start < glitch.stop
            ):
                sides[1] = False
                preserve = True
                status.append("right overlap")

            elif (
                glitch.start < waveform.stop <= glitch.stop
            ):
                sides[0] = False
                preserve = True
                status.append("left overlap")

            else:
                # No overlap
                status.append("no overlap")

        if not glitch.left_exist:
            sides[0] = False
            status.append("glitch on left edge")

        if not glitch.right_exist:
            sides[1] = False
            status.append("glitch on right edge")

        self.preserve = preserve

        if print_status or sides == [False, False]:
            print(
                f"Event: {self.event_id}, "
                f"f: {self.f_bin}: {status}"
            )

        return sides

    def find_best_replacement(
        self,
        x,
        b,
        glitch,
        sides,
    ):
        b_candidates = b.unfold(
            dimension=0,
            size=glitch.context_width,
            step=1,
        )

        losses = []

        if sides[0]:
            left_target = x[glitch.left]

            candidate_left = (
                b_candidates[:, :glitch.left_width]
            )

            losses.append(
                (candidate_left - left_target)
                .abs()
                .square()
                .mean(dim=1)
            )

        if sides[1]:
            right_target = x[glitch.right]

            candidate_right = (
                b_candidates[:, -glitch.right_width:]
            )

            losses.append(
                (candidate_right - right_target)
                .abs()
                .square()
                .mean(dim=1)
            )

        losses = torch.stack(losses).sum(dim=0)

        best_idx = losses.argmin()

        return b_candidates[best_idx]

    def soft_replace(
        self,
        original,
        replacement,
        slice_,
    ):
        window = torch.ones_like(
            replacement,
            dtype=replacement.real.dtype
            if replacement.is_complex()
            else replacement.dtype,
        )

        if slice_.left_exist:
            n = slice_.left_width
            window[:n] = self.smooth[-n:]

        if slice_.right_exist:
            n = slice_.right_width
            window[-n:] = 1 - self.smooth[:n]

        result = original.clone()

        result[slice_.context] = (
            original[slice_.context] * (1 - window)
            + replacement * window
        )

        return result

## 1D mask loop

def smoothstep(N):
    s = torch.linspace(0, 1, N)
    return 6*s**5 - 15*s**4 + 10*s**3

def ReplaceSTFT(
    stft_X: torch.Tensor,
    stft_B: torch.Tensor,
    glitch_mask: torch.Tensor,
    waveform_mask: torch.Tensor | None = None,
    context: int = 15,
    max_search: int = 250,
    split_complex: bool = False
):
    """
    stft_X: [B, C, F, T] complex
    glitch_mask: [B, 1 or C, F, T] bool
    waveform_mask: [B, 1 or C, F, T] bool or None
    """

    y = stft_X.clone()
    B, C, FF, T = stft_X.shape
    smooth = smoothstep(context)

    g_mask = merge_nearby_glitch_masks(glitch_mask, context)

    handler = MaskHandler1D(
        context_len = context,
        max_search = max_search,
        event_id = 0,
        f_bin = None,
        smooth = smooth,
    )

    for b in range(B):
        g_masks = cluster_glitches(g_mask[b, 0])
        for masks in g_masks:
            for f in range(FF):
                gm = masks[f]
                wm = None
                if waveform_mask is not None:
                    wm = waveform_mask[b, 0, f]
    
                mask_lines = [gm, wm]
                if gm.any():
                    if split_complex:
                        data_lines_real = [y[b, 0, f].real, stft_B[b, 0, f].real]
                        data_lines_imag = [y[b, 0, f].imag, stft_B[b, 0, f].imag]
                        y_real = handler.handle(
                            mask_lines,
                            data_lines_real,
                        )
                        y_imag = handler.handle(
                            mask_lines,
                            data_lines_imag,
                        )
                        y[b, 0, f] = torch.complex(y_real, y_imag)
                    else:
                        data_lines = [y[b, 0, f], stft_B[b, 0, f]]
                        y_ = handler.handle(
                            mask_lines,
                            data_lines,
                        )
                        y[b, 0, f] = y_
                        
    return y




