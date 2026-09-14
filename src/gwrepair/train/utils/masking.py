import torch
import torch.nn.functional as F

def dilate_mask(mask: torch.Tensor, expand: int = 1):
    """
    mask: [B, 1, F, T] bool or 0/1 tensor
    returns: same shape, bool
    """
    x = mask.float()

    for _ in range(expand):
        x = F.max_pool2d(
            x,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    return x.bool()

def mag_mask(stft_injection, stft_B, z=0, expand=None, close=(3, 3)):

    noise_mean = stft_B.mean(dim=(2, 3), keepdims=True)
    noise_std = stft_B.std(dim=(2, 3), keepdims=True)

    eps = 1e-12
    mask = ((stft_injection - noise_mean) / (noise_std + eps)) > z

    if expand is not None:
        mask = dilate_mask(mask, expand)

    if close is not None:
        mask = close_mask(mask, close)
        
    return mask

def close_mask(mask, kernel_size=(3, 3)):
    """
    Morphological closing for mask of shape [B, C, F, T].
    """
    x = mask.float()

    kf, kt = kernel_size
    padding = (kf // 2, kt // 2)

    # dilation
    x = F.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    # erosion
    x = 1 - F.max_pool2d(
        1 - x,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    return x > 0.5