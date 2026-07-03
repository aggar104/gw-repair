import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ----------------------------
# Residual blocks
# ----------------------------

class ResBlock1D(nn.Module):
    def __init__(self, channels: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g = min(groups, channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(self.act(self.norm2(h)))
        return h


class Down1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.tconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.tconv(x)


# ----------------------------
# Residual 1D U-Net
# ----------------------------

class ResidualUNet1D(nn.Module):
    def __init__(
        self,
        c_in=3,          # det0, det1, det2
        c_out=1,         # predict glitch in detector 0
        base=64,
        depth=4,
        blocks_per_level=2,
        dropout=0.0,
    ):
        super().__init__()

        self.in_conv = nn.Conv1d(c_in, base, kernel_size=3, padding=1)

        enc_chs = [base * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()

        ch = base
        for i in range(depth):
            level_ch = enc_chs[i]
            if i > 0:
                self.downs.append(Down1D(ch, level_ch))
                ch = level_ch
            self.enc_blocks.append(
                nn.Sequential(*[ResBlock1D(ch, dropout=dropout) for _ in range(blocks_per_level)])
            )

        self.mid = nn.Sequential(
            ResBlock1D(ch, dropout=dropout),
            ResBlock1D(ch, dropout=dropout),
        )

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in reversed(range(depth - 1)):
            up_out = enc_chs[i]
            self.ups.append(Up1D(ch, up_out))
            ch = up_out
            self.dec_blocks.append(
                nn.Sequential(
                    *[ResBlock1D(2 * ch, dropout=dropout) for _ in range(blocks_per_level)],
                    nn.Conv1d(2 * ch, ch, kernel_size=1),
                )
            )

        self.out_conv = nn.Conv1d(base, c_out, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.in_conv(x)
        skips = []

        for i, blocks in enumerate(self.enc_blocks):
            h = blocks(h)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        h = self.mid(h)

        for up, dec in zip(self.ups, self.dec_blocks):
            h = up(h)
            skip = skips.pop(-2)
            if h.shape[-1] != skip.shape[-1]:
                L = min(h.shape[-1], skip.shape[-1])
                h = h[..., :L]
                skip = skip[..., :L]
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return self.out_conv(h)