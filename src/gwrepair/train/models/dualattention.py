import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class AttentionGate(nn.Module):
    """
    Attention gate for U-Net skip connections.

    skip: encoder feature
    gate: decoder feature
    """

    def __init__(self, skip_ch, gate_ch, inter_ch=None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(skip_ch // 2, 1)

        self.skip_proj = nn.Conv2d(skip_ch, inter_ch, kernel_size=1)
        self.gate_proj = nn.Conv2d(gate_ch, inter_ch, kernel_size=1)

        self.psi = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(inter_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, skip, gate):
        if skip.shape[-2:] != gate.shape[-2:]:
            gate = F.interpolate(
                gate,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        attn = self.psi(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * attn


class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        use_attention=False,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2,
        )

        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionGate(skip_ch=skip_ch, gate_ch=out_ch)

        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.use_attention:
            skip = self.attn(skip, x)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()

        self.stem = ConvBlock(in_ch, channels[0])
        self.downs = nn.ModuleList(
            [
                DownBlock(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x):
        skips = []

        x = self.stem(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        return skips


class DualEncoderAttentionUNet(nn.Module):
    """
    Dual-encoder U-Net for gwrepair.

    Example use:
        glitch_model = DualEncoderAttentionUNet(
            h1_in_channels=1,
            ref_in_channels=1,
            out_channels=1,
            h1_channels=(32, 64, 128, 256),
            ref_channels=(16, 32, 64, 128),
            attention_skips=2,
        )

    attention_skips:
        0 = no attention
        1 = attention on first decoder skip
        2 = attention on first two decoder skips
        etc.

    Here "first decoder skip" means the highest-level / deepest skip used first
    by the decoder.
    """

    def __init__(
        self,
        h1_in_channels=1,
        ref_in_channels=1,
        out_channels=1,
        h1_channels=(32, 64, 128, 256),
        ref_channels=(16, 32, 64, 128),
        attention_skips=0,
        output_activation="sigmoid",
    ):
        super().__init__()

        assert len(h1_channels) == len(ref_channels)
        assert attention_skips >= 0

        self.output_activation = output_activation

        self.h1_encoder = Encoder(h1_in_channels, h1_channels)
        self.ref_encoder = Encoder(ref_in_channels, ref_channels)

        fused_channels = [
            h + r for h, r in zip(h1_channels, ref_channels)
        ]

        self.bottleneck = ConvBlock(
            fused_channels[-1],
            fused_channels[-1] * 2,
        )

        decoder_in = fused_channels[-1] * 2

        self.ups = nn.ModuleList()
        n_skips = len(fused_channels) - 1

        for i in range(n_skips):
            skip_idx = len(fused_channels) - 2 - i
            skip_ch = fused_channels[skip_idx]
            out_ch = skip_ch

            use_attention = i < attention_skips

            self.ups.append(
                UpBlock(
                    in_ch=decoder_in,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    use_attention=use_attention,
                )
            )

            decoder_in = out_ch

        self.head = nn.Conv2d(decoder_in, out_channels, kernel_size=1)

    def forward(self, h1, ref):
        """
        h1:
            [B, C_h1, F, T]

        ref:
            [B, C_ref, F, T]

        returns:
            mask [B, out_channels, F, T]
        """

        h1_skips = self.h1_encoder(h1)
        ref_skips = self.ref_encoder(ref)

        fused_skips = [
            torch.cat([h, r], dim=1)
            for h, r in zip(h1_skips, ref_skips)
        ]

        x = self.bottleneck(fused_skips[-1])

        decoder_skips = fused_skips[:-1][::-1]

        for up, skip in zip(self.ups, decoder_skips):
            x = up(x, skip)

        x = self.head(x)

        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.output_activation in [None, "none"]:
            return x
        else:
            raise ValueError(f"Unknown output_activation={self.output_activation}")