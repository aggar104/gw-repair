import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """
    Classic Attention U-Net gate.

    x: skip feature from encoder
    g: gating feature from decoder
    """

    def __init__(self, skip_ch, gate_ch, inter_ch=None):
        super().__init__()

        if inter_ch is None:
            inter_ch = max(skip_ch // 2, 1)

        self.theta_x = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.phi_g = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False)

        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)

        a = self.relu(self.theta_x(x) + self.phi_g(g))
        a = self.psi(a)

        return x * a


class AttentionUNet2D(nn.Module):
    """
    Attention U-Net for STFT input.

    Input:
        stft_X: [B, 2, F, T]

    Output:
        y: [B, out_ch, F, T]

    Parameters:
        depth:
            Number of encoder/decoder levels.
        num_attention_gates:
            Number of skip connections that use attention gates.
            If 0, this becomes a regular U-Net.
            If depth, every skip is gated.
    """

    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        base_ch: int = 32,
        depth: int = 4,
        num_attention_gates: int = 4,
    ):
        super().__init__()

        assert depth >= 1
        assert 0 <= num_attention_gates <= depth

        self.depth = depth
        self.num_attention_gates = num_attention_gates

        channels = [base_ch * (2 ** i) for i in range(depth)]

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_ch

        for ch in channels:
            self.encoders.append(ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()

        decoder_in_ch = channels[-1] * 2

        for i in reversed(range(depth)):
            skip_ch = channels[i]

            self.upconvs.append(
                nn.ConvTranspose2d(
                    decoder_in_ch,
                    skip_ch,
                    kernel_size=2,
                    stride=2,
                )
            )

            use_attention = (depth - 1 - i) < num_attention_gates

            if use_attention:
                self.attention_gates.append(
                    AttentionGate(
                        skip_ch=skip_ch,
                        gate_ch=skip_ch,
                    )
                )
            else:
                self.attention_gates.append(nn.Identity())

            self.decoders.append(
                ConvBlock(
                    in_ch=skip_ch * 2,
                    out_ch=skip_ch,
                )
            )

            decoder_in_ch = skip_ch

        self.out = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, stft_X):
        """
        stft_X: [B, 2, F, T]
        """

        skips = []
        x = stft_X

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for upconv, gate, decoder, skip in zip(
            self.upconvs,
            self.attention_gates,
            self.decoders,
            reversed(skips),
        ):
            x = upconv(x)

            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if isinstance(gate, AttentionGate):
                skip = gate(skip, x)

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.out(x)