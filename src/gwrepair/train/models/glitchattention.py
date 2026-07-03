import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic blocks
# -------------------------

class ConvBlock2D(nn.Module):
    """
    Two-layer conv block for [B, C, F, T].
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock2D(nn.Module):
    """
    Conv block + 2x downsample.
    Returns:
        feat: features before pooling (skip)
        down: pooled features
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock2D(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class UpBlock2D(nn.Module):
    """
    Upsample decoder feature, optionally gate skip, concat, then conv.
    """
    def __init__(self, in_ch, skip_ch, out_ch, attention_gate=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attention_gate = attention_gate
        self.conv = ConvBlock2D(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle odd shapes after pooling / upsampling
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        if self.attention_gate is not None:
            skip = self.attention_gate(g=x, x=skip)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# -------------------------
# Attention gate
# -------------------------

class AttentionGate2D(nn.Module):
    """
    Standard additive attention gate for U-Net skips.

    g: decoder/gating feature   [B, g_ch, F, T]
    x: encoder/skip feature     [B, x_ch, F, T]

    Returns:
        gated skip feature with same shape as x
    """
    def __init__(self, g_ch, x_ch, inter_ch=None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(1, min(g_ch, x_ch) // 2)

        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        attn = self.psi(self.W_g(g) + self.W_x(x))   # [B, 1, F, T]
        return x * attn


# -------------------------
# Flexible attention U-Net
# -------------------------

class GlitchAttentionUNet(nn.Module):
    """
    U-Net for a single glitch-channel spectrogram input.

    Input:
        x: [B, in_ch, F, T]

    Output dict:
        mask_logits: [B, out_ch, F, T]
        mask_prob:   [B, out_ch, F, T]
        masked_input:[B, out_ch, F, T] if out_ch == in_ch or broadcastable
        attention_maps: dict of attention maps if requested

    Attention configuration:
        n_attention_gates = 0 -> no attention
        n_attention_gates = 1 -> bottleneck only
        n_attention_gates = 2 -> bottleneck + deepest skip
        n_attention_gates = 3 -> bottleneck + 2 deepest skips
        n_attention_gates = 4 -> bottleneck + 3 deepest skips
        ...
    """
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        channels=(32, 64, 128, 256),
        n_attention_gates=2,
        use_bottleneck_attention=True,
    ):
        super().__init__()

        if len(channels) < 2:
            raise ValueError("channels must have at least 2 entries")
        if n_attention_gates < 0:
            raise ValueError("n_attention_gates must be >= 0")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.channels = channels
        self.n_attention_gates = n_attention_gates
        self.use_bottleneck_attention = use_bottleneck_attention

        c1, c2, c3, c4 = channels

        # Encoder
        self.down1 = DownBlock2D(in_ch, c1)
        self.down2 = DownBlock2D(c1, c2)
        self.down3 = DownBlock2D(c2, c3)
        self.down4 = DownBlock2D(c3, c4)

        # Bottleneck
        self.bottleneck = ConvBlock2D(c4, c4 * 2)

        # Optional bottleneck attention (self-style reweighting)
        # Simple channel/spatial recalibration block.
        self.bottleneck_attn = BottleneckAttention2D(c4 * 2) if self._use_bottleneck_attention() else None

        # Decide which skip levels get attention
        # Skip levels available in decoder: up4, up3, up2, up1
        # We apply from deepest to shallowest: up4 -> up3 -> up2 -> up1
        skip_gate_names = self._choose_skip_gates()

        attn_up4 = AttentionGate2D(g_ch=c4, x_ch=c4) if "up4" in skip_gate_names else None
        attn_up3 = AttentionGate2D(g_ch=c3, x_ch=c3) if "up3" in skip_gate_names else None
        attn_up2 = AttentionGate2D(g_ch=c2, x_ch=c2) if "up2" in skip_gate_names else None
        attn_up1 = AttentionGate2D(g_ch=c1, x_ch=c1) if "up1" in skip_gate_names else None

        # Decoder
        self.up4 = UpBlock2D(in_ch=c4 * 2, skip_ch=c4, out_ch=c4, attention_gate=attn_up4)
        self.up3 = UpBlock2D(in_ch=c4,     skip_ch=c3, out_ch=c3, attention_gate=attn_up3)
        self.up2 = UpBlock2D(in_ch=c3,     skip_ch=c2, out_ch=c2, attention_gate=attn_up2)
        self.up1 = UpBlock2D(in_ch=c2,     skip_ch=c1, out_ch=c1, attention_gate=attn_up1)

        # Output head
        self.head = nn.Conv2d(c1, out_ch, kernel_size=1)

    def _use_bottleneck_attention(self):
        return self.use_bottleneck_attention and self.n_attention_gates >= 1

    def _choose_skip_gates(self):
        """
        n_attention_gates counts bottleneck first, then deepest-to-shallowest skips.
        """
        remaining = self.n_attention_gates

        if self.use_bottleneck_attention and remaining >= 1:
            remaining -= 1

        ordered_skips = ["up4", "up3", "up2", "up1"]
        return set(ordered_skips[:remaining])

    def forward(self, x):
        # Encoder
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)
        if self.bottleneck_attn is not None:
            x = self.bottleneck_attn(x)

        # Decoder
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        mask_logits = self.head(x)
        mask_prob = torch.sigmoid(mask_logits)

        out = {
            "mask_logits": mask_logits,
            "mask_prob": mask_prob,
        }

        # Helpful if you want predicted glitch STFT directly
        if self.out_ch == self.in_ch:
            out["masked_input"] = mask_prob * x.new_zeros(1)  # placeholder overwritten below

        return out


class BottleneckAttention2D(nn.Module):
    """
    Lightweight bottleneck attention:
    channel attention + spatial attention.
    Not strict CBAM, but same spirit and simple.
    """
    def __init__(self, ch, reduction=8):
        super().__init__()
        hidden = max(1, ch // reduction)

        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, kernel_size=1, bias=True),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        ch_attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ch_attn

        # Spatial attention
        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp, _ = x.max(dim=1, keepdim=True)
        sp_attn = self.spatial(torch.cat([avg_sp, max_sp], dim=1))
        x = x * sp_attn

        return x