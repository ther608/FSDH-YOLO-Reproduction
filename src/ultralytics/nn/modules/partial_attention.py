# Ultralytics AGPL-3.0 License
"""
Partial Attention Modules for YOLO11 — light-weight plug-and-play blocks.

Adapted from the official PartialNet implementation (haiduo/PartialNet, AAAI 2026).
Paper: "Partial Channel Network: Compute Fewer, Perform Better"
        https://arxiv.org/abs/2502.01303

Simplified for Ultralytics YOLO — pure PyTorch, no timm / mmcv dependency.

Modules:
    PATChannelAttention  — Gaussian-enhanced channel attention (mean + std SE)
    PATSpatialAttention   — 1×1 Conv spatial attention with HardSigmoid
    PATConv               — Partial Attention Convolution (split-channel conv + attn)
    PATBlockLite          — Lightweight PATConv + MLP + PATsp residual block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PATChannelAttention",
    "PATSpatialAttention",
    "PATConv",
    "PATBlockLite",
    "SELayer",
]


# ============================================================================
# Helper: HardSigmoid (compatible with earlier PyTorch versions)
# ============================================================================

def hard_sigmoid(x, inplace: bool = False):
    return F.relu6(x + 3.0, inplace=inplace) / 6.0


# ============================================================================
# PATChannelAttention — Gaussian-enhanced SE (mean + std statistics)
# ============================================================================

class PATChannelAttention(nn.Module):
    """Partial Channel Attention with Gaussian statistics (mean + std).

    Unlike standard SE which uses only global average pooling, PAT_ch uses both
    mean and standard deviation per channel, providing a richer channel descriptor.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]  (same shape)
    """

    def __init__(self, channels, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        # Compress (mean, std) pair from [B, C, 1, 2] → [B, C, 1, 1]
        self.compress = nn.Conv2d(2, 1, kernel_size=(1, 2), bias=False)
        # Channel weight generator
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        # Per-channel statistics  [B, C, 1, 2]
        mu = x.mean(dim=[2, 3], keepdim=True)        # [B, C, 1, 1]
        std = x.std(dim=[2, 3], keepdim=True)         # [B, C, 1, 1]
        stats = torch.cat([mu, std], dim=-1)           # [B, C, 1, 2]

        # Compress statistic dimension
        y = self.compress(stats)                       # [B, C, 1, 1]
        y = self.fc(y)                                 # [B, C, 1, 1]
        weight = hard_sigmoid(y)
        return x * weight


# ============================================================================
# PATSpatialAttention — lightweight 1×1 spatial attention
# ============================================================================

class PATSpatialAttention(nn.Module):
    """Lightweight spatial attention via 1×1 Conv + HardSigmoid.

    Projects channels to a single spatial attention map, then broadcasts back.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]  (same shape)
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        attn = self.conv(x)                   # [B, 1, H, W]
        attn = hard_sigmoid(attn)
        return x * attn


# ============================================================================
# PATConv — Partial Attention Convolution (split-channel conv + attention)
# ============================================================================

class PATConv(nn.Module):
    """Partial Attention Convolution — channel-split conv + attention.

    Splits input channels: conv branch (3×3 depthwise + pointwise) on the first
    portion, attention branch on the remainder.  Branches are concatenated.

    Args:
        c1:        input channels (output channels = input channels)
        n_div:     channel divisor — roughly 1/n_div channels go to conv,
                   the rest to attention
        attn_type: "ch" → PATChannelAttention
                   "sp" → PATSpatialAttention
                   (default "ch")
    """

    def __init__(self, c1, n_div: int = 4, attn_type: str = "ch"):
        super().__init__()
        self.n_div = n_div
        self.attn_type = attn_type

        # Conv branch: applies to dim_conv channels
        self.dim_conv = max(1, c1 // n_div)
        self.conv = nn.Sequential(
            nn.Conv2d(self.dim_conv, self.dim_conv, 3, padding=1,
                       groups=self.dim_conv, bias=False),   # depthwise
            nn.BatchNorm2d(self.dim_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim_conv, self.dim_conv, 1),     # pointwise
            nn.BatchNorm2d(self.dim_conv),
        )

        # Attention branch: applies to dim_attn channels
        self.dim_attn = c1 - self.dim_conv
        if attn_type == "ch":
            self.attn = PATChannelAttention(self.dim_attn)
        elif attn_type == "sp":
            self.attn = PATSpatialAttention(self.dim_attn)
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        # Fuse conv + attn branches back to c1
        self.fuse = nn.Sequential(
            nn.Conv2d(c1, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Split channels
        x_conv = x[:, :self.dim_conv, :, :]
        x_attn = x[:, self.dim_conv:, :, :]

        # Process branches
        conv_out = self.conv(x_conv)
        attn_out = self.attn(x_attn)

        # Concat and fuse
        fused = self.fuse(torch.cat([conv_out, attn_out], dim=1))
        return self.act(fused)


# ============================================================================
# PATBlockLite — lightweight plug-and-play PAT block
# ============================================================================

class PATBlockLite(nn.Module):
    """Lightweight Partial Attention Block — PATConv + MLP + PATsp + residual.

    Architecture:
        shortcut = x
        x = PATConv(x, n_div, attn_type="ch")   → channel attention path
        x = shortcut + x                         → residual 1
        y = Conv1×1(c1 → hidden) + BN + Act
        y = Conv1×1(hidden → c1) + BN           → MLP
        y = PATSpatialAttention(y)               → spatial attention
        out = x + y                              → residual 2

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, c1, n_div: int = 4, mlp_ratio: float = 2.0, drop_path: float = 0.0):
        super().__init__()
        # PATConv with channel attention
        self.pat_conv = PATConv(c1, n_div=n_div, attn_type="ch")

        # MLP
        hidden = max(8, int(c1 * mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=False),
            nn.BatchNorm2d(c1),
        )

        # Spatial attention
        self.pat_sp = PATSpatialAttention(c1)

        # DropPath (identity by default — can be replaced with stochastic depth)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)

    def forward(self, x):
        # PATConv + residual
        identity = x
        x = self.pat_conv(x)
        x = identity + self.drop_path(x)

        # MLP + spatial attention + residual
        y = self.mlp(x)
        y = self.pat_sp(y)
        x = x + self.drop_path(y)

        return x


# ============================================================================
# DropPath (simplified, no timm dependency)
# ============================================================================

class DropPath(nn.Module):
    """Stochastic Depth (DropPath) — simplified, no timm dependency."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ============================================================================
# SELayer — Squeeze-and-Excitation (standard SE, GAP → FC → ReLU → FC → Sigmoid)
# ============================================================================

class SELayer(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    https://arxiv.org/abs/1709.01507

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, channels, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(4, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w
