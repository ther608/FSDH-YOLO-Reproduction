# Ultralytics AGPL-3.0 License
"""
Advanced attention & feature modules for YOLO11 Lychee detection.

Modules:
    IEMA   — Improved EMA with spatial enhancement
    ELA    — Efficient Local Attention (lightweight 1D strip attention)
    PConv  — Partial Convolution (PConv block for C3k2-Faster)
    SPDConv— Original project Space-to-Depth + bare convolution
    SPOConv— PixelUnshuffle + YOLO Conv-BN-SiLU downsampling
    Wavelet— Wavelet-based downsampling + feature enhancement
    DASI   — Dual Attention & Spatial Interaction for Neck fusion
    CTE    — Color-Texture Enhancement for ripeness detection
    RAAM   — Ripeness-Aware Attention Module
    FreqFusionLite — Lightweight frequency/spatial fusion
    ECA    — Efficient Channel Attention (1D conv local cross-channel)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ["IEMA", "ELA", "PConv", "C3k2_Faster", "SPDConv", "SPOConv", "WaveletDown", "WaveletEnhance", "DASI",
           "CTE", "RAAM", "FreqFusionLite", "ECA"]


# ============================================================================
# IEMA — Improved Efficient Multi-Scale Attention
# ============================================================================
# Reference: "IEMA: Improving Efficient Multi-Scale Attention" (2024)
# Enhances EMA by adding a simple convolution path for richer spatial context.

class IEMA(nn.Module):
    """Improved EMA with extra 3x3 spatial enhancement branch."""

    def __init__(self, channels, factor=32):
        super().__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, 1)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, 3, padding=1)
        # IEMA addition: extra 3x3 for spatial enhancement
        self.conv3x3_enhance = nn.Conv2d(channels // self.groups, channels // self.groups, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)

        # EMA pathway
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # Cross-spatial weight via matrix multiplication
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # IEMA: spatial enhancement branch
        enhance = self.conv3x3_enhance(group_x)
        out = (group_x * weights.sigmoid() + enhance).reshape(b, c, h, w)
        return out


# ============================================================================
# ELA — Efficient Local Attention (1D strip attention, ultra-lightweight)
# ============================================================================
# Reference: "ELA: Efficient Local Attention for Image Restoration" (2024)
# Uses 1D convolutions along H and W strips for O(H+W) complexity attention.

class ELA(nn.Module):
    """Efficient Local Attention using 1D strip convolutions."""

    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv_w = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        # H-strip attention
        x_h = self.pool_h(x).reshape(b * c, 1, h)
        attn_h = self.sigmoid(self.conv_h(x_h)).reshape(b, c, h, 1)
        # W-strip attention
        x_w = self.pool_w(x).reshape(b * c, 1, w)
        attn_w = self.sigmoid(self.conv_w(x_w)).reshape(b, c, 1, w)
        return x * attn_h * attn_w


# ============================================================================
# PConv + C3k2_Faster — Partial Convolution for faster inference
# ============================================================================
# Reference: Chen et al., "Run, Don't Walk" (CVPR 2023)
# PConv applies standard Conv on only a subset of input channels for efficiency.

class PConv(nn.Module):
    """Partial Convolution: conv on cp channels, identity on remaining (c-cp)."""

    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)


class C3k2_Faster(nn.Module):
    """C3k2 variant with PConv replacing BottleNeck Conv in the C2f block.
    Follows YOLOv8-C2f-Faster-EMA (2024) pattern for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            PConv(self.c) if shortcut else nn.Sequential(
                nn.Conv2d(self.c, self.c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ============================================================================
# SPDConv — Space-to-Depth Convolution (replaces stride-2 Conv)
# ============================================================================
# Reference: "SPD-Conv: A Novel CNN Architecture for Small Object Detection"
# Replaces stride-2 downsampling with space-to-depth + conv to preserve
# fine-grained spatial info for small objects.

class SPDConv(nn.Module):
    """Space-to-Depth Convolution: preserves spatial detail during downsampling.
    Instead of stride-2 Conv (loses 75% spatial info), uses space-to-depth
    reshape followed by a non-strided Conv."""

    def __init__(self, c1, c2, kernel_size=3):
        super().__init__()
        self.cv = nn.Conv2d(c1 * 4, c2, kernel_size, 1, kernel_size // 2)

    def forward(self, x):
        # space-to-depth: (b,c,h,w) → (b,c*4,h/2,w/2)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(b, c * 4, h // 2, w // 2)
        return self.cv(x)


class SPOConv(nn.Module):
    """Standardized space-to-depth downsampling for the controlled study.

    PixelUnshuffle performs the lossless 2x spatial-to-channel rearrangement;
    the project-standard Conv then applies Conv2d, BatchNorm2d and SiLU with
    stride 1. SPDConv remains unchanged so completed checkpoints and the
    original implementation stay available as a separate control.
    """

    def __init__(self, c1, c2, kernel_size=3):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.cv = Conv(c1 * 4, c2, kernel_size, 1)

    def forward(self, x):
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(f"SPOConv requires even spatial dimensions, got {tuple(x.shape[-2:])}")
        return self.cv(self.unshuffle(x))


# ============================================================================
# Wavelet — Wavelet-based Downsampling & Feature Enhancement
# ============================================================================
# Uses Haar wavelet decomposition to extract low-frequency (LL) component
# as a detail-preserving downsampling alternative.

class WaveletDown(nn.Module):
    """Haar wavelet downsampling: replaces stride-2 Conv with LL subband extraction.
    Preserves more structural information than strided convolution."""

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x):
        # Haar wavelet decomposition: avg(2x2 blocks) → LL subband
        x = F.avg_pool2d(x, 2, 2)  # simplest wavelet LL equivalent
        return self.proj(x)


class WaveletEnhance(nn.Module):
    """Lightweight multi-frequency feature enhancement before SPPF.
    Decomposes features into low/high frequency bands via simple avg pooling
    differences, then fuses back for richer multi-scale representation."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * 3, 1),
        )

    def forward(self, x):
        # Multi-frequency decomposition
        low = F.avg_pool2d(x, 3, 1, 1)  # low-freq (smoothed)
        high = x - low                     # high-freq (detail)
        mid = F.avg_pool2d(x, 2, 2)
        mid = F.interpolate(mid, size=x.shape[-2:], mode="nearest")

        # Learned frequency mixing
        w = self.fc(x).sigmoid()
        w_low, w_mid, w_high = w.chunk(3, 1)
        return x + w_low * low + w_mid * mid + w_high * high


# ============================================================================
# DASI — Dual Attention & Spatial Interaction for Neck multi-scale fusion
# ============================================================================
# Reference: "DASI: Dual Attention and Spatial Interaction for Feature Fusion"
# Lightweight neck fusion that applies channel + spatial dual attention to
# the concatenated multi-scale features.

class DASI(nn.Module):
    """Dual Attention & Spatial Interaction for 2-feature neck fusion.
    Replaces standard Concat with attention-guided weighted fusion.
    Expects two input tensors of same spatial size (after upsampling)."""

    def __init__(self, c1, c2):
        super().__init__()
        total_c = c1 + c2
        # Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_c, total_c // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_c // 8, total_c, 1),
            nn.Sigmoid(),
        )
        # Spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(total_c, 1, 7, padding=3),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(total_c, c1, 1)

    def forward(self, xs):
        # xs: list of 2 tensors (after upsampling to same spatial size)
        assert len(xs) == 2, f"DASI expects 2 inputs, got {len(xs)}"
        x = torch.cat(xs, dim=1)
        x = x * self.ca(x) * self.sa(x)
        return self.proj(x)


# ============================================================================
# CTE — Color-Texture Enhancement for ripeness detection
# ============================================================================
# Enhances color trends and local texture critical for distinguishing
# unripe → semi-ripe → ripe.  Lightweight, channel-invariant refinement block.

class CTE(nn.Module):
    """Color-Texture Enhancement: enhances low-freq color + high-freq texture.

    Architecture:
        low = AvgPool(x) for smoothed color trends
        high = x - low for edge/texture details
        texture = DWConv3x3 + DWConv5x5 for multi-scale local texture
        gate = Conv1x1(fuse) → Sigmoid
        out = x + learnable_scale * x * gate
    """

    def __init__(self, c1, reduction=16):
        super().__init__()
        hidden = max(8, c1 // reduction)
        # Low-frequency color branch
        self.low_pool = nn.AvgPool2d(3, 1, 1)
        self.low_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )
        # Multi-scale texture branch
        self.dw3x3 = nn.Conv2d(c1, c1, 3, padding=1, groups=c1)
        self.dw5x5 = nn.Conv2d(c1, c1, 5, padding=2, groups=c1)
        self.texture_fuse = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.ReLU(inplace=True),
        )
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Conv2d(c1 * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Frequency decomposition
        low = self.low_pool(x)                          # smoothed color regions
        high = x - low                                   # edges / fine texture
        # Texture extraction
        t3 = self.dw3x3(x)
        t5 = self.dw5x5(x)
        texture = self.texture_fuse(torch.cat([t3, t5], dim=1))
        # Global color context (expand to spatial size)
        color_ctx = self.low_gate(low)                   # (B, C, 1, 1)
        color_ctx = color_ctx.expand(-1, -1, x.size(2), x.size(3))
        # Combined gate: texture ⊗ color
        gate = self.gate(torch.cat([texture, color_ctx], dim=1))
        return x + self.scale * x * gate


# ============================================================================
# RAAM — Ripeness-Aware Attention Module
# ============================================================================
# Channel attention + spatial attention with a learnable ripeness bias.
# Designed to emphasize maturity-relevant color/texture regions.

class RAAM(nn.Module):
    """Ripeness-Aware Attention Module: channel + spatial attention with ripeness bias.

    Architecture:
        Channel attn: GAP(x) ⊕ GMP(x) → shared MLP (1×1 Conv) → sigmoid
        Spatial attn: mean(x) ⊕ max(x) over channels → Conv7×7 → sigmoid
        Ripeness bias: learnable per-channel offset (init zero)
        out = x + scale * x * ch_weight * sp_weight
    """

    def __init__(self, c1, reduction=16, spatial_kernel=7):
        super().__init__()
        hidden = max(8, c1 // reduction)
        # Channel attention: GAP + GMP → shared MLP
        self.channel_att = nn.Sequential(
            nn.Conv2d(c1 * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2),
            nn.Sigmoid(),
        )
        # Learnable ripeness bias: allows per-channel sensitivity to ripeness cues
        self.ripeness_bias = nn.Parameter(torch.zeros(1, c1, 1, 1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Channel attention
        avg_out = torch.mean(x, dim=[2, 3], keepdim=True)   # GAP
        max_out, _ = torch.max(x.reshape(x.size(0), x.size(1), -1), dim=2, keepdim=True)
        max_out = max_out.unsqueeze(-1)                       # GMP
        ch_w = self.channel_att(torch.cat([avg_out, max_out], dim=1))
        ch_w = ch_w + self.ripeness_bias
        # Spatial attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        sp_w = self.spatial_att(torch.cat([avg_s, max_s], dim=1))
        # Combined modulation
        return x + self.scale * x * ch_w * sp_w


# ============================================================================
# FreqFusionLite — Lightweight frequency/spatial fusion
# ============================================================================
# Decomposes features into low-freq (color/smooth regions) and high-freq
# (edges/texture), then fuses with learned gates.  No explicit FFT —
# uses AvgPool subtraction as a stable frequency proxy.

class FreqFusionLite(nn.Module):
    """Lightweight frequency-aware fusion via AvgPool low/high decomposition.

    Architecture:
        low = AvgPool(x)                          # smoothed / color-consistent
        high = x - low                            # edges / fine details
        low_gate = Conv1×1(low) → Sigmoid         # per-channel low-freq gate
        high_gate = DWConv3×3(high) → Sigmoid     # spatial-aware high-freq gate
        out = x + γ_low * low * low_gate + γ_high * high * high_gate
    """

    def __init__(self, c1, reduction=16):
        super().__init__()
        hidden = max(8, c1 // reduction)
        self.pool = nn.AvgPool2d(3, 1, 1)
        # Low-frequency gate (channel-wise)
        self.low_gate = nn.Sequential(
            nn.Conv2d(c1, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )
        # High-frequency gate (spatial-aware via grouped conv)
        self.high_conv = nn.Conv2d(c1, c1, 3, padding=1, groups=c1)
        self.high_gate = nn.Sequential(
            nn.Conv2d(c1, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1),
            nn.Sigmoid(),
        )
        self.gamma_low = nn.Parameter(torch.ones(1))
        self.gamma_high = nn.Parameter(torch.ones(1))

    def forward(self, x):
        low = self.pool(x)
        high = x - low
        low_g = self.low_gate(low)
        high_feat = self.high_conv(high)
        high_g = self.high_gate(high_feat)
        return x + self.gamma_low * low * low_g + self.gamma_high * high_feat * high_g


# ============================================================================
# ECA — Efficient Channel Attention
# ============================================================================
# Reference: Wang et al., "ECA-Net: Efficient Channel Attention" (CVPR 2020)
# https://arxiv.org/abs/1910.03151
# Uses 1D convolution for local cross-channel interaction without dimension
# reduction.  Extremely lightweight — suitable as a control baseline.

class ECA(nn.Module):
    """Efficient Channel Attention: GAP → Conv1d(k) → Sigmoid → channel reweight.

    Args:
        c1: input channels
        k_size: 1D convolution kernel size (controls local cross-channel span)
    """

    def __init__(self, c1, k_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 1D convolution along the channel dimension
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.gap(x)                     # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2) # (B, 1, C)
        y = self.conv1d(y)                   # (B, 1, C)  1D cross-channel interaction
        y = y.transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y
