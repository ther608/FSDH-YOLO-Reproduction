# Ultralytics AGPL-3.0 License
"""
Channel-preserving feature-enhancement modules for small-object detection.

Modules:
    SHAB                 — local/mid/global multi-branch enhancement
    ASPP                 — atrous spatial pyramid pooling
    DilatedConv          — single dilated-convolution control
    RepConvBlock         — re-parameterizable convolution control
    MultiScaleDWConv     — parallel 3/5/7 depthwise convolutions
    ReceptiveFieldBlock  — parallel receptive-field branches
    CFSB                 — low/high-frequency gate fusion

Note: CARAFE already exists in ultralytics/nn/modules/carafe.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, RepConv

__all__ = [
    "ASPP",
    "CFSB",
    "DilatedConv",
    "MultiScaleDWConv",
    "ReceptiveFieldBlock",
    "RepConvBlock",
    "SHAB",
]


def _reduced_channels(channels, reduction=4):
    """Return a practical bottleneck width for the YOLO11n feature levels."""
    return max(channels // reduction, 16)


# ============================================================================
# SHAB — Spatial Hierarchy Attention Block
# ============================================================================
# Reference: inspired by FSDETR spatial hierarchy attention.
# Three-branch lightweight attention: local detail, mid-range context, global prior.
# Input/output shape identical — safe to insert anywhere.

class SHAB(nn.Module):
    """Spatial Hierarchy Attention Block with selectable local/mid/global branches.

    Paper name: SHEB (Spatial Hierarchy Enhancement Block).
    Code class name: SHAB (kept for compatibility with released checkpoints).

    Maintains channel and spatial invariance (C→C, H×W→H×W).

    Architecture:
        local:   DWConv 3×3                         → fine details
        middle:  Dilated DWConv 5×5 (dilation=2)    → larger receptive field
        global:  GAP → 1×1 → Sigmoid               → channel attention prior
        fusion:  1×1 Conv over concatenated branches + skip connection
    """

    def __init__(self, c1, branches=3, *_):
        super().__init__()
        legacy_selectors = {1: "L", 2: "LM", 3: "LMG"}
        if isinstance(branches, int):
            if branches not in legacy_selectors:
                raise ValueError(f"Legacy SHAB branches must be 1, 2, or 3, got {branches}")
            selector = legacy_selectors[branches]
        elif isinstance(branches, str):
            selector = branches.upper().replace("+", "").replace("_", "").replace("-", "").replace(" ", "")
        else:
            raise TypeError(f"SHAB branches must be an int or an L/M/G selector string, got {type(branches).__name__}")
        if not selector or any(branch not in "LMG" for branch in selector) or len(set(selector)) != len(selector):
            raise ValueError(f"SHAB selector must be a non-empty unique combination of L, M, and G, got {branches}")

        # Canonical order makes "ML" and "LM" architecturally identical.
        self.branch_names = tuple(branch for branch in "LMG" if branch in selector)
        self.branches = len(self.branch_names)
        mid_c = max(c1 // 4, 16)

        # Local detail branch
        self.local = (
            nn.Sequential(
                nn.Conv2d(c1, mid_c, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, c1, 3, padding=1, groups=mid_c),
            )
            if "L" in self.branch_names
            else None
        )

        # Mid-range context branch (dilated for larger RF)
        self.mid = (
            nn.Sequential(
                nn.Conv2d(c1, mid_c, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, c1, 5, padding=4, groups=mid_c, dilation=2),
            )
            if "M" in self.branch_names
            else None
        )

        # Global context branch
        self.global_pool = nn.AdaptiveAvgPool2d(1) if "G" in self.branch_names else None
        self.global_fc = (
            nn.Sequential(
                nn.Conv2d(c1, mid_c, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, c1, 1),
                nn.Sigmoid(),
            )
            if "G" in self.branch_names
            else None
        )

        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(c1 * self.branches, c1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Older checkpoints have only the integer `branches` attribute. Preserve
        # their 1=L, 2=LM, 3=LMG behavior when loaded with this implementation.
        branch_names = getattr(self, "branch_names", tuple("LMG"[: getattr(self, "branches", 3)]))
        outputs = []
        if "L" in branch_names and self.local is not None:
            outputs.append(self.local(x))
        if "M" in branch_names and self.mid is not None:
            outputs.append(self.mid(x))
        if "G" in branch_names and self.global_pool is not None and self.global_fc is not None:
            global_out = self.global_fc(self.global_pool(x))
            outputs.append(global_out * x)

        # Fuse only the explicitly enabled branches.
        cat = torch.cat(outputs, dim=1)
        enhanced = self.fuse(cat)                                  # [B, C, H, W]
        return x + enhanced


# ============================================================================
# Controlled alternatives for the SHAB feature-enhancement comparison
# ============================================================================


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling with a residual, channel-preserving output.

    The block follows the standard ASPP ingredients: a 1x1 branch, parallel
    3x3 atrous branches, and an image-level pooling branch. The final 1x1
    projection restores the input channel count so it can replace SHAB at the
    exact same P3/P4/P5 positions.
    """

    def __init__(self, c1, rates=(1, 3, 5), *_):
        super().__init__()
        if not rates or any(int(rate) < 1 for rate in rates):
            raise ValueError(f"ASPP dilation rates must be positive, got {rates}")
        mid_c = _reduced_channels(c1)
        self.pointwise = Conv(c1, mid_c, 1, 1)
        self.atrous = nn.ModuleList(Conv(c1, mid_c, 3, 1, d=int(rate)) for rate in rates)
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_proj = Conv(c1, mid_c, 1, 1)
        self.project = Conv(mid_c * (len(rates) + 2), c1, 1, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        pooled = F.interpolate(self.image_proj(self.image_pool(x)), size=size, mode="bilinear", align_corners=False)
        features = [self.pointwise(x), *(branch(x) for branch in self.atrous), pooled]
        return self.act(x + self.project(torch.cat(features, dim=1)))


class DilatedConv(nn.Module):
    """Single dilated 3x3 bottleneck used to isolate atrous context from branching."""

    def __init__(self, c1, dilation=2, *_):
        super().__init__()
        if int(dilation) < 1:
            raise ValueError(f"DilatedConv dilation must be positive, got {dilation}")
        mid_c = _reduced_channels(c1)
        self.reduce = Conv(c1, mid_c, 1, 1)
        self.context = Conv(mid_c, mid_c, 3, 1, d=int(dilation))
        self.project = Conv(mid_c, c1, 1, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.project(self.context(self.reduce(x))))


class RepConvBlock(nn.Module):
    """Same-channel RepConv control with its identity BN branch enabled."""

    def __init__(self, c1, *_):
        super().__init__()
        self.rep = RepConv(c1, c1, k=3, s=1, p=1, bn=True)

    def forward(self, x):
        return self.rep(x)


class MultiScaleDWConv(nn.Module):
    """Parallel 3x3, 5x5 and 7x7 depthwise-convolution feature enhancement."""

    def __init__(self, c1, kernels=(3, 5, 7), *_):
        super().__init__()
        if not kernels or any(int(kernel) < 1 or int(kernel) % 2 == 0 for kernel in kernels):
            raise ValueError(f"MultiScaleDWConv kernels must be positive odd integers, got {kernels}")
        mid_c = _reduced_channels(c1)
        self.branches = nn.ModuleList(
            nn.Sequential(
                Conv(c1, mid_c, 1, 1),
                Conv(mid_c, mid_c, int(kernel), 1, g=mid_c),
            )
            for kernel in kernels
        )
        self.project = Conv(mid_c * len(kernels), c1, 1, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.project(torch.cat([branch(x) for branch in self.branches], dim=1)))


class ReceptiveFieldBlock(nn.Module):
    """Compact RFB-style block with parallel local and dilated receptive fields."""

    def __init__(self, c1, dilations=(1, 3, 5), scale=0.1, *_):
        super().__init__()
        if not dilations or any(int(dilation) < 1 for dilation in dilations):
            raise ValueError(f"RFB dilation rates must be positive, got {dilations}")
        mid_c = _reduced_channels(c1)
        self.local = Conv(c1, mid_c, 1, 1)
        self.context = nn.ModuleList(
            nn.Sequential(
                Conv(c1, mid_c, 1, 1),
                Conv(mid_c, mid_c, 3, 1, d=int(dilation)),
            )
            for dilation in dilations
        )
        self.project = Conv(mid_c * (len(dilations) + 1), c1, 1, 1, act=False)
        self.scale = float(scale)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        features = [self.local(x), *(branch(x) for branch in self.context)]
        return self.act(x + self.scale * self.project(torch.cat(features, dim=1)))


# ============================================================================
# CFSB — Combined Frequency-Spatial Block
# ============================================================================
# Reference: inspired by FSDETR frequency-spatial enhancement.
# Decomposes into low-freq (context / colour consistency) and high-freq
# (edges / texture / boundary) via AvgPool subtraction, then fuses with
# a learned gate.  No explicit FFT — lightweight and stable.

class CFSB(nn.Module):
    """Combined Frequency-Spatial Block — low/high-frequency gate fusion.

    Maintains channel and spatial invariance (C→C, H×W→H×W).

    Architecture:
        low  = AvgPool(x)                               → smoothed / colour-consistent
        high = x - low                                  → edges / fine details
        low_feat  = large-kernel DWConv(low)            → enhanced low-freq context
        high_feat = DWConv 3×3(high)                    → enhanced high-freq texture
        gate = Conv(cat(low_feat, high_feat)) → Sigmoid → per-channel spatial gate
        enhanced = gate * high_feat + (1 - gate) * low_feat
        out = x + Conv(enhanced)
    """

    def __init__(self, c1, *_):
        super().__init__()
        mid_c = max(c1 // 4, 16)

        # Frequency decomposition
        self.pool = nn.AvgPool2d(5, 1, 2)  # large-kernel smooth for low-freq

        # Low-frequency enhancement: large-kernel DWConv
        self.low_conv = nn.Sequential(
            nn.Conv2d(c1, mid_c, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, c1, 7, padding=3, groups=mid_c),
        )

        # High-frequency enhancement: 3×3 DWConv
        self.high_conv = nn.Sequential(
            nn.Conv2d(c1, mid_c, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, c1, 3, padding=1, groups=mid_c),
        )

        # Gate from concatenated frequency features
        self.gate = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.Sigmoid(),
        )

        # Final projection
        self.proj = nn.Sequential(
            nn.Conv2d(c1, c1, 1),
        )

    def forward(self, x):
        low = self.pool(x)                           # smoothed
        high = x - low                                # detail

        low_feat = self.low_conv(low)                 # [B, C, H, W]
        high_feat = self.high_conv(high)              # [B, C, H, W]

        gate = self.gate(torch.cat([low_feat, high_feat], dim=1))  # [B, C, H, W]
        enhanced = gate * high_feat + (1 - gate) * low_feat         # [B, C, H, W]

        return x + self.proj(enhanced)
