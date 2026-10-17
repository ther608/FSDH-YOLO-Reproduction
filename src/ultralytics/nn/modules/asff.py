# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Adaptive Spatial Feature Fusion (ASFF) for YOLO Neck.

ASFF learns spatially-varying fusion weights for multi-scale features.
For each target scale, all three input scales (P3/P4/P5) are resized to the
target resolution, and a learned spatial attention map determines the per-pixel
fusion ratio via softmax.

Reference:
    Liu et al., "Learning Spatial Fusion for Single-Shot Object Detection"
    https://arxiv.org/abs/1911.09516
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ASFF2", "ASFF3", "FeatureIndex"]


# FeatureIndex is imported from ltcsf.py — re-export for convenience
from .ltcsf import FeatureIndex  # noqa: F811


class _ScaleFusion(nn.Module):
    """Fuse three scale features to one target scale using learned spatial weights."""

    def __init__(self, target_channels: int, other_channels: list[int], compress_c: int = 8):
        super().__init__()
        self.target_c = target_channels

        # 1x1 conv to unify channels from other scales to target_channels
        self.compress_others = nn.ModuleList([
            nn.Conv2d(c, target_channels, 1, bias=False)
            for c in other_channels
        ])

        # Spatial weight layers — one per input scale
        self.weight_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(target_channels, compress_c, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(compress_c, 1, 1),
            )
            for _ in range(3)  # 3 scales: target + 2 others
        ])

    def forward(self, target: torch.Tensor, others: list[torch.Tensor]) -> torch.Tensor:
        """Fuse others into target scale using learned spatial weights."""
        B, C, H, W = target.shape

        # Resize other scales to target spatial size
        resized = [target]  # level 0 = target itself
        for i, other in enumerate(others):
            x = self.compress_others[i](other)
            if x.shape[-2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            resized.append(x)

        # Stack: [B, 3, C, H, W]
        stacked = torch.stack(resized, dim=1)  # [B, 3, C, H, W]

        # Compute spatial weights
        weights = []
        for i in range(3):
            w = self.weight_convs[i](stacked[:, i])  # [B, 1, H, W]
            weights.append(w)
        weights = torch.cat(weights, dim=1)  # [B, 3, H, W]
        weights = F.softmax(weights, dim=1)  # softmax across scales

        # Weighted fusion
        out = torch.zeros_like(target)
        for i in range(3):
            out += weights[:, i:i + 1] * stacked[:, i]
        return out


class ASFF3(nn.Module):
    """
    Adaptive Spatial Feature Fusion for 3-scale features.

    Args:
        channels: [c3, c4, c5] input channels
        compress_c: channel compression ratio for spatial weight convs
    """

    def __init__(self, channels, compress_c: int = 8):
        super().__init__()
        assert len(channels) == 3, f"ASFF3 expects 3 channels, got {len(channels)}"
        c3, c4, c5 = channels

        # Fusion blocks — one per target scale
        self.fuse3 = _ScaleFusion(c3, [c4, c5], compress_c)
        self.fuse4 = _ScaleFusion(c4, [c3, c5], compress_c)
        self.fuse5 = _ScaleFusion(c5, [c3, c4], compress_c)

        # Learnable residual strength
        self.gamma = nn.Parameter(torch.zeros(3))

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            xs: [P3, P4, P5]
        Returns:
            [P3_fused, P4_fused, P5_fused]
        """
        p3, p4, p5 = xs

        p3_out = p3 + self.gamma[0] * self.fuse3(p3, [p4, p5])
        p4_out = p4 + self.gamma[1] * self.fuse4(p4, [p3, p5])
        p5_out = p5 + self.gamma[2] * self.fuse5(p5, [p3, p4])

        return [p3_out, p4_out, p5_out]


class _ScaleFusion2(nn.Module):
    """Fuse two scale features to one target scale using learned spatial weights."""

    def __init__(self, target_channels: int, other_channels: list[int], compress_c: int = 8):
        super().__init__()
        self.compress_other = nn.Conv2d(other_channels[0], target_channels, 1, bias=False)
        self.weight_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(target_channels, compress_c, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(compress_c, 1, 1),
            )
            for _ in range(2)
        ])

    def forward(self, target: torch.Tensor, others: list[torch.Tensor]) -> torch.Tensor:
        B, C, H, W = target.shape
        resized = [target]
        other = self.compress_other(others[0])
        if other.shape[-2:] != (H, W):
            other = F.interpolate(other, size=(H, W), mode="bilinear", align_corners=False)
        resized.append(other)
        stacked = torch.stack(resized, dim=1)  # [B, 2, C, H, W]
        weights = torch.cat([w(stacked[:, i]) for i, w in enumerate(self.weight_convs)], dim=1)
        weights = F.softmax(weights, dim=1)
        out = weights[:, 0:1] * stacked[:, 0] + weights[:, 1:2] * stacked[:, 1]
        return out


class ASFF2(nn.Module):
    """Adaptive Spatial Feature Fusion for 2-scale features.

    Args:
        channels: [c_low, c_high] input channels
        compress_c: channel compression ratio for spatial weight convs (default 8)
    """

    def __init__(self, channels, compress_c: int = 8):
        super().__init__()
        assert len(channels) == 2, f"ASFF2 expects 2 channels, got {len(channels)}"
        c_low, c_high = channels
        self.fuse_low = _ScaleFusion2(c_low, [c_high], compress_c)
        self.fuse_high = _ScaleFusion2(c_high, [c_low], compress_c)
        self.gamma = nn.Parameter(torch.zeros(2))

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        x_low, x_high = xs
        low_out = x_low + self.gamma[0] * self.fuse_low(x_low, [x_high])
        high_out = x_high + self.gamma[1] * self.fuse_high(x_high, [x_low])
        return [low_out, high_out]
