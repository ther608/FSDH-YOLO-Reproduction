# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
DCNv2 — Deformable Convolution v2 wrapper for YOLO11 Neck.

Uses torchvision.ops.deform_conv2d (no custom CUDA).
Reference: Zhu et al., "Deformable ConvNets v2" (IJCV 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

__all__ = ["DCNConv2", "C3k2_DCN"]


class DCNConv2(nn.Module):
    """Deformable Convolution v2 — learns per-pixel sampling offsets + modulation.

    Wraps torchvision.ops.deform_conv2d. Offsets and modulator mask are predicted
    from the input via separate conv branches, initialised to zero for stable start.

    Args:
        c1: Input channels.
        c2: Output channels.
        k: Kernel size (default 3).
        s: Stride (default 1).
        p: Padding (default k//2).
        g: Groups (default 1).
        bias: Use bias in the weight (default False).
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, bias=False):
        super().__init__()
        self.k = k
        self.s = s
        self.g = g
        p = k // 2 if p is None else p
        self.p = p

        # Offset branch: 2 × k_h × k_w × groups offsets (dx, dy per kernel position)
        offset_c = 2 * k * k * g
        self.offset_conv = nn.Conv2d(c1, offset_c, k, s, p, bias=True)

        # Modulator mask branch: k_h × k_w × groups (one scalar per kernel position)
        mask_c = k * k * g
        self.mask_conv = nn.Conv2d(c1, mask_c, k, s, p, bias=True)

        # Deformable convolution weight
        self.weight = nn.Parameter(torch.randn(c2, c1 // g, k, k))
        self.bias = nn.Parameter(torch.zeros(c2)) if bias else None

        # Zero-init for stable training start
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        nn.init.constant_(self.mask_conv.weight, 0.0)
        nn.init.constant_(self.mask_conv.bias, 0.0)

    def forward(self, x):
        if getattr(self, "_thop_safe_forward", False):
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                stride=(self.s, self.s),
                padding=(self.p, self.p),
                groups=self.g,
            )

        offset = self.offset_conv(x)      # [B, 2*k*k*g, H, W]
        mask = self.mask_conv(x).sigmoid()  # [B, k*k*g, H, W]
        return deform_conv2d(
            x, offset, self.weight, self.bias,
            stride=(self.s, self.s),
            padding=(self.p, self.p),
            mask=mask,
        )


class C3k2_DCN(nn.Module):
    """C3k2 variant with DCNv2 in the second Bottleneck conv.

    Same interface as C3k2 but replaces final processing with a DCN-capable block.
    Designed for neck P3/P4 positions to handle shape deformation.

    Args:
        c1: Input channels.
        c2: Output channels.
        n: Number of sub-blocks (default 1).
        shortcut: Use residual connection (default True).
        e: Expansion ratio for hidden channels (default 0.5).
    """

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList()
        for _ in range(n):
            block = nn.Sequential(
                nn.Conv2d(self.c, self.c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
                DCNConv2(self.c, self.c, 3, 1),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
            ) if shortcut else nn.Sequential(
                DCNConv2(self.c, self.c, 3, 1),
                nn.BatchNorm2d(self.c),
                nn.SiLU(inplace=True),
            )
            self.m.append(block)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
