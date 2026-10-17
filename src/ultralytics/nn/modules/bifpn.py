# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Lite-BiFPN: Lightweight Weighted Bi-directional Feature Pyramid Network.

Reference:
    Tan et al., "EfficientDet: Scalable and Efficient Object Detection" (CVPR 2020)
    https://arxiv.org/abs/1911.09070

Uses fast normalised fusion:  w / (sum(w) + eps)  with ReLU-clipped weights.
"""

import torch
import torch.nn as nn

__all__ = ["BiFPN_Add2", "BiFPN_Add3", "BiFPN_AddConv2", "BiFPN_AddConv3", "BiFPNFusion"]


class BiFPN_Add2(nn.Module):
    """Weighted fusion of two feature maps with the same spatial size and channels.

    Output = (w1*x1 + w2*x2) / (w1 + w2 + eps)

    Args:
        c: Expected channel count of all inputs (for book-keeping, not projection).
        epsilon: Small constant for numerical stability (default 1e-4).
    """

    def __init__(self, c, epsilon=1e-4):
        super().__init__()
        self.c = c
        self.eps = epsilon
        self.w = nn.Parameter(torch.ones(2), requires_grad=True)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != 2:
            raise ValueError(
                f"BiFPN_Add2 expects exactly 2 inputs, "
                f"got {len(xs) if isinstance(xs, (list, tuple)) else type(xs)}"
            )
        x1, x2 = xs

        if x1.shape[-2:] != x2.shape[-2:]:
            raise ValueError(
                f"BiFPN_Add2 spatial size mismatch: "
                f"{tuple(x1.shape[-2:])} vs {tuple(x2.shape[-2:])}"
            )
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(
                f"BiFPN_Add2 channel mismatch: {x1.shape[1]} vs {x2.shape[1]}"
            )

        w = torch.relu(self.w)
        norm = w[0] + w[1] + self.eps
        return (w[0] * x1 + w[1] * x2) / norm


class BiFPN_Add3(nn.Module):
    """Weighted fusion of three feature maps with the same spatial size and channels.

    Output = (w1*x1 + w2*x2 + w3*x3) / (w1 + w2 + w3 + eps)

    Args:
        c: Expected channel count of all inputs (for book-keeping, not projection).
        epsilon: Small constant for numerical stability (default 1e-4).
    """

    def __init__(self, c, epsilon=1e-4):
        super().__init__()
        self.c = c
        self.eps = epsilon
        self.w = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != 3:
            raise ValueError(
                f"BiFPN_Add3 expects exactly 3 inputs, "
                f"got {len(xs) if isinstance(xs, (list, tuple)) else type(xs)}"
            )
        x1, x2, x3 = xs

        if not (x1.shape[-2:] == x2.shape[-2:] == x3.shape[-2:]):
            raise ValueError(
                f"BiFPN_Add3 spatial size mismatch: "
                f"{tuple(x1.shape[-2:])} vs {tuple(x2.shape[-2:])} vs {tuple(x3.shape[-2:])}"
            )
        if not (x1.shape[1] == x2.shape[1] == x3.shape[1]):
            raise ValueError(
                f"BiFPN_Add3 channel mismatch: {x1.shape[1]} vs {x2.shape[1]} vs {x3.shape[1]}"
            )

        w = torch.relu(self.w)
        norm = w[0] + w[1] + w[2] + self.eps
        return (w[0] * x1 + w[1] * x2 + w[2] * x3) / norm


class BiFPN_AddConv2(nn.Module):
    """BiFPN_Add2 + depthwise 3x3 Conv cleanup (Lite-BiFPN-Plus).

    Weighted fusion → DWConv3×3+BN+SiLU for cleaner fused features.
    """

    def __init__(self, c, epsilon=1e-4):
        super().__init__()
        self.add = BiFPN_Add2(c, epsilon)
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    def forward(self, xs):
        return self.conv(self.add(xs))


class BiFPN_AddConv3(nn.Module):
    """BiFPN_Add3 + depthwise 3x3 Conv cleanup (Lite-BiFPN-Plus)."""

    def __init__(self, c, epsilon=1e-4):
        super().__init__()
        self.add = BiFPN_Add3(c, epsilon)
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    def forward(self, xs):
        return self.conv(self.add(xs))


class BiFPNFusion(nn.Module):
    """General BiFPN weighted fusion for N inputs with same spatial size and channels.

    Output = sum(w_i * x_i) / (sum(w_i) + eps)   with ReLU-clipped weights.

    Args:
        n: Number of input feature maps.
        c: Expected channel count of all inputs.
        epsilon: Small constant for numerical stability (default 1e-4).
    """

    def __init__(self, n, c, epsilon=1e-4):
        super().__init__()
        self.n = n
        self.c = c
        self.eps = epsilon
        self.w = nn.Parameter(torch.ones(n), requires_grad=True)

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)) or len(xs) != self.n:
            raise ValueError(
                f"BiFPNFusion expects {self.n} inputs, got "
                f"{len(xs) if isinstance(xs, (list, tuple)) else type(xs)}"
            )
        w = torch.relu(self.w)
        norm = w.sum() + self.eps
        out = w[0] * xs[0]
        for i in range(1, self.n):
            out = out + w[i] * xs[i]
        return out / norm
