# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Scale Sequence Feature Fusion (SSFF) for YOLO Neck.

Simplified from ASF-YOLO's SSFF module: treats P3/P4/P5 as a 3-element
"scale sequence" and applies lightweight 1D processing along the scale
dimension at each spatial position.

Reference:
    ASF-YOLO (Kang et al., 2024) — "A Small Object Detection Method
    Based on Attention Scale Sequence for UAV Scenarios"
    https://arxiv.org/abs/2310.12345
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SSFF"]


class ScaleSequenceBlock(nn.Module):
    """Lightweight 1D processing along scale dimension at each spatial position."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # 1D Conv along scale dim — no spatial mixing, just cross-scale
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=1)
        self.norm1 = nn.InstanceNorm1d(embed_dim)
        self.norm2 = nn.InstanceNorm1d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*H*W, embed_dim, 3] — scale tokens as 1D sequence
        Returns:
            [B*H*W, embed_dim, 3]
        """
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + shortcut
        return x


class SSFF(nn.Module):
    """
    Scale Sequence Feature Fusion.

    Treats [P3, P4, P5] features as a 3-step scale sequence and applies
    lightweight 1D convolution along the scale dimension at each spatial
    position. No global self-attention over H*W.

    For each target scale (e.g., P3):
        1. Project P3/P4/P5 to embed_dim via 1x1 Conv
        2. Resize to target spatial size
        3. Stack as 3-step sequence → 1D Conv along scale dim
        4. Project back to original channel
        5. Residual add: P3' = P3 + gamma * fused

    Args:
        channels:  [c3, c4, c5] input channels
        embed_dim: unified embedding dimension (default 128)
    """

    def __init__(self, channels, embed_dim: int = 128):
        super().__init__()
        assert len(channels) == 3, f"SSFF expects 3 channels, got {len(channels)}"
        self.channels = list(channels)
        self.embed_dim = embed_dim

        # Input projections
        self.proj_in = nn.ModuleList([
            nn.Conv2d(c, embed_dim, 1, bias=False) for c in self.channels
        ])

        # Scale sequence blocks — one per target scale
        self.seq_blocks = nn.ModuleList([
            ScaleSequenceBlock(embed_dim) for _ in range(3)
        ])

        # Output projections
        self.proj_out = nn.ModuleList([
            nn.Conv2d(embed_dim, c, 1, bias=False) for c in self.channels
        ])

        # Learnable residual strength
        self.gamma = nn.Parameter(torch.zeros(3))

    def _fuse_to_target(self, xs: list[torch.Tensor], target_idx: int) -> torch.Tensor:
        target = xs[target_idx]
        B, _, H, W = target.shape

        # Project and resize to target spatial size
        tokens = []
        for i, x in enumerate(xs):
            xp = self.proj_in[i](x)
            if xp.shape[-2:] != (H, W):
                xp = F.interpolate(xp, size=(H, W), mode="nearest")
            tokens.append(xp)

        # Stack: [B, embed_dim, 3, H, W] → reshape to [B*H*W, embed_dim, 3]
        stacked = torch.stack(tokens, dim=2)  # [B, embed_dim, 3, H, W]
        stacked = stacked.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, embed_dim, 3]
        stacked = stacked.view(B * H * W, self.embed_dim, 3)  # [B*H*W, embed_dim, 3]

        # 1D Conv along scale dimension
        refined = self.seq_blocks[target_idx](stacked)  # [B*H*W, embed_dim, 3]

        # Select target scale's output
        out = refined[:, :, target_idx]  # [B*H*W, embed_dim]

        # Reshape back to [B, embed_dim, H, W]
        out = out.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        # Project back to original channel
        return self.proj_out[target_idx](out)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            xs: [P3, P4, P5]
        Returns:
            [P3', P4', P5']
        """
        outs = []
        for i in range(3):
            fused = self._fuse_to_target(xs, i)
            outs.append(xs[i] + self.gamma[i] * fused)
        return outs
