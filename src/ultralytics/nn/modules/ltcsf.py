# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
Lightweight Transformer Cross-Scale Fusion (LTCSF) for YOLO Neck.

LTCSF fuses P3/P4/P5 features using scale-dimension attention:
- Each spatial location sees 3 scale tokens (one per feature level)
- MultiheadAttention operates ONLY on the scale dimension (3 tokens)
- NOT on H*W global self-attention — lightweight and fast

Architecture (for each target scale, e.g. P3):
    1. Resize P3/P4/P5 to target spatial size
    2. Project each to embed_dim via 1x1 Conv
    3. Stack → [B, 3, embed_dim, H, W] → [B*H*W, 3, embed_dim]
    4. ScaleTokenAttention (MHA + FFN) along scale dim
    5. Select target scale token (refined via cross-scale attention)
    6. 1x1 Conv project back to original P3 channel
    7. Residual add: P3' = P3 + gamma * fused

Reference:
    LTCSF — Lightweight Transformer Cross-Scale Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ScaleTokenAttention", "LTCSF", "FeatureIndex"]


class ScaleTokenAttention(nn.Module):
    """
    Lightweight attention over scale tokens.

    For each spatial location, P3/P4/P5 features are treated as three tokens.
    Attention is applied only along the scale dimension (3 tokens), NOT
    across all spatial locations — thus O(3^2) per spatial position, not O((HW)^2).

    Args:
        embed_dim:  embedding dimension for scale tokens
        num_heads:  number of MHA heads
        mlp_ratio:  FFN hidden dim = embed_dim * mlp_ratio
        dropout:    dropout rate for MHA and FFN
    """

    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*H*W, 3, embed_dim] — scale tokens at each spatial position
        Returns:
            [B*H*W, 3, embed_dim] — refined scale tokens
        """
        shortcut = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = shortcut + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class LTCSF(nn.Module):
    """
    Lightweight Transformer Cross-Scale Fusion for YOLO Neck.

    Takes [P3, P4, P5] feature maps and produces enhanced [P3', P4', P5']
    via scale-dimension attention fusion.

    For each target scale, ALL three input scales are projected to embed_dim,
    resized to the target spatial size, and fused via ScaleTokenAttention.

    Args:
        channels:  list[int], input channels of [P3, P4, P5]
        embed_dim: transformer embedding dimension (default 128)
        num_heads: MHA heads (default 4)
        mlp_ratio: FFN expansion ratio (default 2.0)
        dropout:   dropout rate (default 0.0)
    """

    def __init__(self, channels, embed_dim=128, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        assert isinstance(channels, (list, tuple)) and len(channels) == 3, (
            f"LTCSF expects channels=[c3, c4, c5], got {channels}"
        )
        self.channels = list(channels)
        self.embed_dim = embed_dim

        # 1x1 input projections: P3/P4/P5 → embed_dim
        self.proj_in = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1, stride=1, padding=0, bias=False)
            for c in self.channels
        ])

        # Three scale-attention blocks — one per target output scale
        self.scale_attn = nn.ModuleList([
            ScaleTokenAttention(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(3)
        ])

        # 1x1 output projections: embed_dim → original P3/P4/P5 channels
        self.proj_out = nn.ModuleList([
            nn.Conv2d(embed_dim, c, kernel_size=1, stride=1, padding=0, bias=False)
            for c in self.channels
        ])

        # Learnable residual strength, initialized near zero for stable early training
        self.gamma = nn.Parameter(torch.zeros(3))

    def _fuse_to_target(self, xs: list[torch.Tensor], target_idx: int) -> torch.Tensor:
        """
        Fuse P3/P4/P5 features to the spatial resolution of xs[target_idx].

        Args:
            xs:         list of [P3, P4, P5] feature tensors
            target_idx: 0→P3 spatial size, 1→P4, 2→P5

        Returns:
            Fused feature at target spatial size and original channel count.
        """
        target = xs[target_idx]
        B, _, H, W = target.shape

        # Project each scale to embed_dim and resize to target spatial size
        tokens = []
        for i, x in enumerate(xs):
            x_proj = self.proj_in[i](x)                         # [B, embed_dim, Hi, Wi]
            if x_proj.shape[-2:] != (H, W):
                x_proj = F.interpolate(x_proj, size=(H, W), mode="nearest")
            tokens.append(x_proj)                                # [B, embed_dim, H, W]

        # Stack: [B, 3, embed_dim, H, W]
        x = torch.stack(tokens, dim=1)

        # Reshape to [B*H*W, 3, embed_dim] for scale-attention
        x = x.permute(0, 3, 4, 1, 2).contiguous()              # [B, H, W, 3, embed_dim]
        x = x.view(B * H * W, 3, self.embed_dim)                # [B*H*W, 3, embed_dim]

        # Scale-dimension attention (only 3 tokens per spatial position)
        x = self.scale_attn[target_idx](x)                      # [B*H*W, 3, embed_dim]

        # Select the target scale token — it has already attended to the other
        # two scale tokens via MHA, so it carries fused cross-scale information
        # while preserving the target scale's own feature characteristics.
        x = x[:, target_idx, :]                                  # [B*H*W, embed_dim]

        # Reshape back to [B, embed_dim, H, W]
        x = x.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        # Project back to original channel count
        x = self.proj_out[target_idx](x)                        # [B, c_target, H, W]
        return x

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            xs: [P3, P4, P5] input feature list

        Returns:
            [P3', P4', P5'] enhanced feature list
        """
        assert isinstance(xs, (list, tuple)) and len(xs) == 3, (
            f"LTCSF forward expects a list of 3 feature tensors, got {len(xs)}"
        )
        outs = []
        for i in range(3):
            fused = self._fuse_to_target(xs, i)
            outs.append(xs[i] + self.gamma[i] * fused)
        return outs


class FeatureIndex(nn.Module):
    """
    Extracts a single tensor from a list of tensors by index.
    Used to unpack the list output of LTCSF for the YOLO Detect head.

    Args:
        index: index into the input list (0, 1, or 2 for P3/P4/P5)
    """

    def __init__(self, index=0):
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return x[self.index]."""
        return x[self.index]
