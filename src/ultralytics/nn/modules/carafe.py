# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
CARAFE — Content-Aware ReAssembly of FEatures (Lightweight Upsampler).

Reference:
    Wang et al., "CARAFE: Content-Aware ReAssembly of FEatures" (ICCV 2019)
    https://arxiv.org/abs/1905.02188

PyTorch-only implementation. No mmcv / custom CUDA dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CARAFE"]


class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures — learned lightweight upsampler.

    For each output position, a per-pixel reassembly kernel (predicted from the
    content encoder) is applied to the corresponding input neighbourhood.

    Input:  [B, C, H, W]
    Output: [B, C, H*scale, W*scale]

    Args:
        c: Input channels (output channels unchanged).
        scale: Upsampling factor (default 2).
        up_kernel: Reassembly kernel size (odd, default 5).
        encoder_kernel: Kernel size for content encoder conv (default 3).
        compressed_channels: Bottleneck channels (default 64).
    """

    def __init__(self, c, scale=2, up_kernel=5, encoder_kernel=3, compressed_channels=64):
        super().__init__()
        self.scale = scale
        self.up_kernel = up_kernel
        self.compress_c = compressed_channels

        # Channel compressor
        self.compress = nn.Conv2d(c, compressed_channels, 1)

        # Content encoder → up_kernel² * scale² kernel weights
        assert encoder_kernel % 2 == 1, "encoder_kernel must be odd"
        pad = encoder_kernel // 2
        self.encoder = nn.Conv2d(
            compressed_channels,
            up_kernel * up_kernel * scale * scale,
            encoder_kernel,
            padding=pad,
        )

        # Pixel shuffle: [B, K²*S², H, W] → [B, K², H*S, W*S]
        self.kernel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.scale
        kup = self.up_kernel
        k2 = kup * kup
        pad_k = kup // 2

        # ── 1. Channel compress + content encode → raw kernels ──
        compressed = self.compress(x)                    # [B, Cc, H, W]
        kernel_raw = self.encoder(compressed)            # [B, K²*S², H, W]

        # ── 2. Pixel-shuffle to output resolution ──
        kernels = self.kernel_shuffle(kernel_raw)        # [B, K², H*S, W*S]

        # ── 3. Softmax normalise each reassembly kernel ──
        kernels = F.softmax(
            kernels.reshape(b, k2, -1), dim=1
        )  # [B, K², Ho*Wo]
        kernels = kernels.reshape(b, k2, h * s, w * s)   # [B, K², Ho, Wo]

        # ── 4. Unfold input to extract K×K neighbourhoods per input pixel ──
        x_pad = F.pad(x, (pad_k, pad_k, pad_k, pad_k), mode="replicate")
        patches = F.unfold(x_pad, kernel_size=(kup, kup))  # [B, C*K², H*W]
        patches = patches.view(b, c, k2, h, w)             # [B, C, K², H, W]

        # ── 5. Group S² sub-pixel kernels per input position ──
        # kernels:  [B, K², H*S, W*S]
        # → reshape to group S² consecutive output pixels
        kernels_b = kernels.reshape(b, k2, s, h, s, w)    # [B, K², S, H, S, W]
        kernels_b = (
            kernels_b.permute(0, 2, 4, 1, 3, 5)           # [B, S, S, K², H, W]
            .reshape(b, s * s, k2, h, w)                  # [B, S², K², H, W]
        )

        # ── 6. Weighted sum: one output block per sub-pixel ──
        out_blocks = torch.einsum("bckhw,bnkhw->bnchw", patches, kernels_b)
        # out_blocks: [B, S², C, H, W]

        # ── 7. Interleave sub-pixel blocks to full output ──
        out_blocks = out_blocks.reshape(b, s, s, c, h, w)    # [B, S, S, C, H, W]
        out = out_blocks.permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.reshape(b, c, h * s, w * s)                # [B, C, Ho, Wo]
        return out
