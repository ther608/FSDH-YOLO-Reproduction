# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""
FG-Net: Frequency-Gated Pyramid Feature Fusion for YOLO.

Based on FG-Net (WACV 2024): "Facial Action Unit Detection with Generalizable Pyramidal Features"
by Yin et al. (https://github.com/ihp-lab/FG-Net).

Core ideas adapted for general object detection:
- Frequency domain decomposition (FFT-based) splits features into low/high frequency bands
- Low-freq captures global semantics, high-freq captures local spatial detail
- Learnable frequency gating adaptively enhances informative frequency components
- Pyramid-style progressive fusion with skip connections (deep low-freq upsampled
  and enriched with shallow high-freq via element-wise addition)

Module inventory:
    FGNetBlock          - Dual-path frequency + spatial feature enhancement
    PyramidFGNetFusion  - Multi-scale pyramid fusion (FG-Net neck for YOLO)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FGNetBlock(nn.Module):
    """Frequency-Gated Feature Block with per-band frequency selectivity.

    Dual-path processing:
      - Frequency path: 2D FFT -> K-band frequency gating -> IFFT
        The frequency spectrum is split into K concentric bands (DC→Nyquist).
        Each channel learns K independent gate weights, one per band.
        This makes the gate genuinely frequency-selective — different frequency
        components within the same channel receive different amplification.
        Unlike a per-channel scalar gate (which is mathematically equivalent to
        spatial SE attention due to FFT linearity), per-band gating cannot be
        reduced to spatial-domain channel scaling.
      - Spatial path: depthwise conv for local detail enhancement.
      - Output = alpha * freq_path + (1-alpha) * spatial_path  (alpha learnable)

    Args:
        channels: Input/output channels.
        reduction: Channel reduction ratio for the frequency gate MLP.
        num_bands: Number of frequency bands (default 4).
    """

    def __init__(self, channels: int, reduction: int = 4, num_bands: int = 4):
        super().__init__()
        self.channels = channels
        self.num_bands = num_bands
        mid = max(channels // reduction, 8)

        # Per-band channel gate: predicts K weights per channel from spatial features
        self.band_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels * num_bands, 1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial detail enhancement (depthwise)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Learnable balance between frequency and spatial paths
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def _build_band_masks(self, h, w, device):
        """Build K soft frequency-band masks [K, Hf, Wf].

        rfft2 output shape is [H, W//2 + 1] (full height, halved width due to
        Hermitian symmetry). Each mask is a Gaussian centered at a different
        radius from DC, creating concentric bands from low to high frequency.
        """
        Hf, Wf = h, w // 2 + 1  # rfft2 output spatial shape
        # Frequency coordinate grid
        fy = torch.fft.fftfreq(h, device=device).abs()       # [Hf] = [h]
        fx = torch.fft.rfftfreq(w, device=device).abs()       # [Wf] = [w//2+1]
        fdist = (fy.view(-1, 1) ** 2 + fx.view(1, -1) ** 2).sqrt()  # [Hf, Wf]

        # K band centers evenly spaced from DC to Nyquist
        max_f = fdist.max()
        band_centers = torch.linspace(0, max_f, self.num_bands, device=device)
        sigma = max_f / (2 * self.num_bands)

        masks = []
        for c in band_centers:
            mask = torch.exp(-((fdist - c) / sigma) ** 2)
            masks.append(mask)
        return torch.stack(masks)  # [K, Hf, Wf]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype

        # Compute band gates in native AMP precision (float16 during training,
        # float32 during validation). Running the convs outside the
        # autocast-disabled block avoids dtype mismatches with stored weights.
        gates = self.band_gate(x)  # [B, C*K, 1, 1]
        gates = gates.view(-1, self.num_bands, self.channels, 1, 1)

        # FFT path: cuFFT requires float32 for non-power-of-2 dims (e.g. 20x20).
        # Run FFT + frequency-domain ops outside AMP autocast in float32.
        with torch.amp.autocast("cuda", enabled=False):
            xf = x.float()

            x_fft = torch.fft.rfft2(xf, norm="ortho")
            mag = torch.abs(x_fft)
            phase = torch.angle(x_fft)

            # Build frequency band masks for current spatial size
            masks = self._build_band_masks(*xf.shape[-2:], xf.device)
            masks = masks.view(1, self.num_bands, 1, *masks.shape[1:])
            # masks: [1, K, 1, Hf, Wf]; gates_f32: [B, K, C, 1, 1]
            gates_f32 = gates.float()

            # Frequency-selective gate: weighted combination of band masks
            gate = (gates_f32 * masks).sum(dim=1)  # [B, C, Hf, Wf]
            mag_gated = mag * gate

            x_freq = torch.fft.irfft2(
                mag_gated * torch.exp(1j * phase), s=xf.shape[-2:], norm="ortho"
            )

        if x_freq.dtype != dtype_in:
            x_freq = x_freq.to(dtype_in)

        # Spatial path (runs in AMP-native precision)
        x_spatial = self.spatial(x)

        # Learnable fusion
        alpha = self.alpha.clamp(0.0, 1.0)
        return alpha * x_freq + (1.0 - alpha) * x_spatial


class UpsampleAdd(nn.Module):
    """Upsample first input to match second, then element-wise add.

    Takes a list of exactly 2 tensors [x_up, x_lateral].
    Bilinear-upsamples x_up to x_lateral's spatial size, then adds them.
    Used as a YAML-compatible building block for FG-Net progressive fusion.

    Args:
        None (parameter-free).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        assert len(x) == 2, f"UpsampleAdd expects 2 inputs, got {len(x)}"
        x_up, x_lat = x
        return F.interpolate(x_up, size=x_lat.shape[-2:], mode="bilinear", align_corners=True) + x_lat


class PyramidFGNetFusion(nn.Module):
    """Multi-scale Pyramid Feature Fusion with Frequency Gating (FG-Net style).

    Takes features at 3 spatial scales (typically from YOLO backbone),
    applies frequency-gated enhancement to each, then progressively fuses
    from deep (small spatial) to shallow (large spatial) with skip connections.
    Outputs enhanced features at all 3 scales for the detection head.

    Architecture (top-down progressive fusion):
        deep  (P5,  1/32)  ──► FGNetBlock ──┐
                                              ├ upsample + skip ──► out5
        mid   (P4,  1/16)  ──► FGNetBlock ──┐
                                              ├ upsample + skip ──► out4
        shallow(P3,  1/8)   ──► FGNetBlock ──► out3

    Args:
        channels (list[int]): Input channels for [deep, mid, shallow] scales.
        reduction (int): Reduction ratio for FGNetBlock internal gate.
    """

    def __init__(self, channels: list[int], reduction: int = 4):
        super().__init__()
        assert len(channels) == 3, f"Expected 3 scale channels, got {len(channels)}"
        c_deep, c_mid, c_shallow = channels

        # Frequency-gated blocks per scale
        self.block_deep = FGNetBlock(c_deep, reduction)
        self.block_mid = FGNetBlock(c_mid, reduction)
        self.block_shallow = FGNetBlock(c_shallow, reduction)

        # Fusion convolutions (deep->mid, mid->shallow lateral connections)
        self.lat_mid = nn.Conv2d(c_mid, c_mid, 1, bias=False)  # lateral for mid
        self.lat_shallow = nn.Conv2d(c_shallow, c_shallow, 1, bias=False)  # lateral for shallow

        self.smooth_deep_mid = nn.Conv2d(c_deep, c_mid, 1, bias=False)  # deep out -> mid channel align
        self.smooth_mid_shallow = nn.Conv2d(c_mid, c_shallow, 1, bias=False)  # mid out -> shallow channel align

        # Output convolutions
        self.out_deep = nn.Conv2d(c_deep, c_deep, 3, padding=1, bias=False)
        self.out_mid = nn.Conv2d(c_mid, c_mid, 3, padding=1, bias=False)
        self.out_shallow = nn.Conv2d(c_shallow, c_shallow, 3, padding=1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _upsample_add(self, x, y):
        """Upsample x to y's spatial size, then add (FG-Net style skip connection)."""
        return F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=True) + y

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            features: [deep_f (P5-size, most semantic), mid_f (P4-size), shallow_f (P3-size, most detailed)]

        Returns:
            list[torch.Tensor]: Enhanced features [P5, P4, P3] (same shapes as input).
        """
        f_deep, f_mid, f_shallow = features  # deep->shallow: spatial size increases

        # Step 1: frequency-gated enhancement per scale
        f_deep = self.block_deep(f_deep)
        f_mid = self.block_mid(f_mid)
        f_shallow = self.block_shallow(f_shallow)

        # Step 2: top-down progressive fusion (deep -> mid -> shallow)
        # deep -> mid fusion
        f_deep_up = self.smooth_deep_mid(f_deep)
        f_mid_fused = self._upsample_add(f_deep_up, self.lat_mid(f_mid))
        f_mid_out = self.out_mid(f_mid_fused)

        # mid -> shallow fusion
        f_mid_up = self.smooth_mid_shallow(f_mid_out)
        f_shallow_fused = self._upsample_add(f_mid_up, self.lat_shallow(f_shallow))
        f_shallow_out = self.out_shallow(f_shallow_fused)

        # deep output (with self-residual)
        f_deep_out = self.out_deep(f_deep)

        return [f_deep_out, f_mid_out, f_shallow_out]
