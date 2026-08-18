from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.af2_spectral.operator import haar_dwt2, rgb_luminance
from coffee_detector.afab.operator import (
    AFABConfig,
    AFABInputEnhancer,
    af2_entropy_threshold,
    afab_gate,
    minmax_spatial,
)

from .config import AF2RefinementConfig


class RadialAF2Recoverer(AFABInputEnhancer):
    """Legacy AF2 reconstruction with only the density index factorized by radius.

    Everything outside AF2's directional-density lookup is inherited from the
    canonical AFABInputEnhancer: rectangular patches, 50% overlap, 360-degree
    bins, hard entropy threshold, independent RGB transforms, inverse FFT, and
    fold/overlap averaging. The sole mechanism change is 3 radial bands x 360
    directional bins, with entropy normalization performed within each radial
    band over direction.
    """

    def __init__(self, config: AF2RefinementConfig | dict | None = None) -> None:
        frozen = AF2RefinementConfig.from_mapping(config)
        super().__init__(
            AFABConfig(
                mode="af2",
                patch_size=frozen.patch_size,
                overlap=frozen.overlap,
                gamma=frozen.gamma,
                angular_bins=frozen.angular_bins,
                chunk_size=frozen.chunk_size,
                eps=frozen.eps,
            )
        )
        self.radial_bands = frozen.radial_bands
        radius = self.frequency_radius.detach().cpu()
        nonzero = radius[radius > 0].flatten()
        boundaries = torch.quantile(
            nonzero,
            torch.arange(1, self.radial_bands, dtype=torch.float32)
            / self.radial_bands,
        )
        radial = torch.bucketize(radius, boundaries).long()
        radial.clamp_(0, self.radial_bands - 1)
        self.register_buffer("radial_bin", radial, persistent=False)

    def _af2_weight(self, shifted_frequency: torch.Tensor) -> torch.Tensor:
        n, channels, m, _ = shifted_frequency.shape
        magnitude = shifted_frequency.abs().reshape(n, channels, m * m)
        angle = self.angle_bin.to(device=magnitude.device).reshape(1, 1, -1)
        radial = self.radial_bin.to(device=magnitude.device).reshape(1, 1, -1)
        combined = radial * self.config.angular_bins + angle
        combined = combined.expand(n, channels, -1)
        size = self.radial_bands * self.config.angular_bins

        density = magnitude.new_zeros((n, channels, size))
        density.scatter_add_(dim=-1, index=combined, src=magnitude)
        density = density.reshape(
            n, channels, self.radial_bands, self.config.angular_bins
        )
        probability = density / density.sum(dim=-1, keepdim=True).clamp_min(
            self.config.eps
        )
        threshold = af2_entropy_threshold(
            probability, gamma=self.config.gamma, eps=self.config.eps
        )
        normalized = density / density.amax(dim=-1, keepdim=True).clamp_min(
            self.config.eps
        )
        direction = torch.where(
            normalized <= threshold.unsqueeze(-1),
            torch.zeros_like(normalized),
            normalized,
        ).reshape(n, channels, size)
        pixel_weight = torch.gather(direction, dim=-1, index=combined)
        return pixel_weight.reshape(n, channels, m, m)


class AF2RefinementInputEnhancer(nn.Module):
    """Parameter-free AF2 follow-up with isolated radial/wavelet mechanisms."""

    def __init__(self, config: AF2RefinementConfig | dict | None = None) -> None:
        super().__init__()
        self.config = AF2RefinementConfig.from_mapping(config)
        legacy_config = AFABConfig(
            mode="af2",
            patch_size=self.config.patch_size,
            overlap=self.config.overlap,
            gamma=self.config.gamma,
            angular_bins=self.config.angular_bins,
            chunk_size=self.config.chunk_size,
            eps=self.config.eps,
        )
        self.legacy = AFABInputEnhancer(legacy_config)
        self.radial = RadialAF2Recoverer(self.config)

    def _validate(self, value: torch.Tensor) -> None:
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("AF2 refinement membutuhkan BCHW RGB")
        if not torch.is_floating_point(value):
            raise TypeError("AF2 refinement membutuhkan tensor floating point")

    def spectral_recovered(self, value: torch.Tensor) -> torch.Tensor:
        self._validate(value)
        if self.config.arm in {"AF2RAD", "AF2RADWAV"}:
            return self.radial.recover(value)
        return self.legacy.recover(value)

    def base_cue(self, value: torch.Tensor) -> torch.Tensor:
        return minmax_spatial(self.spectral_recovered(value), eps=self.config.eps)

    def wavelet_cue(self, value: torch.Tensor) -> torch.Tensor:
        """Exact WAV1 luminance two-level Haar detail-energy cue."""

        self._validate(value)
        luminance = rgb_luminance(value)
        target = luminance.shape[-2:]
        current = luminance
        energies = []
        for _ in range(self.config.wavelet_levels):
            bands, _ = haar_dwt2(current)
            current = bands[:, :, 0]
            detail = bands[:, :, 1:].square().sum(2).add(self.config.eps).sqrt()
            detail = F.interpolate(
                detail, size=target, mode="bilinear", align_corners=False
            )
            energies.append(minmax_spatial(detail, self.config.eps))
        cue = torch.stack(energies).mean(0)
        return cue.expand_as(value)

    def fused_cue(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base_cue(value)
        if self.config.arm not in {"AF2WAV", "AF2RADWAV"}:
            return base
        wavelet = self.wavelet_cue(value)
        # Frozen coefficient-free union: a wavelet cue can add support but cannot
        # numerically attenuate the AF2 cue at any pixel/channel.
        return torch.maximum(base, wavelet)

    def recover(self, value: torch.Tensor) -> torch.Tensor:
        """Return the cue used by the residual gate, not a second image stream."""

        return self.fused_cue(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self._validate(value)
        if self.config.arm == "AF2C":
            # Preserve the canonical control byte-for-byte rather than routing it
            # through an algebraically equivalent rewritten gate.
            return self.legacy(value)
        if self.config.arm == "AF2RAD":
            return afab_gate(
                value, self.radial.recover(value), eps=self.config.eps
            )
        cue = self.fused_cue(value)
        return value + value * cue
