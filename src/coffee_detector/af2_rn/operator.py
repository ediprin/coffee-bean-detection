from __future__ import annotations

import torch

from coffee_detector.afab.operator import (
    AFABConfig,
    AFABInputEnhancer,
    af2_entropy_threshold,
)

from .config import AF2RNConfig


class AF2RNInputEnhancer(AFABInputEnhancer):
    """Legacy AF2 with one change: radial-baseline-normalized angle density.

    Radius never masks or separately thresholds a coefficient. It indexes the
    within-annulus median used to remove the natural radial magnitude baseline
    before the original 360-bin AF2 angular statistic is accumulated.
    """

    def __init__(self, config: AF2RNConfig | dict | None = None) -> None:
        frozen = AF2RNConfig.from_mapping(config)
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
        self.af2rn_config = frozen
        annulus = torch.floor(
            self.frequency_radius / float(frozen.annulus_width)
        ).long()
        # An even 32x32 fftshift grid contains only one represented point at
        # the outermost corner radius (-16, -16). A singleton median would
        # force that coefficient to zero for geometric rather than spectral
        # reasons, so merge only such non-DC singleton annuli inward.
        counts = torch.bincount(annulus.flatten())
        for ring_id in torch.nonzero(counts == 1).flatten().tolist():
            if ring_id > 0:
                annulus[annulus == ring_id] = ring_id - 1
        self.register_buffer("annulus_bin", annulus, persistent=False)

    @property
    def annulus_count(self) -> int:
        return int(self.annulus_bin.max().item()) + 1

    def radial_normalize_magnitude(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Return positive magnitude excess over the within-annulus median.

        ``magnitude`` may be ``[N,C,H,W]`` or ``[N,C,H*W]``. The result has the
        same shape. The DC annulus has one member and therefore contributes
        exactly zero without a special learned or data-derived rule.
        """

        if magnitude.ndim not in (3, 4):
            raise ValueError("magnitude AF2RN harus NCK atau NCHW")
        original_shape = magnitude.shape
        flat = magnitude.reshape(*magnitude.shape[:2], -1)
        if flat.shape[-1] != self.annulus_bin.numel():
            raise ValueError(
                "ukuran magnitude tidak cocok dengan grid FFT AF2RN: "
                f"{flat.shape[-1]} != {self.annulus_bin.numel()}"
            )

        ring_index = self.annulus_bin.to(flat.device).reshape(-1)
        normalized = torch.zeros_like(flat)
        for ring_id in range(self.annulus_count):
            selected = ring_index == ring_id
            values = flat[..., selected]
            median = values.median(dim=-1, keepdim=True).values
            normalized[..., selected] = torch.relu(
                values / median.clamp_min(self.af2rn_config.eps) - 1.0
            )
        return normalized.reshape(original_shape)

    def angular_density(self, shifted_frequency: torch.Tensor) -> torch.Tensor:
        """Accumulate radially normalized excess into legacy signed directions."""

        n, channels, m, _ = shifted_frequency.shape
        magnitude = self.radial_normalize_magnitude(
            shifted_frequency.abs().reshape(n, channels, m * m)
        )
        index = (
            self.angle_bin.to(magnitude.device)
            .reshape(1, 1, -1)
            .expand(n, channels, -1)
        )
        density = magnitude.new_zeros(
            (n, channels, self.af2rn_config.angular_bins)
        )
        density.scatter_add_(dim=-1, index=index, src=magnitude)
        return density

    def _af2_weight(self, shifted_frequency: torch.Tensor) -> torch.Tensor:
        n, channels, m, _ = shifted_frequency.shape
        density = self.angular_density(shifted_frequency)
        probability = density / density.sum(dim=-1, keepdim=True).clamp_min(
            self.af2rn_config.eps
        )
        threshold = af2_entropy_threshold(
            probability,
            gamma=self.af2rn_config.gamma,
            eps=self.af2rn_config.eps,
        )
        normalized_density = density / density.amax(
            dim=-1, keepdim=True
        ).clamp_min(self.af2rn_config.eps)
        direction_weight = torch.where(
            normalized_density <= threshold.unsqueeze(-1),
            torch.zeros_like(normalized_density),
            normalized_density,
        )
        index = (
            self.angle_bin.to(density.device)
            .reshape(1, 1, -1)
            .expand(n, channels, -1)
        )
        pixel_weight = torch.gather(direction_weight, dim=-1, index=index)
        return pixel_weight.reshape(n, channels, m, m)
