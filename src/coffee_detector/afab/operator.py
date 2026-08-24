from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AFABConfig:
    mode: str = "af12"  # af1 | af2 | af12
    patch_size: int = 32
    overlap: float = 0.50
    radius_ratio: float = 0.05
    gamma: float = 0.10
    angular_bins: int = 360
    chunk_size: int = 128
    eps: float = 1e-8

    @classmethod
    def from_mapping(cls, payload: "AFABConfig | dict[str, Any] | None") -> "AFABConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.mode not in {"af1", "af2", "af12"}:
            raise ValueError("mode AFAB harus af1, af2, atau af12")
        if result.patch_size <= 1:
            raise ValueError("patch_size harus >1")
        if not 0.0 <= result.overlap < 1.0:
            raise ValueError("overlap harus di [0,1)")
        stride = int(round(result.patch_size * (1.0 - result.overlap)))
        if stride <= 0:
            raise ValueError("overlap menghasilkan stride nol")
        if result.radius_ratio <= 0 or result.gamma <= 0:
            raise ValueError("radius_ratio/gamma harus positif")
        if result.angular_bins <= 1 or result.chunk_size <= 0 or result.eps <= 0:
            raise ValueError("angular_bins/chunk_size/eps tidak valid")
        return result

    @property
    def stride(self) -> int:
        return int(round(self.patch_size * (1.0 - self.overlap)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def af2_entropy_threshold(probability: torch.Tensor, gamma: float, eps: float = 1e-8) -> torch.Tensor:
    """LFDet AFAB-2 Eq. (10)-(11) on normalized angular density."""
    entropy = -(probability * torch.log(probability.clamp_min(eps))).sum(dim=-1)
    return float(gamma) / (1.0 + torch.exp(-entropy))


def minmax_spatial(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-image, per-channel min-max normalization of recovered spatial content."""
    low = value.amin(dim=(-2, -1), keepdim=True)
    high = value.amax(dim=(-2, -1), keepdim=True)
    return (value - low) / (high - low).clamp_min(eps)


def afab_gate(raw: torch.Tensor, recovered: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """LFDet-described gating: raw * normalized(recovered), followed by residual addition."""
    return raw + raw * minmax_spatial(recovered, eps=eps)


class AFABInputEnhancer(nn.Module):
    """Patch-wise AFAB input enhancer transferred from LFDet.

    Paper-derived settings/equations retained:
      * patch size m=32;
      * AFAB-1 patch energy and adaptive high-pass radius, r=0.05;
      * AFAB-2 angular-density entropy threshold, gamma=0.1;
      * inverse DFT followed by min-max gate and residual addition.

    Explicit transfer choices because the paper text does not specify them:
      * RGB channels are transformed independently;
      * theta in [0, 360 deg) is discretized to nearest lower integer degree
        when angular_bins=360;
      * overlapping patches are reconstructed with fold/overlap averaging.

    FFT sections always use float32 for CUDA/AMP safety; output dtype matches input.
    """

    def __init__(self, config: AFABConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = AFABConfig.from_mapping(config)
        radius, angle_bin = self._build_frequency_geometry(
            self.config.patch_size, self.config.angular_bins
        )
        self.register_buffer("frequency_radius", radius, persistent=False)
        self.register_buffer("angle_bin", angle_bin, persistent=False)

    @staticmethod
    def _build_frequency_geometry(patch_size: int, angular_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
        center = patch_size // 2
        coord = torch.arange(patch_size, dtype=torch.float32) - float(center)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        degrees = torch.remainder(torch.rad2deg(torch.atan2(yy, xx)), 360.0)
        # Paper gives theta as a continuous degree domain but not code-level bins.
        # Floor-to-bin is a frozen transfer discretization; DC maps to bin zero.
        bins = torch.floor(degrees / (360.0 / angular_bins)).long().clamp_(0, angular_bins - 1)
        return radius, bins

    def _pad_for_windows(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = image.shape
        m, stride = self.config.patch_size, self.config.stride
        target_h = max(h, m)
        target_w = max(w, m)
        pad_h = (stride - ((target_h - m) % stride)) % stride
        pad_w = (stride - ((target_w - m) % stride)) % stride
        pad_h += target_h - h
        pad_w += target_w - w
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        return image, (h, w)

    def _af1_mask(self, energy: torch.Tensor, max_energy: torch.Tensor) -> torch.Tensor:
        """LFDet Eq. (6)-(8): patch-adaptive circular high-pass mask."""
        m = self.config.patch_size
        rb = (m / 2.0) * self.config.radius_ratio
        radius_per_patch = rb * torch.exp(
            1.0 - energy / max_energy.clamp_min(self.config.eps)
        )
        grid = self.frequency_radius.to(device=energy.device, dtype=energy.dtype)
        return (grid.view(1, 1, m, m) > radius_per_patch[..., None, None]).to(energy.dtype)

    def _af2_weight(self, shifted_frequency: torch.Tensor) -> torch.Tensor:
        """LFDet Eq. (9)-(13): entropy-conditioned directional amplitude suppression."""
        n, c, m, _ = shifted_frequency.shape
        magnitude = shifted_frequency.abs().reshape(n, c, m * m)
        index = self.angle_bin.to(device=magnitude.device).reshape(1, 1, -1).expand(n, c, -1)
        density = magnitude.new_zeros((n, c, self.config.angular_bins))
        density.scatter_add_(dim=-1, index=index, src=magnitude)
        total = density.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        probability = density / total
        threshold = af2_entropy_threshold(
            probability, gamma=self.config.gamma, eps=self.config.eps
        )
        normalized_density = density / density.amax(dim=-1, keepdim=True).clamp_min(self.config.eps)
        direction_weight = torch.where(
            normalized_density <= threshold.unsqueeze(-1),
            torch.zeros_like(normalized_density),
            normalized_density,
        )
        pixel_weight = torch.gather(direction_weight, dim=-1, index=index)
        return pixel_weight.reshape(n, c, m, m)

    def _filter_patch_chunk(
        self,
        patches: torch.Tensor,
        *,
        energy: torch.Tensor | None = None,
        max_energy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_dtype = patches.dtype
        device_type = patches.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            work = patches.float()
            frequency = torch.fft.fft2(work, dim=(-2, -1), norm="ortho")
            frequency = torch.fft.fftshift(frequency, dim=(-2, -1))
            if self.config.mode in {"af1", "af12"}:
                if energy is None or max_energy is None:
                    raise ValueError("AFAB-1 memerlukan energy dan max_energy")
                frequency = frequency * self._af1_mask(energy.float(), max_energy.float())
            if self.config.mode in {"af2", "af12"}:
                frequency = frequency * self._af2_weight(frequency)
            recovered = torch.fft.ifft2(
                torch.fft.ifftshift(frequency, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ).real
        return recovered.to(dtype=original_dtype)

    def _recover_one(self, image: torch.Tensor) -> torch.Tensor:
        """Recover one [1,C,H,W] image from overlapping filtered patches."""
        padded, original_shape = self._pad_for_windows(image)
        _, channels, hp, wp = padded.shape
        m, stride = self.config.patch_size, self.config.stride
        columns = F.unfold(padded, kernel_size=m, stride=stride)
        count = columns.shape[-1]
        patches = columns.transpose(1, 2).reshape(count, channels, m, m)

        energy_all = None
        max_energy = None
        if self.config.mode in {"af1", "af12"}:
            # Parseval with orthonormal FFT: spatial sum(x^2) == spectral energy.
            energy_all = patches.float().square().sum(dim=(-2, -1))
            max_energy = energy_all.amax(dim=0, keepdim=True)

        recovered_columns = torch.empty_like(columns)
        for start in range(0, count, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, count)
            current = patches[start:stop]
            filtered = self._filter_patch_chunk(
                current,
                energy=None if energy_all is None else energy_all[start:stop],
                max_energy=max_energy,
            )
            recovered_columns[:, :, start:stop] = filtered.reshape(stop - start, -1).transpose(0, 1).unsqueeze(0)

        recovered_sum = F.fold(
            recovered_columns,
            output_size=(hp, wp),
            kernel_size=m,
            stride=stride,
        )
        ones = torch.ones(
            (1, m * m, count), device=image.device, dtype=image.dtype
        )
        divisor = F.fold(ones, output_size=(hp, wp), kernel_size=m, stride=stride)
        recovered = recovered_sum / divisor.clamp_min(self.config.eps)
        h, w = original_shape
        return recovered[:, :, :h, :w]

    def recover(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"AFAB membutuhkan BCHW, diterima {tuple(value.shape)}")
        if value.shape[1] != 3:
            raise ValueError("Transfer AFAB ini dikunci untuk input RGB 3-channel")
        if not torch.is_floating_point(value):
            raise TypeError("AFAB memerlukan tensor floating point")
        return torch.cat([self._recover_one(value[i : i + 1]) for i in range(value.shape[0])], dim=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        recovered = self.recover(value)
        return afab_gate(value, recovered, eps=self.config.eps)
