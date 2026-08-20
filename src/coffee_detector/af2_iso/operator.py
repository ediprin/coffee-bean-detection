from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.afab.operator import af2_entropy_threshold, afab_gate

from .config import AF2IsolatedConfig


class AF2IsolatedInputEnhancer(nn.Module):
    """AF2 with exactly one controlled geometry change.

    This implementation intentionally preserves the legacy AF2 transfer choices:
      * RGB channels are transformed independently;
      * patch size 32 and 50% overlap by default;
      * hard entropy-conditioned AFAB-2 suppression;
      * original phase is retained;
      * overlap reconstruction uses fold averaging;
      * recovered RGB is min-max gated per channel and added residually.

    The only experimental degrees of freedom are radial decomposition or folding
    signed directions into unsigned 180-degree orientations.
    """

    def __init__(self, config: AF2IsolatedConfig | dict | None = None) -> None:
        super().__init__()
        self.config = AF2IsolatedConfig.from_mapping(config)
        radius_norm, angle_bin, radial_bin = self._build_frequency_geometry()
        self.register_buffer("frequency_radius_norm", radius_norm, persistent=False)
        self.register_buffer("angle_bin", angle_bin, persistent=False)
        self.register_buffer("radial_bin", radial_bin, persistent=False)

    def _build_frequency_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.config.patch_size
        center = m // 2
        coord = torch.arange(m, dtype=torch.float32) - float(center)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")

        radius = torch.sqrt(xx.square() + yy.square())
        radius_norm = radius / radius.max().clamp_min(self.config.eps)

        period = float(self.config.orientation_period)
        degrees = torch.remainder(torch.rad2deg(torch.atan2(yy, xx)), period)
        angle = torch.floor(degrees / (period / self.config.angular_bins)).long()
        angle.clamp_(0, self.config.angular_bins - 1)

        if self.config.radial_bands == 1:
            radial = torch.zeros_like(angle)
        else:
            boundaries = torch.tensor(self.config.radial_boundaries, dtype=torch.float32)
            radial = torch.bucketize(radius_norm, boundaries).long()
            radial.clamp_(0, self.config.radial_bands - 1)
        return radius_norm, angle, radial

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

    def _af2_weight(self, shifted_frequency: torch.Tensor) -> torch.Tensor:
        """Hard AF2 selection with optional radial conditioning.

        For AF2_BASE this reduces to the original AFAB-2 angular-density equation.
        AF2_RADIAL computes the same entropy/normalization independently inside
        each frozen radial band. AF2_ORIENT changes only theta's period from
        360 degrees to 180 degrees while retaining ~1 degree/bin resolution.
        """

        n, channels, m, _ = shifted_frequency.shape
        magnitude = shifted_frequency.abs().reshape(n, channels, m * m)

        angle = self.angle_bin.to(device=magnitude.device).reshape(1, 1, -1)
        radial = self.radial_bin.to(device=magnitude.device).reshape(1, 1, -1)
        combined = radial * self.config.angular_bins + angle
        combined = combined.expand(n, channels, -1)
        size = self.config.radial_bands * self.config.angular_bins

        density = magnitude.new_zeros((n, channels, size))
        density.scatter_add_(dim=-1, index=combined, src=magnitude)
        density = density.reshape(
            n, channels, self.config.radial_bands, self.config.angular_bins
        )

        probability = density / density.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        threshold = af2_entropy_threshold(
            probability, gamma=self.config.gamma, eps=self.config.eps
        )
        normalized_density = density / density.amax(
            dim=-1, keepdim=True
        ).clamp_min(self.config.eps)
        direction_weight = torch.where(
            normalized_density <= threshold.unsqueeze(-1),
            torch.zeros_like(normalized_density),
            normalized_density,
        )

        direction_weight = direction_weight.reshape(n, channels, size)
        pixel_weight = torch.gather(direction_weight, dim=-1, index=combined)
        return pixel_weight.reshape(n, channels, m, m)

    def _filter_patch_chunk(self, patches: torch.Tensor) -> torch.Tensor:
        original_dtype = patches.dtype
        with torch.autocast(device_type=patches.device.type, enabled=False):
            work = patches.float()
            frequency = torch.fft.fftshift(
                torch.fft.fft2(work, dim=(-2, -1), norm="ortho"),
                dim=(-2, -1),
            )
            filtered = frequency * self._af2_weight(frequency)
            recovered = torch.fft.ifft2(
                torch.fft.ifftshift(filtered, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ).real
        return recovered.to(dtype=original_dtype)

    def _recover_one(self, image: torch.Tensor) -> torch.Tensor:
        padded, original_shape = self._pad_for_windows(image)
        _, channels, hp, wp = padded.shape
        m, stride = self.config.patch_size, self.config.stride
        columns = F.unfold(padded, kernel_size=m, stride=stride)
        count = columns.shape[-1]
        patches = columns.transpose(1, 2).reshape(count, channels, m, m)

        recovered_columns = torch.empty_like(columns)
        for start in range(0, count, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, count)
            filtered = self._filter_patch_chunk(patches[start:stop])
            recovered_columns[:, :, start:stop] = (
                filtered.reshape(stop - start, -1).transpose(0, 1).unsqueeze(0)
            )

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
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("AF2 isolated membutuhkan BCHW RGB 3-channel")
        if not torch.is_floating_point(value):
            raise TypeError("AF2 isolated membutuhkan tensor floating point")
        return torch.cat(
            [self._recover_one(value[i : i + 1]) for i in range(value.shape[0])],
            dim=0,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        recovered = self.recover(value)
        return afab_gate(value, recovered, eps=self.config.eps)
