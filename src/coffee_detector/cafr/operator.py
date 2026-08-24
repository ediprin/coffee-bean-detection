from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.afab.operator import af2_entropy_threshold, minmax_spatial

from .config import CAFRConfig


def rgb_luminance(value: torch.Tensor) -> torch.Tensor:
    """Rec.709 luminance used only as the spectral guide signal."""

    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("CAFR membutuhkan BCHW RGB")
    coefficients = value.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return (value * coefficients).sum(dim=1, keepdim=True)


def shared_residual_gate(raw_rgb: torch.Tensor, recovered_luminance: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply one shared gain to RGB so channel ratios are preserved exactly.

    X' = X + X * G, with G=N(R) in [0,1] and one channel shared by R/G/B.
    For any non-zero channels, R'/G' == R/G and G'/B' == G/B.
    """

    if recovered_luminance.ndim != 4 or recovered_luminance.shape[1] != 1:
        raise ValueError("recovered_luminance harus B1HW")
    gate = minmax_spatial(recovered_luminance, eps=eps)
    return raw_rgb + raw_rgb * gate


def soft_spectral_weight(normalized_density: torch.Tensor, threshold: torch.Tensor, temperature: float) -> torch.Tensor:
    """Entropy-conditioned soft alternative to Xu AFAB-2 hard suppression."""

    return torch.sigmoid((normalized_density - threshold.unsqueeze(-1)) / float(temperature))


class CAFRInputEnhancer(nn.Module):
    """Coffee-Adaptive Frequency Representation (CAFR).

    The operator deliberately keeps the native RGB image and uses a shared luminance-derived
    spectral gate as a complementary cue.  It differs from the transferred AF2 parent in four
    explicit ways: shared chromaticity-preserving gating, radial x directional density,
    optional soft selection, and an externally calibrated patch size.

    No learned parameters or persistent buffers are introduced.
    """

    def __init__(self, config: CAFRConfig | dict | None = None) -> None:
        super().__init__()
        self.config = CAFRConfig.from_mapping(config)
        radius_norm, angle_bin, radial_bin = self._frequency_geometry()
        self.register_buffer("frequency_radius_norm", radius_norm, persistent=False)
        self.register_buffer("angle_bin", angle_bin, persistent=False)
        self.register_buffer("radial_bin", radial_bin, persistent=False)

    def _frequency_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        target_h, target_w = max(h, m), max(w, m)
        pad_h = (stride - ((target_h - m) % stride)) % stride + target_h - h
        pad_w = (stride - ((target_w - m) % stride)) % stride + target_w - w
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
        return image, (h, w)

    def _spectral_weight(self, shifted_frequency: torch.Tensor) -> torch.Tensor:
        """Compute W(b,theta) and map it back to every Fourier coefficient."""

        n, channels, m, _ = shifted_frequency.shape
        if channels != 1:
            raise ValueError("CAFR spectral guide harus luminance satu channel")
        magnitude = shifted_frequency.abs().reshape(n, 1, m * m)
        angle = self.angle_bin.to(magnitude.device).reshape(1, 1, -1)
        radial = self.radial_bin.to(magnitude.device).reshape(1, 1, -1)
        combined = radial * self.config.angular_bins + angle
        combined = combined.expand(n, 1, -1)
        size = self.config.radial_bands * self.config.angular_bins

        density = magnitude.new_zeros((n, 1, size))
        density.scatter_add_(-1, combined, magnitude)
        density = density.reshape(n, 1, self.config.radial_bands, self.config.angular_bins)

        probability = density / density.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        threshold = af2_entropy_threshold(probability, gamma=self.config.gamma, eps=self.config.eps)
        normalized = density / density.amax(dim=-1, keepdim=True).clamp_min(self.config.eps)

        if self.config.soft_selection:
            selected = soft_spectral_weight(normalized, threshold, self.config.soft_temperature)
        else:
            selected = torch.where(
                normalized <= threshold.unsqueeze(-1),
                torch.zeros_like(normalized),
                normalized,
            )

        selected = selected.reshape(n, 1, size)
        pixel_weight = torch.gather(selected, dim=-1, index=combined)
        return pixel_weight.reshape(n, 1, m, m)

    def _filter_patch_chunk(self, patches: torch.Tensor) -> torch.Tensor:
        original_dtype = patches.dtype
        with torch.autocast(device_type=patches.device.type, enabled=False):
            work = patches.float()
            frequency = torch.fft.fftshift(
                torch.fft.fft2(work, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
            )
            filtered = frequency * self._spectral_weight(frequency)
            recovered = torch.fft.ifft2(
                torch.fft.ifftshift(filtered, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ).real
        return recovered.to(original_dtype)

    def _recover_one(self, luminance: torch.Tensor) -> torch.Tensor:
        padded, original_shape = self._pad_for_windows(luminance)
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
        divisor_columns = torch.ones(
            (1, m * m, count), device=luminance.device, dtype=luminance.dtype
        )
        divisor = F.fold(divisor_columns, output_size=(hp, wp), kernel_size=m, stride=stride)
        recovered = recovered_sum / divisor.clamp_min(self.config.eps)
        h, w = original_shape
        return recovered[:, :, :h, :w]

    def recover(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("CAFR membutuhkan BCHW RGB")
        if not torch.is_floating_point(value):
            raise TypeError("CAFR membutuhkan tensor floating point")
        luminance = rgb_luminance(value)
        return torch.cat(
            [self._recover_one(luminance[i : i + 1]) for i in range(luminance.shape[0])],
            dim=0,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        recovered = self.recover(value)
        return shared_residual_gate(value, recovered, eps=self.config.eps)
