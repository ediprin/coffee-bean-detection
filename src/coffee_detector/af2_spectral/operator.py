from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.afab.operator import (
    AFABConfig,
    AFABInputEnhancer,
    af2_entropy_threshold,
    afab_gate,
    minmax_spatial,
)

from .config import SpectralFrontendConfig


_PATCH_ARMS = {"AF2WIN", "AF2ORI", "AF2POL", "AF2SOFT", "AF2LUM"}
_POLAR_ARMS = {"AF2POL", "AF2SOFT", "AF2LUM"}
_SOFT_ARMS = {"AF2SOFT", "AF2LUM"}


def rgb_luminance(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("luminance membutuhkan BCHW RGB")
    coefficients = value.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return (value * coefficients).sum(dim=1, keepdim=True)


def soft_direction_weight(
    normalized_density: torch.Tensor,
    threshold: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return normalized_density * torch.sigmoid(
        (normalized_density - threshold.unsqueeze(-1)) / float(temperature)
    )


def _periodic_sqrt_hann(size: int) -> torch.Tensor:
    return torch.hann_window(size, periodic=True, dtype=torch.float32).sqrt()


def haar_dwt2(value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Orthonormal 2-D Haar analysis returning [LL,LH,HL,HH]."""

    if value.ndim != 4:
        raise ValueError("Haar DWT membutuhkan BCHW")
    h, w = value.shape[-2:]
    pad_h, pad_w = h % 2, w % 2
    work = F.pad(value, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else value
    root = 1.0 / math.sqrt(2.0)
    low = value.new_tensor((root, root))
    high = value.new_tensor((-root, root))
    filters = torch.stack(
        (
            torch.outer(low, low),
            torch.outer(low, high),
            torch.outer(high, low),
            torch.outer(high, high),
        )
    )
    channels = value.shape[1]
    weight = filters[:, None].repeat(channels, 1, 1, 1)
    output = F.conv2d(work, weight, stride=2, groups=channels)
    output = output.reshape(value.shape[0], channels, 4, output.shape[-2], output.shape[-1])
    return output, (h, w)


def haar_idwt2(bands: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if bands.ndim != 5 or bands.shape[2] != 4:
        raise ValueError("Haar inverse membutuhkan BC4HW")
    root = 1.0 / math.sqrt(2.0)
    low = bands.new_tensor((root, root))
    high = bands.new_tensor((-root, root))
    filters = torch.stack(
        (
            torch.outer(low, low),
            torch.outer(low, high),
            torch.outer(high, low),
            torch.outer(high, high),
        )
    )
    b, channels, _, h, w = bands.shape
    weight = filters[:, None].repeat(channels, 1, 1, 1)
    packed = bands.reshape(b, channels * 4, h, w)
    output = F.conv_transpose2d(packed, weight, stride=2, groups=channels)
    return output[..., : shape[0], : shape[1]]


class SpectralInputEnhancer(nn.Module):
    """Parameter-free input frontend for the eight-arm AF2 factorization."""

    def __init__(self, config: SpectralFrontendConfig | dict | None = None) -> None:
        super().__init__()
        self.config = SpectralFrontendConfig.from_mapping(config)
        self.legacy = None
        if self.config.arm == "AF2C":
            self.legacy = AFABInputEnhancer(
                AFABConfig(
                    mode="af2",
                    patch_size=32,
                    overlap=0.50,
                    gamma=0.10,
                    angular_bins=360,
                    chunk_size=self.config.chunk_size,
                    eps=self.config.eps,
                )
            )
        if self.config.arm in _PATCH_ARMS:
            radius, angle_bin, radial_bin = self._frequency_geometry()
            window_1d = _periodic_sqrt_hann(self.config.patch_size)
            self.register_buffer("frequency_radius", radius, persistent=False)
            self.register_buffer("angle_bin", angle_bin, persistent=False)
            self.register_buffer("radial_bin", radial_bin, persistent=False)
            self.register_buffer(
                "analysis_window", torch.outer(window_1d, window_1d), persistent=False
            )

    def _frequency_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.config.patch_size
        center = m // 2
        coord = torch.arange(m, dtype=torch.float32) - float(center)
        yy, xx = torch.meshgrid(coord, coord, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square())
        period = 360.0 if self.config.arm == "AF2WIN" else 180.0
        degrees = torch.remainder(torch.rad2deg(torch.atan2(yy, xx)), period)
        angle = torch.floor(degrees / (period / self.config.angular_bins)).long()
        angle.clamp_(0, self.config.angular_bins - 1)
        if self.config.radial_bands == 1:
            radial = torch.zeros_like(angle)
        else:
            nonzero = radius[radius > 0].flatten()
            boundaries = torch.quantile(
                nonzero,
                torch.arange(1, self.config.radial_bands, dtype=torch.float32)
                / self.config.radial_bands,
            )
            radial = torch.bucketize(radius, boundaries).long()
            radial.clamp_(0, self.config.radial_bands - 1)
        return radius, angle, radial

    def _pad_for_windows(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        _, _, h, w = image.shape
        m, stride = self.config.patch_size, self.config.stride
        # A Hann window is zero at its first sample. Symmetric context padding
        # keeps every real image pixel inside a non-zero overlap region.
        top = left = stride
        base_h, base_w = h + 2 * stride, w + 2 * stride
        target_h, target_w = max(base_h, m), max(base_w, m)
        extra_h = (stride - ((target_h - m) % stride)) % stride
        extra_w = (stride - ((target_w - m) % stride)) % stride
        bottom, right = stride + extra_h, stride + extra_w
        image = F.pad(image, (left, right, top, bottom), mode="replicate")
        return image, (h, w, top, left)

    def _direction_weight(self, shifted_frequency: torch.Tensor) -> torch.Tensor:
        n, channels, m, _ = shifted_frequency.shape
        guide = shifted_frequency
        if self.config.arm == "AF2LUM":
            coefficients = shifted_frequency.real.new_tensor((0.2126, 0.7152, 0.0722))
            guide = (shifted_frequency * coefficients.view(1, 3, 1, 1)).sum(1, keepdim=True)
        magnitude = guide.abs().reshape(n, guide.shape[1], m * m)
        angle = self.angle_bin.to(magnitude.device).reshape(1, 1, -1)
        if self.config.arm in _POLAR_ARMS:
            radial = self.radial_bin.to(magnitude.device).reshape(1, 1, -1)
            combined = radial * self.config.angular_bins + angle
            combined = combined.expand(n, magnitude.shape[1], -1)
            size = self.config.radial_bands * self.config.angular_bins
            density = magnitude.new_zeros((n, magnitude.shape[1], size))
            density.scatter_add_(-1, combined, magnitude)
            density = density.reshape(
                n, magnitude.shape[1], self.config.radial_bands, self.config.angular_bins
            )
            probability = density / density.sum(-1, keepdim=True).clamp_min(self.config.eps)
            threshold = af2_entropy_threshold(probability, self.config.gamma, self.config.eps)
            normalized = density / density.amax(-1, keepdim=True).clamp_min(self.config.eps)
            if self.config.arm in _SOFT_ARMS:
                direction = soft_direction_weight(
                    normalized, threshold, self.config.soft_temperature
                )
            else:
                direction = torch.where(
                    normalized <= threshold.unsqueeze(-1),
                    torch.zeros_like(normalized),
                    normalized,
                )
            direction = direction.reshape(n, magnitude.shape[1], size)
            pixel = torch.gather(direction, -1, combined)
        else:
            index = angle.expand(n, magnitude.shape[1], -1)
            density = magnitude.new_zeros(
                (n, magnitude.shape[1], self.config.angular_bins)
            )
            density.scatter_add_(-1, index, magnitude)
            probability = density / density.sum(-1, keepdim=True).clamp_min(self.config.eps)
            threshold = af2_entropy_threshold(probability, self.config.gamma, self.config.eps)
            normalized = density / density.amax(-1, keepdim=True).clamp_min(self.config.eps)
            direction = torch.where(
                normalized <= threshold.unsqueeze(-1),
                torch.zeros_like(normalized),
                normalized,
            )
            pixel = torch.gather(direction, -1, index)
        pixel = pixel.reshape(n, guide.shape[1], m, m)
        return pixel.expand(n, channels, m, m) if pixel.shape[1] == 1 else pixel

    def _filter_patch_chunk(self, patches: torch.Tensor) -> torch.Tensor:
        dtype = patches.dtype
        with torch.autocast(device_type=patches.device.type, enabled=False):
            window = self.analysis_window.to(patches.device, torch.float32)
            work = patches.float() * window
            frequency = torch.fft.fftshift(
                torch.fft.fft2(work, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
            )
            filtered = frequency * self._direction_weight(frequency)
            recovered = torch.fft.ifft2(
                torch.fft.ifftshift(filtered, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ).real
            recovered = recovered * window
        return recovered.to(dtype)

    def _recover_patchwise_one(self, image: torch.Tensor) -> torch.Tensor:
        padded, original_shape = self._pad_for_windows(image)
        _, channels, hp, wp = padded.shape
        m, stride = self.config.patch_size, self.config.stride
        columns = F.unfold(padded, kernel_size=m, stride=stride)
        count = columns.shape[-1]
        patches = columns.transpose(1, 2).reshape(count, channels, m, m)
        recovered_columns = torch.empty_like(columns)
        for start in range(0, count, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, count)
            current = self._filter_patch_chunk(patches[start:stop])
            recovered_columns[:, :, start:stop] = (
                current.reshape(stop - start, -1).transpose(0, 1).unsqueeze(0)
            )
        recovered_sum = F.fold(
            recovered_columns, output_size=(hp, wp), kernel_size=m, stride=stride
        )
        window_square = self.analysis_window.to(image.device, image.dtype).square()
        divisor_columns = window_square.reshape(1, -1, 1).repeat(1, 1, count)
        divisor = F.fold(
            divisor_columns, output_size=(hp, wp), kernel_size=m, stride=stride
        )
        recovered = recovered_sum / divisor.clamp_min(self.config.eps)
        h, w, top, left = original_shape
        return recovered[:, :, top : top + h, left : left + w]

    def _phase_congruency(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        luminance = rgb_luminance(value).float()
        _, _, h, w = luminance.shape
        fy = torch.fft.fftfreq(h, device=value.device)
        fx = torch.fft.fftfreq(w, device=value.device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        radius = torch.sqrt(xx.square() + yy.square()).clamp_min(self.config.eps)
        theta = torch.atan2(-yy, xx)
        lowpass = 1.0 / (
            1.0
            + (radius / self.config.phase_lowpass_cutoff).pow(
                2 * self.config.phase_lowpass_order
            )
        )
        spectrum = torch.fft.fft2(luminance, dim=(-2, -1))
        orientation_maps = []
        theta_sigma = (
            math.pi / self.config.phase_orientations
        ) / self.config.phase_dtheta_on_sigma
        for orientation in range(self.config.phase_orientations):
            center = orientation * math.pi / self.config.phase_orientations
            # A log-Gabor orientation is an axis, not a signed direction:
            # theta and theta + pi must select the same filter.
            delta = 0.5 * torch.atan2(
                torch.sin(2.0 * (theta - center)),
                torch.cos(2.0 * (theta - center)),
            )
            angular = torch.exp(-delta.square() / (2.0 * theta_sigma**2))
            responses, amplitudes = [], []
            wavelength = self.config.phase_min_wavelength
            for _ in range(self.config.phase_scales):
                center_frequency = 1.0 / wavelength
                log_gabor = torch.exp(
                    -torch.log(radius / center_frequency).square()
                    / (2.0 * math.log(self.config.phase_sigma_on_f) ** 2)
                )
                log_gabor = log_gabor * lowpass
                log_gabor[0, 0] = 0.0
                response = torch.fft.ifft2(
                    spectrum * (log_gabor * angular), dim=(-2, -1)
                )
                responses.append(response)
                amplitudes.append(response.abs())
                wavelength *= self.config.phase_multiplier
            sum_response = torch.stack(responses).sum(0)
            sum_amplitude = torch.stack(amplitudes).sum(0)
            mean_norm = sum_response.abs().clamp_min(self.config.eps)
            mean_even = sum_response.real / mean_norm
            mean_odd = sum_response.imag / mean_norm
            energy = sum(
                response.real * mean_even
                + response.imag * mean_odd
                - torch.abs(response.real * mean_odd - response.imag * mean_even)
                for response in responses
            )
            smallest = amplitudes[0]
            tau = smallest.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
            tau = tau / math.sqrt(math.log(4.0))
            geometric = (
                1.0 - self.config.phase_multiplier ** (-self.config.phase_scales)
            ) / (1.0 - self.config.phase_multiplier ** -1)
            total_tau = tau * geometric
            noise = total_tau * (
                math.sqrt(math.pi / 2.0)
                + self.config.phase_noise_k * math.sqrt((4.0 - math.pi) / 2.0)
            )
            width = (
                sum_amplitude
                / (torch.stack(amplitudes).amax(0) + self.config.eps)
                - 1.0
            ) / max(self.config.phase_scales - 1, 1)
            spread = 1.0 / (1.0 + torch.exp((0.5 - width) * 10.0))
            orientation_maps.append(
                spread * F.relu(energy - noise) / sum_amplitude.clamp_min(self.config.eps)
            )
        cue = torch.stack(orientation_maps).amax(0).clamp(0.0, 1.0)
        return cue.to(dtype)

    def _wavelet_energy(self, value: torch.Tensor) -> torch.Tensor:
        luminance = rgb_luminance(value)
        target = luminance.shape[-2:]
        current = luminance
        energies = []
        for _ in range(self.config.wavelet_levels):
            bands, _ = haar_dwt2(current)
            current = bands[:, :, 0]
            detail = bands[:, :, 1:].square().sum(2).add(self.config.eps).sqrt()
            detail = F.interpolate(detail, size=target, mode="bilinear", align_corners=False)
            energies.append(minmax_spatial(detail, self.config.eps))
        return torch.stack(energies).mean(0)

    def recover(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("frontend membutuhkan BCHW RGB")
        if not torch.is_floating_point(value):
            raise TypeError("frontend membutuhkan tensor floating point")
        if self.config.arm == "AF2C":
            assert self.legacy is not None
            return self.legacy.recover(value)
        if self.config.arm in _PATCH_ARMS:
            return torch.cat(
                [self._recover_patchwise_one(value[i : i + 1]) for i in range(value.shape[0])]
            )
        cue = (
            self._phase_congruency(value)
            if self.config.arm == "PCG1"
            else self._wavelet_energy(value)
        )
        return cue.expand_as(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.config.arm == "AF2C":
            assert self.legacy is not None
            return self.legacy(value)
        return afab_gate(value, self.recover(value), eps=self.config.eps)
