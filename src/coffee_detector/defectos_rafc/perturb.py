from __future__ import annotations

import math

import torch

from .config import RAFCConfig


def _centered_frequency_grid(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=device)).to(dtype)
    fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=device)).to(dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(xx.square() + yy.square()), torch.atan2(yy, xx)


def _smooth_low_frequency_mask(
    radius: torch.Tensor, low: float, high: float
) -> torch.Tensor:
    if high <= low:
        raise ValueError("high harus lebih besar daripada low")
    transition = ((radius - low) / (high - low)).clamp(0.0, 1.0)
    return torch.where(
        radius <= low,
        torch.ones_like(radius),
        torch.where(
            radius >= high,
            torch.zeros_like(radius),
            0.5 * (1.0 + torch.cos(math.pi * transition)),
        ),
    )


def _unsigned_angular_mask(
    angle: torch.Tensor,
    centers: torch.Tensor,
    width_degrees: float,
) -> torch.Tensor:
    """Centrosymmetric soft sectors; FFT magnitude orientation is modulo pi."""

    period = math.pi
    difference = torch.remainder(
        angle[None, None] - centers[:, None, None, None] + period / 2.0,
        period,
    ) - period / 2.0
    half_width = math.radians(width_degrees) / 2.0
    normalized = (difference.abs() / max(half_width, 1.0e-12)).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * normalized))


def angular_low_frequency_amplitude_mix(
    image: torch.Tensor,
    config: RAFCConfig | dict | None = None,
    *,
    orientation_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mix donor amplitude in weak low-frequency unsigned-orientation sectors.

    The source phase is retained and the mask is centrosymmetric, so spatial
    geometry and the real-valued inverse FFT contract are preserved. Donors are
    a cyclic batch permutation. For a single-image batch the donor amplitude is
    contrast-scaled so the audit and final partial batch remain active.
    """

    cfg = RAFCConfig.from_mapping(config)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"RAFC mengharapkan BCHW RGB, diterima {tuple(image.shape)}")
    if not torch.is_floating_point(image):
        raise TypeError("RAFC memerlukan input floating point")
    batch, _, height, width = image.shape
    work = image.float() if image.dtype in {torch.float16, torch.bfloat16} else image
    spectrum = torch.fft.fftshift(torch.fft.fft2(work, dim=(-2, -1)), dim=(-2, -1))
    amplitude, phase = spectrum.abs(), torch.angle(spectrum)
    if batch > 1:
        donor = amplitude.roll(1, dims=0)
    else:
        # A deterministic, bounded fallback; DC stays source-controlled below.
        donor = amplitude * (1.0 + cfg.mix_strength)
    radius, angle = _centered_frequency_grid(
        height, width, device=work.device, dtype=work.dtype
    )
    radial = _smooth_low_frequency_mask(radius, cfg.low_radius, cfg.high_radius)
    if orientation_indices is None:
        orientation_indices = torch.randint(
            cfg.orientation_bins, (batch,), device=work.device
        )
    orientation_indices = orientation_indices.to(device=work.device, dtype=torch.long)
    if orientation_indices.shape != (batch,):
        raise ValueError("orientation_indices RAFC harus berbentuk [B]")
    centers = orientation_indices.to(work.dtype) * (math.pi / cfg.orientation_bins)
    angular = _unsigned_angular_mask(angle, centers, cfg.angular_width_degrees)
    mask = radial[None, None] * angular
    # Never alter the exact DC coefficient; brightness remains raw-anchored.
    mask[..., height // 2, width // 2] = 0.0
    mixed_amplitude = amplitude + cfg.mix_strength * mask * (donor - amplitude)
    mixed = torch.polar(mixed_amplitude.clamp_min(cfg.eps), phase)
    spatial = torch.fft.ifft2(
        torch.fft.ifftshift(mixed, dim=(-2, -1)), dim=(-2, -1)
    ).real
    return spatial.clamp(0.0, 1.0).to(image.dtype)
