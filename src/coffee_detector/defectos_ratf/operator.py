from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import RATFConfig


def _luminance(image: torch.Tensor) -> torch.Tensor:
    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _standardize(value: torch.Tensor, clip: float) -> torch.Tensor:
    mean = value.mean(dim=(-2, -1), keepdim=True)
    scale = value.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
    return ((value - mean) / scale).clamp(-clip, clip) / clip


def _haar_level3_vertical(luminance: torch.Tensor) -> torch.Tensor:
    """Return the level-3 low-horizontal/high-vertical Haar response."""

    root = math.sqrt(2.0)
    low = torch.tensor([1.0, 1.0], device=luminance.device, dtype=luminance.dtype) / root
    high = torch.tensor([-1.0, 1.0], device=luminance.device, dtype=luminance.dtype) / root
    ll_kernel = torch.outer(low, low).view(1, 1, 2, 2)
    vertical_kernel = torch.outer(high, low).view(1, 1, 2, 2)
    approximation = luminance
    vertical = None
    for _ in range(3):
        vertical = F.conv2d(approximation, vertical_kernel, stride=2)
        approximation = F.conv2d(approximation, ll_kernel, stride=2)
    if vertical is None:
        raise RuntimeError("DWT level-3 tidak terbentuk")
    return vertical.abs()


def _gabor_energy(
    luminance: torch.Tensor,
    *,
    theta_degrees: float,
    wavelength: float,
    sigma: float,
    gamma: float,
    kernel_size: int,
) -> torch.Tensor:
    radius = kernel_size // 2
    axis = torch.arange(
        -radius, radius + 1, device=luminance.device, dtype=luminance.dtype
    )
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    theta = math.radians(theta_degrees)
    rotated_x = xx * math.cos(theta) + yy * math.sin(theta)
    rotated_y = -xx * math.sin(theta) + yy * math.cos(theta)
    envelope = torch.exp(
        -(rotated_x.square() + gamma**2 * rotated_y.square()) / (2.0 * sigma**2)
    )
    phase = 2.0 * math.pi * rotated_x / wavelength
    real = envelope * torch.cos(phase)
    imaginary = envelope * torch.sin(phase)
    real = real - real.mean()
    imaginary = imaginary - imaginary.mean()
    real = real / real.square().sum().sqrt().clamp_min(1e-8)
    imaginary = imaginary / imaginary.square().sum().sqrt().clamp_min(1e-8)
    real_response = F.conv2d(luminance, real.view(1, 1, kernel_size, kernel_size), padding=radius)
    imaginary_response = F.conv2d(
        luminance, imaginary.view(1, 1, kernel_size, kernel_size), padding=radius
    )
    return torch.sqrt(real_response.square() + imaginary_response.square() + 1e-8)


def ratf_texture_cue(
    image: torch.Tensor,
    config: RATFConfig | dict | None = None,
) -> torch.Tensor:
    """Extract the three frozen texture cues selected for RATF1.

    The operator uses luminance only: one coarse level-3 directional Haar
    response and two low-frequency oblique Gabor energies.  It deliberately
    excludes color statistics, high-frequency diagonal wavelets, and any
    dataset-fitted threshold.
    """

    cfg = RATFConfig.from_mapping(config)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("RATF input harus [B,3,H,W]")
    if not image.is_floating_point():
        raise TypeError("RATF input harus floating point")
    original_dtype = image.dtype
    work = image.float()
    luminance = _luminance(work)
    dwt = _haar_level3_vertical(luminance)
    dwt = F.interpolate(dwt, size=image.shape[-2:], mode="bilinear", align_corners=False)
    gabor_luminance = F.avg_pool2d(
        luminance,
        kernel_size=cfg.gabor_analysis_downsample,
        stride=cfg.gabor_analysis_downsample,
    )
    gabor = [
        _gabor_energy(
            gabor_luminance,
            theta_degrees=orientation,
            wavelength=cfg.gabor_wavelength,
            sigma=cfg.gabor_sigma,
            gamma=cfg.gabor_gamma,
            kernel_size=cfg.gabor_kernel_size,
        )
        for orientation in cfg.gabor_orientations_degrees
    ]
    gabor = [
        F.interpolate(value, size=image.shape[-2:], mode="bilinear", align_corners=False)
        for value in gabor
    ]
    cue = torch.cat(
        [_standardize(value, cfg.cue_clip) for value in (dwt, *gabor)], dim=1
    ).to(original_dtype)
    if cue.shape[1] != cfg.cue_channels or not torch.isfinite(cue).all():
        raise RuntimeError("Cue RATF tidak valid")
    return cue
