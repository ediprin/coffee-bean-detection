from __future__ import annotations

import torch

from .config import DIDAAF2Config


def _uniform(
    image: torch.Tensor, low: float, high: float, shape: tuple[int, ...]
) -> torch.Tensor:
    return torch.empty(shape, device=image.device, dtype=image.dtype).uniform_(low, high)


def diversify_appearance(
    image: torch.Tensor, config: DIDAAF2Config
) -> torch.Tensor:
    """Geometry-preserving, deliberately mild source-style diversification.

    Operations are global per image/channel. There is no crop, resize, affine,
    blur, cutout, or spatial resampling, so every GT box remains unchanged.
    The transform is applied before the deterministic AF2 enhancer.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"DIDA style mengharapkan BCHW RGB, diterima {tuple(image.shape)}")
    if not torch.is_floating_point(image):
        raise TypeError("DIDA style memerlukan tensor floating point")
    batch = image.shape[0]
    contrast = _uniform(
        image, config.contrast_low, config.contrast_high, (batch, 1, 1, 1)
    )
    brightness = _uniform(
        image, -config.brightness, config.brightness, (batch, 1, 1, 1)
    )
    gamma = _uniform(image, config.gamma_low, config.gamma_high, (batch, 1, 1, 1))
    channel_gain = _uniform(
        image,
        config.channel_gain_low,
        config.channel_gain_high,
        (batch, 3, 1, 1),
    )
    mean = image.mean(dim=(-2, -1), keepdim=True)
    diversified = (image - mean) * contrast + mean + brightness
    diversified = diversified.clamp(0.0, 1.0).pow(gamma)
    return (diversified * channel_gain).clamp(0.0, 1.0)
