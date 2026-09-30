from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _same_stationary_response(
    value: torch.Tensor,
    kernel: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Apply a 2x2 dilated filter while preserving the input grid."""

    if dilation <= 0:
        raise ValueError("Dilation harus positif")
    # A 2x2 dilated kernel spans dilation+1 pixels. Right/bottom padding keeps
    # the response aligned with the original top-left sampling grid.
    padded = F.pad(value, (0, dilation, 0, dilation), mode="replicate")
    return F.conv2d(padded, kernel, dilation=dilation)


def stationary_haar_energy(image: torch.Tensor, *, levels: int = 2) -> torch.Tensor:
    """Return shift-preserving luminance Haar detail energies.

    Output order per level is horizontal, vertical, and diagonal energy. No
    trainable parameter, dataset statistic, or spatial downsampling is used.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("FSRC image harus [B,3,H,W]")
    if not image.is_floating_point():
        raise TypeError("FSRC image harus floating point")
    if levels <= 0:
        raise ValueError("Jumlah level Haar harus positif")
    original_dtype = image.dtype
    work = image.float()
    red, green, blue = work[:, 0:1], work[:, 1:2], work[:, 2:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    root = math.sqrt(2.0)
    low = work.new_tensor([1.0, 1.0]) / root
    high = work.new_tensor([-1.0, 1.0]) / root
    kernels = torch.stack(
        (
            torch.outer(low, high),
            torch.outer(high, low),
            torch.outer(high, high),
        )
    )[:, None]
    responses = []
    for level in range(levels):
        dilation = 2**level
        response = _same_stationary_response(luminance, kernels, dilation).abs()
        scale = response.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        responses.append((response / scale).clamp(0.0, 8.0) / 8.0)
    result = torch.cat(responses, dim=1).to(original_dtype)
    if result.shape[-2:] != image.shape[-2:] or not torch.isfinite(result).all():
        raise RuntimeError("Stationary Haar menghasilkan cue tidak valid")
    return result
