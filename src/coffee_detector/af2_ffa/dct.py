"""Fixed selected-DCT descriptor for the AF2 feature-frequency adapter.

This is deliberately not a full FcaNet implementation.  It changes only the
frequency descriptor used by FFAB2.  Eight fixed high-frequency 2-D DCT-II
projections are measured per channel and normalized by the channel RMS.  The
DCT basis is non-learned and cached per feature shape/device/dtype.
"""

from __future__ import annotations

import math

import torch


# Frozen before training.  Every pair lies safely above the original FFAB2
# radial cutoff (0.35) in normalized frequency space.  The set is intentionally
# small so the experiment tests whether sparse spectral evidence can replace a
# full rFFT2 descriptor.
DCT_HIGH_FREQUENCY_PAIRS: tuple[tuple[float, float], ...] = (
    (0.00, 0.50),
    (0.50, 0.00),
    (0.50, 0.50),
    (0.00, 0.75),
    (0.75, 0.00),
    (0.50, 0.75),
    (0.75, 0.50),
    (0.75, 0.75),
)

_BASIS_CACHE: dict[tuple[int, int, str, int | None, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _orthonormal_dct_vectors(
    length: int,
    normalized_frequencies: tuple[float, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length < 2:
        raise ValueError("DCT descriptor memerlukan feature dimension >= 2")
    positions = torch.arange(length, device=device, dtype=dtype)
    rows = []
    for normalized in normalized_frequencies:
        index = int(round(float(normalized) * (length - 1)))
        index = min(max(index, 0), length - 1)
        scale = math.sqrt(1.0 / length) if index == 0 else math.sqrt(2.0 / length)
        rows.append(
            scale
            * torch.cos(
                math.pi * (2.0 * positions + 1.0) * float(index) / (2.0 * length)
            )
        )
    return torch.stack(rows, dim=0)


def _basis(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, device.type, device.index, dtype)
    cached = _BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    ys = tuple(pair[0] for pair in DCT_HIGH_FREQUENCY_PAIRS)
    xs = tuple(pair[1] for pair in DCT_HIGH_FREQUENCY_PAIRS)
    by = _orthonormal_dct_vectors(height, ys, device=device, dtype=dtype)
    bx = _orthonormal_dct_vectors(width, xs, device=device, dtype=dtype)
    _BASIS_CACHE[key] = (by, bx)
    return by, bx


def selected_dct_descriptor(value: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Return one bounded high-frequency DCT descriptor per batch/channel.

    The selected coefficients are orthonormal DCT-II projections.  Their mean
    absolute magnitude is divided by the channel RMS, giving a scale-free
    relative spectral response.  ``r/(1+r)`` maps it to [0, 1), matching the
    bounded descriptor range expected by the existing FFAB2 gate.
    """

    if value.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(value.shape)}")
    original_dtype = value.dtype
    work = value.float()
    height, width = work.shape[-2:]
    by, bx = _basis(height, width, device=work.device, dtype=work.dtype)

    # Separable selected 2-D DCT projections.  K=8 is frozen above.
    projected_y = torch.einsum("bchw,kh->bckw", work, by)
    coefficients = torch.einsum("bckw,kw->bck", projected_y, bx)
    high_response = coefficients.abs().mean(dim=-1)
    rms = work.square().mean(dim=(-2, -1)).add(float(eps)).sqrt()
    relative = high_response / rms
    descriptor = relative / (1.0 + relative)
    return descriptor.to(original_dtype)
