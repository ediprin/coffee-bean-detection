from __future__ import annotations

import torch
import torch.nn.functional as F


def _haar_kernels(value: torch.Tensor) -> torch.Tensor:
    kernels = value.new_tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[-1.0, -1.0], [1.0, 1.0]],
            [[-1.0, 1.0], [-1.0, 1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ]
    )
    return kernels[:, None] * 0.5


def haar_bands(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return orthonormal Haar LL, LH, HL, and HH bands."""

    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("Haar input harus [B,1,H,W]")
    if value.shape[-2] < 2 or value.shape[-1] < 2:
        raise ValueError("Haar input terlalu kecil")
    height = value.shape[-2] - value.shape[-2] % 2
    width = value.shape[-1] - value.shape[-1] % 2
    bands = F.conv2d(value[..., :height, :width], _haar_kernels(value), stride=2)
    return bands[:, :1], bands[:, 1:2], bands[:, 2:3], bands[:, 3:4]


def haar_decompose(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return orthonormal Haar LL and rotation-neutral detail energy."""

    ll, lh, hl, hh = haar_bands(value)
    detail = torch.sqrt(
        torch.cat((lh, hl, hh), dim=1).square().mean(dim=1, keepdim=True) + 1e-8
    )
    return ll, detail


def _standardize(value: torch.Tensor, clip: float) -> torch.Tensor:
    mean = value.mean(dim=(-2, -1), keepdim=True)
    scale = value.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
    return ((value - mean) / scale).clamp(-clip, clip) / clip


def chromatic_wavelet_cue(
    image: torch.Tensor,
    *,
    clip: float = 4.0,
    detail_mode: str = "energy",
) -> torch.Tensor:
    """Build illumination-reduced chroma plus two-level luminance wavelet cues.

    Energy mode preserves the original CWCF1 representation:
    [Cb, Cr, D1, D2], where each D collapses LH/HL/HH into
    rotation-neutral detail energy.

    Directional mode preserves all signed detail sub-bands:
    [Cb, Cr, LH1, HL1, HH1, LH2, HL2, HH2].

    Hybrid mode keeps the original energy channels and adds the six signed
    detail sub-bands:
    [Cb, Cr, D1, D2, LH1, HL1, HH1, LH2, HL2, HH2].

    RGB remains untouched for the native detector; cues are consumed only by
    the classification branches.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("CWCF input harus [B,3,H,W]")
    if not image.is_floating_point():
        raise TypeError("CWCF input harus floating point")
    if detail_mode not in {"energy", "directional", "hybrid"}:
        raise ValueError(
            "detail_mode harus 'energy', 'directional', atau 'hybrid'"
        )

    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    cb = blue - luminance
    cr = red - luminance
    ll1, lh1, hl1, hh1 = haar_bands(luminance)
    _, lh2, hl2, hh2 = haar_bands(ll1)
    size = image.shape[-2:]

    detail1 = torch.sqrt(
        torch.cat((lh1, hl1, hh1), dim=1).square().mean(dim=1, keepdim=True)
        + 1e-8
    )
    detail2 = torch.sqrt(
        torch.cat((lh2, hl2, hh2), dim=1).square().mean(dim=1, keepdim=True)
        + 1e-8
    )
    detail1 = F.interpolate(
        detail1, size=size, mode="bilinear", align_corners=False
    )
    detail2 = F.interpolate(
        detail2, size=size, mode="bilinear", align_corners=False
    )
    directional = [
        F.interpolate(band, size=size, mode="bilinear", align_corners=False)
        for band in (lh1, hl1, hh1, lh2, hl2, hh2)
    ]

    if detail_mode == "energy":
        cue = torch.cat((cb, cr, detail1, detail2), dim=1)
    elif detail_mode == "directional":
        cue = torch.cat((cb, cr, *directional), dim=1)
    else:
        cue = torch.cat((cb, cr, detail1, detail2, *directional), dim=1)

    return _standardize(cue, float(clip))
