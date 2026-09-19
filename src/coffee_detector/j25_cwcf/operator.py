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


def haar_decompose(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return orthonormal Haar LL and rotation-neutral detail energy."""

    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("Haar input harus [B,1,H,W]")
    if value.shape[-2] < 2 or value.shape[-1] < 2:
        raise ValueError("Haar input terlalu kecil")
    height = value.shape[-2] - value.shape[-2] % 2
    width = value.shape[-1] - value.shape[-1] % 2
    bands = F.conv2d(value[..., :height, :width], _haar_kernels(value), stride=2)
    ll = bands[:, :1]
    detail = torch.sqrt(bands[:, 1:].square().mean(dim=1, keepdim=True) + 1e-8)
    return ll, detail


def _standardize(value: torch.Tensor, clip: float) -> torch.Tensor:
    mean = value.mean(dim=(-2, -1), keepdim=True)
    scale = value.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
    return ((value - mean) / scale).clamp(-clip, clip) / clip


def chromatic_wavelet_cue(image: torch.Tensor, *, clip: float = 4.0) -> torch.Tensor:
    """Build illumination-reduced chroma plus two-level luminance-detail cues.

    The operator is fixed and parameter free. It deliberately does not replace
    RGB: the native detector still sees the original image and this cue is
    consumed only by the classification branches.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("CWCF input harus [B,3,H,W]")
    if not image.is_floating_point():
        raise TypeError("CWCF input harus floating point")
    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    cb = blue - luminance
    cr = red - luminance
    ll1, detail1 = haar_decompose(luminance)
    _, detail2 = haar_decompose(ll1)
    size = image.shape[-2:]
    detail1 = F.interpolate(detail1, size=size, mode="bilinear", align_corners=False)
    detail2 = F.interpolate(detail2, size=size, mode="bilinear", align_corners=False)
    cue = torch.cat((cb, cr, detail1, detail2), dim=1)
    return _standardize(cue, float(clip))
