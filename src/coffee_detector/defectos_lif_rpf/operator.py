from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LIFRPFConfig


def _luminance(image: torch.Tensor) -> torch.Tensor:
    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _standardize(value: torch.Tensor, clip: float) -> torch.Tensor:
    mean = value.mean(dim=(-2, -1), keepdim=True)
    scale = value.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
    return ((value - mean) / scale).clamp(-clip, clip) / clip


def luminance_illumination_preprocess(
    image: torch.Tensor,
    config: LIFRPFConfig | dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten local illumination while preserving raw RGB differences.

    A large local mean estimates illumination.  Only the luminance residual is
    corrected, with a hard bound on its magnitude.  The same scalar correction
    is added to R/G/B, so no channel is independently filtered.  The detector
    still consumes the unmodified RGB image; the returned preprocessed image is
    used to construct an explicit auxiliary cue.
    """

    cfg = LIFRPFConfig.from_mapping(config)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("LIF-RPF input harus [B,3,H,W]")
    if not image.is_floating_point():
        raise TypeError("LIF-RPF input harus floating point")
    original_dtype = image.dtype
    work = image.float()
    luminance = _luminance(work)
    kernel = min(cfg.illumination_kernel, luminance.shape[-2], luminance.shape[-1])
    if kernel % 2 == 0:
        kernel -= 1
    if kernel < 3:
        raise ValueError("Resolusi input terlalu kecil untuk estimasi illumination")
    illumination = F.avg_pool2d(
        luminance, kernel_size=kernel, stride=1, padding=kernel // 2,
        count_include_pad=False,
    )
    global_luminance = luminance.mean(dim=(-2, -1), keepdim=True)
    ratio = ((global_luminance + 1e-3) / (illumination + 1e-3)).clamp(0.80, 1.25)
    proposed = luminance * ratio
    delta = (proposed - luminance).clamp(
        -cfg.maximum_luminance_delta, cfg.maximum_luminance_delta
    )
    # Do not modify already-uniform regions.  The gate is fixed and depends
    # only on deviation from each image's own global illumination.
    need = ((illumination - global_luminance).abs() / 0.25).clamp(0.0, 1.0)
    delta = delta * need
    preprocessed = (work + delta).clamp(0.0, 1.0)
    log_reflectance = torch.log(luminance + 1e-3) - torch.log(illumination + 1e-3)
    return (
        preprocessed.to(original_dtype),
        delta.to(original_dtype),
        log_reflectance.to(original_dtype),
    )


def luminance_residual_cue(
    image: torch.Tensor,
    config: LIFRPFConfig | dict | None = None,
) -> torch.Tensor:
    """Return bounded preprocessing and reflectance cues at input resolution."""

    cfg = LIFRPFConfig.from_mapping(config)
    _, delta, reflectance = luminance_illumination_preprocess(image, cfg)
    normalized_delta = (delta / cfg.maximum_luminance_delta).clamp(-1.0, 1.0)
    normalized_reflectance = _standardize(reflectance.float(), cfg.cue_clip).to(
        image.dtype
    )
    cue = torch.cat((normalized_delta, normalized_reflectance), dim=1)
    if not torch.isfinite(cue).all():
        raise RuntimeError("Cue LIF-RPF tidak finite")
    return cue
