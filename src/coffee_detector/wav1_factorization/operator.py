from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.af2_spectral.config import frozen_arm_config as frozen_spectral_arm_config
from coffee_detector.af2_spectral.operator import (
    SpectralInputEnhancer,
    haar_dwt2,
    rgb_luminance,
)
from coffee_detector.afab.operator import afab_gate

from .config import WAV1FactorizationConfig


# Numerical guard only for the NEW causal controls. WAV1_REF must remain bitwise
# delegated to the already-confirmed implementation. The floor is derived from
# float32 machine precision, not tuned on validation performance.
_FLOAT32_FLAT_TOL = 64.0 * torch.finfo(torch.float32).eps


def stable_minmax_spatial(value: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Min-max normalize while mapping numerically-flat maps exactly to zero.

    Plain min-max can amplify CPU/GPU convolution roundoff in an otherwise
    constant cue into a full [0,1] spatial gate. For the new controls we first
    compute the normalization in float32 and treat a spatial range within
    64 float32 eps (relative to max(|x|, 1)) as numerical flatness. This is a
    numerical contract, not a validation-tuned experimental hyperparameter.
    """

    if not torch.is_floating_point(value):
        raise TypeError("normalisasi membutuhkan tensor floating point")
    dtype = value.dtype
    work = value.float()
    low = work.amin(dim=(-2, -1), keepdim=True)
    high = work.amax(dim=(-2, -1), keepdim=True)
    spread = high - low
    scale = work.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    threshold = scale * max(float(eps), _FLOAT32_FLAT_TOL)
    normalized = (work - low) / spread.clamp_min(threshold)
    normalized = torch.where(spread > threshold, normalized, torch.zeros_like(normalized))
    return normalized.to(dtype=dtype)


def wavelet_detail_levels(
    value: torch.Tensor, eps: float = 1.0e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw resized Haar detail-energy maps for WAV1 levels 1 and 2.

    The calculation intentionally matches the frozen WAV1 operator before its
    per-level min-max normalization: luminance -> Haar DWT -> L2 magnitude of
    LH/HL/HH -> bilinear resize to the input resolution.
    """

    luminance = rgb_luminance(value)
    target = luminance.shape[-2:]
    current = luminance
    details: list[torch.Tensor] = []
    for _ in range(2):
        bands, _ = haar_dwt2(current)
        current = bands[:, :, 0]
        detail = bands[:, :, 1:].square().sum(2).add(float(eps)).sqrt()
        detail = F.interpolate(detail, size=target, mode="bilinear", align_corners=False)
        details.append(detail)
    return details[0], details[1]


def fixed_local_highpass(value: torch.Tensor) -> torch.Tensor:
    """Parameter-free local-detail control using a fixed 3x3 binomial blur.

    This is a generic high-pass control, not a learned filter and not claimed
    to reproduce any specific published LoG implementation. The 3x3 kernel is
    fixed to [1,2,1]^T[1,2,1]/16 so the study has no validation-tuned sigma.
    """

    luminance = rgb_luminance(value)
    kernel = luminance.new_tensor(
        ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))
    ).view(1, 1, 3, 3) / 16.0
    padded = F.pad(luminance, (1, 1, 1, 1), mode="replicate")
    smooth = F.conv2d(padded, kernel)
    return (luminance - smooth).abs()


class WAV1FactorizationEnhancer(nn.Module):
    """Parameter-free causal controls around the already-confirmed WAV1 cue."""

    def __init__(self, config: WAV1FactorizationConfig | dict | None = None) -> None:
        super().__init__()
        self.config = WAV1FactorizationConfig.from_mapping(config)
        self.reference = None
        if self.config.arm == "WAV1_REF":
            # Delegate instead of reimplementing so WAV1_REF remains the exact
            # frozen implementation used by the completed confirmation study.
            self.reference = SpectralInputEnhancer(frozen_spectral_arm_config("WAV1"))

    def recover(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("frontend membutuhkan BCHW RGB")
        if not torch.is_floating_point(value):
            raise TypeError("frontend membutuhkan tensor floating point")
        if self.config.arm == "WAV1_REF":
            assert self.reference is not None
            return self.reference.recover(value)
        if self.config.arm == "HP1":
            cue = stable_minmax_spatial(fixed_local_highpass(value), self.config.eps)
        else:
            detail1, detail2 = wavelet_detail_levels(value, self.config.eps)
            if self.config.arm == "WAV_L1":
                cue = stable_minmax_spatial(detail1, self.config.eps)
            elif self.config.arm == "WAV_L2":
                cue = stable_minmax_spatial(detail2, self.config.eps)
            elif self.config.arm == "WAV_RAWFUSE":
                # Deliberately removes WAV1's per-level equalization. The only
                # normalization happens after the raw scale evidence is fused.
                cue = stable_minmax_spatial(detail1 + detail2, self.config.eps)
            else:  # defensive: config validation should make this unreachable
                raise RuntimeError(f"arm tidak ditangani: {self.config.arm}")
        return cue.expand_as(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.config.arm == "WAV1_REF":
            assert self.reference is not None
            return self.reference(value)
        return afab_gate(value, self.recover(value), eps=self.config.eps)
