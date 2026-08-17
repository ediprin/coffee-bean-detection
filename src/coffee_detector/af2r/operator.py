from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.afab import AFABConfig, AFABInputEnhancer, minmax_spatial

from .config import AF2RConfig


def illumination_features(
    raw: torch.Tensor,
    recovered_normalized: torch.Tensor,
    *,
    local_kernel: int = 15,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return six bounded cues without changing spatial geometry.

    The cues expose absolute/local luminance, local contrast, relative
    illumination, red-blue temperature, and AF2 recovery strength. They are
    intentionally low-level: no labels, decoded boxes, proposals, or ROI
    operations are used.
    """

    if raw.ndim != 4 or raw.shape[1] != 3:
        raise ValueError("AF2R memerlukan tensor RGB BCHW")
    if recovered_normalized.shape != raw.shape:
        raise ValueError("Recovered AF2 harus memiliki bentuk yang sama dengan input")
    padding = local_kernel // 2
    luminance = (
        0.299 * raw[:, 0:1] + 0.587 * raw[:, 1:2] + 0.114 * raw[:, 2:3]
    )
    local_mean = F.avg_pool2d(
        luminance, kernel_size=local_kernel, stride=1, padding=padding
    )
    local_variance = F.avg_pool2d(
        (luminance - local_mean).square(),
        kernel_size=local_kernel,
        stride=1,
        padding=padding,
    )
    local_contrast = torch.sqrt(local_variance.clamp_min(0.0) + eps)
    relative_illumination = (luminance / local_mean.clamp_min(eps)).clamp(0.0, 2.0) * 0.5
    red_blue = ((raw[:, 0:1] - raw[:, 2:3]) + 1.0) * 0.5
    recovery_strength = recovered_normalized.mean(dim=1, keepdim=True)
    return torch.cat(
        (
            luminance,
            local_mean,
            local_contrast,
            relative_illumination,
            red_blue,
            recovery_strength,
        ),
        dim=1,
    )


class AF2ResidualGateEnhancer(nn.Module):
    """AFAB-2 with a learnable, raw-preserving residual reliability gate.

    The fixed AFAB-2 residual is ``raw * normalize(recovered)``. The learned
    gate is ``1 + tanh(delta)`` and therefore starts at exactly one. At zero
    initialization this module reproduces ordinary AF2; optimization may
    suppress the residual toward zero or amplify it up to two. The AF2R0
    control owns the same parameters but receives zero conditioning features.
    """

    feature_channels = 6

    def __init__(
        self,
        afab: AFABConfig | dict | None = None,
        config: AF2RConfig | dict | None = None,
    ) -> None:
        super().__init__()
        self.afab_config = AFABConfig.from_mapping(afab)
        if self.afab_config.mode != "af2":
            raise ValueError("AF2R hanya kompatibel dengan AFAB mode af2")
        self.config = AF2RConfig.from_mapping(config)
        self.af2 = AFABInputEnhancer(self.afab_config)
        hidden = self.config.hidden_channels
        self.gate = nn.Sequential(
            nn.Conv2d(self.feature_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def conditioning(self, raw: torch.Tensor, recovered_normalized: torch.Tensor) -> torch.Tensor:
        features = illumination_features(
            raw,
            recovered_normalized,
            local_kernel=self.config.local_kernel,
            eps=self.config.eps,
        )
        if self.config.conditioning == "zero":
            return torch.zeros_like(features)
        return features

    def forward_with_gate(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recovered = self.af2.recover(raw)
        normalized = minmax_spatial(recovered, eps=self.afab_config.eps)
        residual = raw * normalized
        delta = torch.tanh(self.gate(self.conditioning(raw, normalized)))
        reliability = 1.0 + delta
        return raw + reliability * residual, reliability

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_gate(raw)
        return output
