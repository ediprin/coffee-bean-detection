from __future__ import annotations

import torch
from torch import nn

from coffee_detector.afab import AFABConfig, AFABInputEnhancer, minmax_spatial


class AF2ChannelCalibratedEnhancer(nn.Module):
    """AF2 with three learned, input-independent RGB residual scales."""

    def __init__(self, afab: AFABConfig | dict | None = None) -> None:
        super().__init__()
        self.afab_config = AFABConfig.from_mapping(afab)
        if self.afab_config.mode != "af2":
            raise ValueError("AF2-CAL3 hanya kompatibel dengan AFAB mode af2")
        self.af2 = AFABInputEnhancer(self.afab_config)
        self.calibration_logits = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def scale(self) -> torch.Tensor:
        return 1.0 + torch.tanh(self.calibration_logits)

    def forward_with_scale(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recovered = self.af2.recover(raw)
        normalized = minmax_spatial(recovered, eps=self.afab_config.eps)
        residual = raw * normalized
        scale = self.scale()
        return raw + scale * residual, scale

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_scale(raw)
        return output

