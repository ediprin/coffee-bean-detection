from __future__ import annotations

import torch
from torch import nn

from coffee_detector.af2_complement.modules import SpaceFrequencySelectionResidual


class MultilevelTrainingScaffold(nn.Module):
    """P3/P4/P5 scaffold that is structurally inactive outside training."""

    def __init__(self, channels: tuple[int, ...], kernel_size: int = 3) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("MultilevelTrainingScaffold memerlukan P3/P4/P5")
        self.channels = tuple(int(value) for value in channels)
        self.adapters = nn.ModuleList(
            SpaceFrequencySelectionResidual(channel, kernel_size)
            for channel in self.channels
        )
        self.register_buffer("strength", torch.tensor(1.0), persistent=False)

    def set_strength(self, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("strength harus dalam [0,1]")
        self.strength.fill_(float(value))

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.adapters):
            raise ValueError("Jumlah feature pyramid tidak sesuai")
        if not self.training or float(self.strength) == 0.0:
            return list(features)
        scale = self.strength.to(device=features[0].device, dtype=features[0].dtype)
        return [
            value + scale * (adapter(value) - value)
            for value, adapter in zip(features, self.adapters)
        ]
