from __future__ import annotations

import torch
from torch import nn

from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer, minmax_spatial


def rec709_luminance(value: torch.Tensor) -> torch.Tensor:
    """Return linear Rec.709 luminance as a single BCHW channel."""
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"Rec.709 membutuhkan BCHW RGB, diterima {tuple(value.shape)}")
    weights = value.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return (value * weights).sum(dim=1, keepdim=True)


class AF2LuminanceInputEnhancer(nn.Module):
    """Legacy AF2 whose spectral cue is computed once from luminance.

    The legacy AF2 transfer computes an independent recovered signal and gate
    for every RGB channel.  This diagnostic variant changes only that transfer
    choice: AF2 is evaluated on Rec.709 luminance and the resulting one-channel
    gate is shared by the untouched RGB channels.  The detector therefore still
    receives RGB and the same raw-preserving residual ``x + x * gate``.
    """

    def __init__(self, config: AFABConfig | dict | None = None) -> None:
        super().__init__()
        frozen = AFABConfig.from_mapping(config)
        if frozen.mode != "af2":
            raise ValueError("AF2 luminance dikunci untuk mode af2")
        self.config = frozen
        self.af2 = AFABInputEnhancer(frozen)

    def recover_luminance(self, value: torch.Tensor) -> torch.Tensor:
        luminance = rec709_luminance(value)
        # _recover_one is channel-agnostic internally; the public legacy
        # recover() deliberately enforces RGB to protect historical behavior.
        return torch.cat(
            [self.af2._recover_one(luminance[i : i + 1]) for i in range(value.shape[0])],
            dim=0,
        )

    def shared_gate(self, value: torch.Tensor) -> torch.Tensor:
        return minmax_spatial(self.recover_luminance(value), eps=self.config.eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(value):
            raise TypeError("AF2 luminance memerlukan tensor floating point")
        gate = self.shared_gate(value)
        return value + value * gate
