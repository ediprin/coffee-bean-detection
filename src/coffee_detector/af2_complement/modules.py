from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def low_high_split(value: torch.Tensor, kernel_size: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """A shape-preserving local low/high decomposition used by both candidates."""

    if value.ndim != 4:
        raise ValueError("feature harus BCHW")
    low = F.avg_pool2d(
        value,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    return low, value - low


class FrequencySelectionResidual(nn.Module):
    """Light spatially-varying low/high selector with an identity initial state.

    This is a conservative YOLO transfer inspired by FreqSelect. It is not a
    reproduction of the full FADC operator: dilation and native convolutions
    remain unchanged. The zero-initialized output makes the initial detector
    exactly equal to its AF2 parent.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.selector = nn.Conv2d(self.channels, 2, 1, bias=True)
        self.output = nn.Conv2d(
            self.channels, self.channels, 1, groups=self.channels, bias=False
        )
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)
        nn.init.zeros_(self.output.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.channels:
            raise ValueError("shape feature tidak sesuai FrequencySelectionResidual")
        low, high = low_high_split(value, self.kernel_size)
        weights = torch.softmax(self.selector(value), dim=1)
        selected = weights[:, :1] * low + weights[:, 1:] * high
        return value + self.output(selected)


class SpaceFrequencySelectionResidual(nn.Module):
    """Shared spatial/frequency feature selector with identity initialization.

    The spatial path is a learnable local depthwise operator and the frequency
    path is a fixed local high-pass residual. A learned spatial selector fuses
    them, then a zero-initialized depthwise projection adds the result to P3.
    Both native box and class branches consume the same adapted feature.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.spatial = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=self.channels,
            bias=False,
        )
        self.selector = nn.Conv2d(self.channels, 2, 1, bias=True)
        self.output = nn.Conv2d(
            self.channels, self.channels, 1, groups=self.channels, bias=False
        )
        nn.init.dirac_(self.spatial.weight, groups=self.channels)
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)
        nn.init.zeros_(self.output.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.channels:
            raise ValueError("shape feature tidak sesuai SpaceFrequencySelectionResidual")
        _low, high = low_high_split(value, self.kernel_size)
        spatial = self.spatial(value)
        weights = torch.softmax(self.selector(value), dim=1)
        selected = weights[:, :1] * spatial + weights[:, 1:] * high
        return value + self.output(selected)
