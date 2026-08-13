from __future__ import annotations

from typing import Any

import torch
from torch import nn

from coffee_detector.stb.model import STBConfig, STBDetectHead


class TokenChannelMixerBlock(nn.Module):
    """Parameter-matched non-spatial control for one Swin block.

    Four C->C linear layers match the QKV plus output projection parameter
    order of multi-head attention. The second residual MLP and both LayerNorms
    match the Swin block. Every operation is pointwise in H/W, so the control
    adds channel capacity without spatial token interaction.
    """

    def __init__(self, channels: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.norm1 = nn.LayerNorm(channels)
        self.channel_mixer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.channel_mixer(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class ClassificationChannelControl(nn.Module):
    """Two token-wise blocks with the same identity-start gate as STB1."""

    def __init__(self, channels: int, config: STBConfig) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            TokenChannelMixerBlock(channels, config.mlp_ratio),
            TokenChannelMixerBlock(channels, config.mlp_ratio),
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        nhwc = value.permute(0, 2, 3, 1).contiguous()
        transformed = self.blocks(nhwc).permute(0, 3, 1, 2).contiguous()
        return value + self.gate * (transformed - value)


class STBCapacityControlDetectHead(STBDetectHead):
    """Native YOLO box path plus non-spatial capacity-matched cls blocks."""

    def __init__(self, base_head: nn.Module, config: STBConfig) -> None:
        super().__init__(base_head, config)
        channels = []
        for branch in base_head.cv2:
            for child in branch.modules():
                if isinstance(child, nn.Conv2d):
                    channels.append(int(child.in_channels))
                    break
        self.blocks = nn.ModuleList(
            [ClassificationChannelControl(channel, config) for channel in channels]
        )


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class STBCapacityControlDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, stb=None):
        self.stb_config = STBConfig.from_mapping(stb)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = STBCapacityControlDetectHead(
            self.model[-1], self.stb_config
        )


def load_stb_control_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, STBCapacityControlDetectHead):
        raise TypeError("Target bukan STBCapacityControlDetectHead")
    if isinstance(source_head, STBCapacityControlDetectHead):
        target_head.load_state_dict(source_head.state_dict(), strict=True)
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Control harus dimulai dari native D0, bukan {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect tidak lengkap")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}
