from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models.swin_transformer import ShiftedWindowAttention

from coffee_detector.stb.model import STBConfig, STBDetectHead
from coffee_detector.stb_control.model import ClassificationChannelControl


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class WindowAttentionResidualBlock(nn.Module):
    """Swin-style window attention residual without the block MLP.

    This intentionally retains only LN -> shifted-window attention -> residual.
    The CMC path already supplies channel mixing and MLP capacity, so STB-SR1
    does not duplicate the Swin MLP in the spatial correction branch.
    Input/output layout is NHWC.
    """

    def __init__(
        self,
        channels: int,
        *,
        window_size: int,
        shift_size: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels={channels} tidak habis dibagi heads={num_heads}")
        self.norm = nn.LayerNorm(channels)
        self.attn = ShiftedWindowAttention(
            dim=channels,
            window_size=[window_size, window_size],
            shift_size=[shift_size, shift_size],
            num_heads=num_heads,
            qkv_bias=True,
            proj_bias=True,
            attention_dropout=0.0,
            dropout=0.0,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.attn(self.norm(value))


class ClassificationCMCSpatialResidual(nn.Module):
    """CMC representation plus gated W-MSA/SW-MSA attention-only correction.

    Both the CMC identity gate and the spatial residual gate start at zero.
    Therefore the complete module is exactly identity at initialization and can
    be trained from the same native D0 checkpoint as CMC0/STB1.
    """

    def __init__(self, channels: int, config: STBConfig) -> None:
        super().__init__()
        self.channel_base = ClassificationChannelControl(channels, config)
        shift = config.window_size // 2
        self.wmsa = WindowAttentionResidualBlock(
            channels,
            window_size=config.window_size,
            shift_size=0,
            num_heads=config.num_heads,
        )
        self.swmsa = WindowAttentionResidualBlock(
            channels,
            window_size=config.window_size,
            shift_size=shift,
            num_heads=config.num_heads,
        )
        self.spatial_gate = nn.Parameter(torch.zeros(()))

    @property
    def channel_gate(self) -> nn.Parameter:
        return self.channel_base.gate

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.channel_base(value)
        nhwc = base.permute(0, 2, 3, 1).contiguous()
        spatial = self.swmsa(self.wmsa(nhwc)).permute(0, 3, 1, 2).contiguous()
        return base + self.spatial_gate * (spatial - base)


class STBSR1DetectHead(STBDetectHead):
    """Native YOLO26 box path with CMC + selective spatial cls refinement."""

    def __init__(self, base_head: nn.Module, config: STBConfig) -> None:
        super().__init__(base_head, config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("STB-SR1 memerlukan P3/P4/P5")
        self.blocks = nn.ModuleList(
            [ClassificationCMCSpatialResidual(channel, config) for channel in channels]
        )


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class STBSR1DetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, stb=None):
        self.stb_config = STBConfig.from_mapping(stb)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = STBSR1DetectHead(self.model[-1], self.stb_config)


def load_stb_sr1_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load native D0 for a fresh run, or exact STB-SR1 state for resume."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, STBSR1DetectHead):
        raise TypeError("Target bukan STBSR1DetectHead")
    if isinstance(source_head, STBSR1DetectHead):
        target_head.load_state_dict(source_head.state_dict(), strict=True)
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(
            f"STB-SR1 harus dimulai dari native D0 atau checkpoint STB-SR1, bukan {type(source_head).__name__}"
        )
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect ke STB-SR1 tidak lengkap")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}
