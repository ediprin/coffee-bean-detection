from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class FocalModulationConfig:
    """FocalNet defaults used in the NeurIPS 2022 reference implementation."""

    focal_window: int = 3
    focal_level: int = 2
    focal_factor: int = 2
    depth: int = 2
    mlp_ratio: float = 4.0
    normalize_modulator: bool = False

    @classmethod
    def from_mapping(
        cls, payload: "FocalModulationConfig | dict[str, Any] | None"
    ) -> "FocalModulationConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if min(result.focal_window, result.focal_level, result.focal_factor, result.depth) <= 0:
            raise ValueError("parameter focal modulation harus positif")
        if result.focal_window % 2 == 0 or result.mlp_ratio <= 0:
            raise ValueError("focal_window harus ganjil dan mlp_ratio harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FocalModulation(nn.Module):
    """Faithful PyTorch port of microsoft/FocalNet's focal modulation operator.

    Input and output are NHWC, matching the official implementation. Context is
    aggregated by nested depth-wise convolutions plus a gated global context;
    query and modulator are then multiplied elementwise.
    """

    def __init__(self, channels: int, config: FocalModulationConfig) -> None:
        super().__init__()
        self.channels = int(channels)
        self.focal_level = config.focal_level
        self.normalize_modulator = config.normalize_modulator
        self.f = nn.Linear(channels, 2 * channels + config.focal_level + 1)
        self.h = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.proj = nn.Linear(channels, channels)
        self.act = nn.GELU()
        self.focal_layers = nn.ModuleList()
        for level in range(config.focal_level):
            kernel = config.focal_factor * level + config.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        stride=1,
                        groups=channels,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.f(value).permute(0, 3, 1, 2).contiguous()
        query, context, gates = torch.split(
            projected, (self.channels, self.channels, self.focal_level + 1), dim=1
        )
        aggregated = torch.zeros_like(context)
        current = context
        for level, layer in enumerate(self.focal_layers):
            current = layer(current)
            aggregated = aggregated + current * gates[:, level : level + 1]
        global_context = self.act(context.mean(dim=(2, 3), keepdim=True))
        aggregated = aggregated + global_context * gates[:, self.focal_level :]
        if self.normalize_modulator:
            aggregated = aggregated / float(self.focal_level + 1)
        modulator = self.h(aggregated)
        output = (query * modulator).permute(0, 2, 3, 1).contiguous()
        return self.proj(output)


class FocalModulationBlock(nn.Module):
    """Default pre-LN FocalNet block without stochastic depth."""

    def __init__(self, channels: int, config: FocalModulationConfig) -> None:
        super().__init__()
        hidden = int(channels * config.mlp_ratio)
        self.norm1 = nn.LayerNorm(channels)
        self.modulation = FocalModulation(channels, config)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.modulation(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class ClassificationFocalModulation(nn.Module):
    """Two FocalNet blocks with an exact identity-start residual gate."""

    def __init__(self, channels: int, config: FocalModulationConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *(FocalModulationBlock(channels, config) for _ in range(config.depth))
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        nhwc = value.permute(0, 2, 3, 1).contiguous()
        transformed = self.layers(nhwc).permute(0, 3, 1, 2).contiguous()
        return value + self.gate * (transformed - value)


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class FocalModulationDetectHead(nn.Module):
    """Focal modulation on P3/P4/P5 classification only; boxes stay native."""

    def __init__(self, base_head: nn.Module, config: FocalModulationConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("FMH1 memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("FMH1 memerlukan P3/P4/P5")
        self.base_head = base_head
        self.config = config
        self.blocks = nn.ModuleList(
            [ClassificationFocalModulation(channel, config) for channel in channels]
        )
        for name in ("i", "f", "type", "np"):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))
        for name in (
            "nc", "nl", "reg_max", "stride", "end2end", "max_det", "export",
            "format", "dynamic", "agnostic_nms",
        ):
            if hasattr(base_head, name):
                setattr(self, name, getattr(base_head, name))

    @property
    def one2many(self):
        return self.base_head.one2many

    @property
    def one2one(self):
        return self.base_head.one2one

    def _sync_runtime_attributes(self) -> None:
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    def _forward_branch(self, features: list[torch.Tensor], branch: dict[str, nn.Module]):
        enhanced = [block(feature) for block, feature in zip(self.blocks, features)]
        boxes, logits = [], []
        for index in range(self.nl):
            boxes.append(branch["box_head"][index](features[index]))
            logits.append(branch["cls_head"][index](enhanced[index]))
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat([x.view(batch, 4 * self.reg_max, -1) for x in boxes], -1),
            "scores": torch.cat([x.view(batch, self.nc, -1) for x in logits], -1),
            "feats": features,
        }

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many),
                "one2one": self._forward_branch([x.detach() for x in features], self.one2one),
            }
        one2many = self._forward_branch(features, self.one2many) if self._has_heads(self.one2many) else None
        one2one = self._forward_branch([x.detach() for x in features], self.one2one)
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_focal_modulation_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, FocalModulationDetectHead):
        raise TypeError("Target bukan FocalModulationDetectHead")
    if isinstance(source_head, FocalModulationDetectHead):
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
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


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class FocalModulationDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, focal_modulation=None):
        self.focal_modulation_config = FocalModulationConfig.from_mapping(focal_modulation)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = FocalModulationDetectHead(
            self.model[-1], self.focal_modulation_config
        )
