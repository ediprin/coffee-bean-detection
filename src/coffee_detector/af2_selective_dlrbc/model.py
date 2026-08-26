from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig
from coffee_detector.dlrbc.model import (
    DLRBCConfig,
    DLRBCDetectHead,
    LowRankClassResidual,
    _shape_compatible_state,
)


@dataclass(frozen=True)
class AF2SelectiveDLRBCConfig:
    selected_class_ids: tuple[int, ...]
    rank: int = 8
    projection_ratio: float = 0.5
    minimum_projection: int = 16
    residual_scale: float = 0.1
    signed_sqrt: bool = True
    eps: float = 1.0e-6

    @classmethod
    def from_mapping(
        cls, payload: "AF2SelectiveDLRBCConfig | Mapping[str, Any]"
    ) -> "AF2SelectiveDLRBCConfig":
        if isinstance(payload, cls):
            result = payload
        else:
            values = dict(payload)
            values["selected_class_ids"] = tuple(
                int(value) for value in values.get("selected_class_ids", ())
            )
            result = cls(**values)
        selected = tuple(sorted(set(result.selected_class_ids)))
        if selected != result.selected_class_ids:
            result = cls(**{**asdict(result), "selected_class_ids": selected})
        if not selected:
            raise ValueError("Sedikitnya satu kelas harus dipilih")
        if selected[0] < 0:
            raise ValueError("Class ID tidak boleh negatif")
        result.dlrbc_config()
        return result

    def dlrbc_config(self) -> DLRBCConfig:
        return DLRBCConfig.from_mapping(
            {
                "mode": "quadratic",
                "rank": self.rank,
                "projection_ratio": self.projection_ratio,
                "minimum_projection": self.minimum_projection,
                "residual_scale": self.residual_scale,
                "signed_sqrt": self.signed_sqrt,
                "eps": self.eps,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_class_ids"] = list(self.selected_class_ids)
        return payload


class SelectiveLowRankResidual(nn.Module):
    """Zero-initialized, bounded class-selective quadratic residual."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        config: AF2SelectiveDLRBCConfig,
    ) -> None:
        super().__init__()
        if max(config.selected_class_ids) >= int(num_classes):
            raise ValueError("Class ID pilihan berada di luar jumlah kelas")
        self.residual = LowRankClassResidual(
            channels, num_classes, config.dlrbc_config()
        )
        mask = torch.zeros(int(num_classes), dtype=torch.float32)
        mask[list(config.selected_class_ids)] = 1.0
        self.register_buffer("class_mask", mask, persistent=True)
        self.gate = nn.Parameter(torch.zeros(int(num_classes)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.class_mask * torch.tanh(self.gate)
        return self.residual(value) * weight.view(1, -1, 1, 1)


class SelectiveDLRBCDetectHead(DLRBCDetectHead):
    def __init__(self, base_head: nn.Module, config: AF2SelectiveDLRBCConfig) -> None:
        super().__init__(base_head, config.dlrbc_config())
        self.selective_config = config
        channels = self.class_tower_channels
        self.one2many_residuals = nn.ModuleList(
            [SelectiveLowRankResidual(c, int(self.nc), config) for c in channels]
        )
        self.one2one_residuals = nn.ModuleList(
            [SelectiveLowRankResidual(c, int(self.nc), config) for c in channels]
        )


class AF2SelectiveDLRBCDetectionModel(AFABDetectionModel):
    def __init__(
        self,
        cfg="yolo26.yaml",
        ch=3,
        nc=None,
        verbose=True,
        *,
        afab: AFABConfig | Mapping[str, Any] | None = None,
        selective: AF2SelectiveDLRBCConfig | Mapping[str, Any],
    ) -> None:
        self.selective_config = AF2SelectiveDLRBCConfig.from_mapping(selective)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.model[-1] = SelectiveDLRBCDetectHead(
            self.model[-1], self.selective_config
        )


def _source_model(weights: Any) -> nn.Module:
    if isinstance(weights, dict):
        source = weights.get("ema")
        if not isinstance(source, nn.Module):
            source = weights.get("model")
    else:
        source = weights
    source = getattr(source, "ema", source)
    if not isinstance(source, nn.Module):
        raise TypeError("Checkpoint tidak mengekspos model torch")
    return source


def load_af2_selective_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load AF2 parent or resumable selective checkpoint."""

    source = _source_model(weights)
    target_layers = getattr(model, "model", None)
    source_layers = getattr(source, "model", None)
    if not isinstance(target_layers, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Target tidak mengekspos layer model")
    if not isinstance(source_layers, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Source tidak mengekspos layer model")
    target_head = target_layers[-1]
    if not isinstance(target_head, SelectiveDLRBCDetectHead):
        raise TypeError("Target head bukan SelectiveDLRBCDetectHead")
    source_head = source_layers[-1]
    if isinstance(source_head, SelectiveDLRBCDetectHead):
        model.load(weights)
        return {"resume": 1, "source_items": len(source.state_dict())}
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"AF2 parent harus memakai native Detect: {type(source_head).__name__}")

    model.load(weights)
    compatible = _shape_compatible_state(target_head.base_head, source_head)
    loaded = target_head.base_head.load_state_dict(compatible, strict=False)
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {
        "resume": 0,
        "source_items": len(source.state_dict()),
        "native_head_compatible_items": len(compatible),
        "native_head_missing_items": len(loaded.missing_keys),
    }


def selective_modules(model: nn.Module) -> Sequence[SelectiveLowRankResidual]:
    return tuple(
        module
        for module in model.modules()
        if isinstance(module, SelectiveLowRankResidual)
    )
