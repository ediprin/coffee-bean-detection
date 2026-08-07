from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class GDSClsConfig:
    grid_size: int = 7
    threshold: float = 0.05
    auxiliary_weight: float = 0.25

    @classmethod
    def from_mapping(cls, payload: "GDSClsConfig | dict[str, Any] | None") -> "GDSClsConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.grid_size <= 0:
            raise ValueError("grid_size harus positif")
        if result.threshold <= 0 or result.auxiliary_weight < 0:
            raise ValueError("threshold/auxiliary_weight GDS tidak valid")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GDSClsDetectHead(nn.Module):
    """Serializable training-only GDS classification-aux wrapper."""

    def __init__(self, base_head: nn.Module, config: GDSClsConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("GDSC1 memerlukan native YOLO26 end-to-end Detect")
        self.base_head = base_head
        self.config = config
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

    def forward(self, features: list[torch.Tensor]):
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))
        return self.base_head(features)

    def fuse(self) -> None:
        self.base_head.fuse()


def load_gds_cls_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, GDSClsDetectHead):
        raise TypeError("Target bukan GDSClsDetectHead")
    if isinstance(source_head, GDSClsDetectHead):
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


class GDSClsDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, gds_cls=None):
        self.gds_cls_config = GDSClsConfig.from_mapping(gds_cls)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = GDSClsDetectHead(self.model[-1], self.gds_cls_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss
        from .loss import GDSAuxDetectionLoss
        return E2ELoss(self, loss_fn=GDSAuxDetectionLoss)
