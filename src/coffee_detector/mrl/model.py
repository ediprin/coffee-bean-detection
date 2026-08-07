from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class MRLConfig:
    """SFRNet Multi-RoI Loss transfer configuration.

    Source-paper facts preserved here:
    - RoI feature size 7x7;
    - foreground/background split at the RoI level;
    - anchor/positive/negative triads;
    - grouped Euclidean distance from center to outer square rings;
    - softplus(d_pos - d_neg) objective.

    Transfer-specific choices are explicit: YOLO one-to-many proposals are used,
    axis-aligned IoU=0.5 defines foreground/background (matching the source
    Oriented-RCNN train config threshold), and P3 is the RoI feature source.
    """

    roi_size: int = 7
    feature_level: int = 0
    training_topk: int = 256
    foreground_iou: float = 0.50
    loss_weight: float = 0.25
    box_expand: float = 1.0

    @classmethod
    def from_mapping(cls, payload: "MRLConfig | dict[str, Any] | None") -> "MRLConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.roi_size <= 0 or result.roi_size % 2 == 0:
            raise ValueError("roi_size MRL harus ganjil dan positif")
        if result.feature_level not in {0, 1, 2}:
            raise ValueError("feature_level MRL harus 0, 1, atau 2")
        if result.training_topk <= 0:
            raise ValueError("training_topk MRL harus positif")
        if not 0.0 < result.foreground_iou <= 1.0:
            raise ValueError("foreground_iou harus berada pada (0,1]")
        if result.loss_weight < 0:
            raise ValueError("loss_weight MRL tidak boleh negatif")
        if result.box_expand < 1.0:
            raise ValueError("box_expand MRL minimal 1.0")
        return result

    @property
    def groups(self) -> int:
        return (self.roi_size + 1) // 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MRLDetectHead(nn.Module):
    """Native YOLO Detect plus a serializable training-only MRL configuration."""

    def __init__(self, base_head: nn.Module, config: MRLConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("MRL memerlukan native Ultralytics Detect")
        if not getattr(base_head, "end2end", False):
            raise ValueError("MRL screening dikunci untuk YOLO26 end-to-end")
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

    def _sync_runtime_attributes(self) -> None:
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        return self.base_head(features)

    def fuse(self) -> None:
        self.base_head.fuse()


def inject_mrl(model: nn.Module, config: MRLConfig | dict[str, Any] | None) -> int:
    frozen = MRLConfig.from_mapping(config)
    detector = getattr(model, "model", model)
    if not isinstance(detector, (nn.Sequential, nn.ModuleList)) or not len(detector):
        raise TypeError("Model tidak memiliki daftar layer Ultralytics")
    if isinstance(detector[-1], MRLDetectHead):
        return 0
    detector[-1] = MRLDetectHead(detector[-1], frozen)
    return 1


def load_mrl_detector_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly map a native D0 Detect state into the MRL wrapper."""

    model.load(weights)
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head = source_model[-1]
    target_head = target[-1]
    if not isinstance(target_head, MRLDetectHead):
        raise TypeError("Target bukan MRLDetectHead")
    if isinstance(source_head, MRLDetectHead):
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
    DetectionModel = nn.Module  # type: ignore[assignment,misc]


class MRLDetectionModel(DetectionModel):
    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, mrl=None):
        self.mrl_config = MRLConfig.from_mapping(mrl)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        inject_mrl(self, self.mrl_config)

    def init_criterion(self):
        from .loss import MRLDetectionLoss
        return MRLDetectionLoss(self)
