from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from coffee_detector.stb.model import STBDetectHead, STBDetectionModel


@dataclass(frozen=True)
class FCSTBConfig:
    """Frozen protocol settings for frequency-consistent STB distillation."""

    mode: str = "distill"  # control | distill
    temperature: float = 2.0
    distillation_weight: float = 0.50
    minimum_teacher_gt_probability: float = 0.10
    teacher_checkpoint: str | None = None

    @classmethod
    def from_mapping(
        cls, payload: "FCSTBConfig | Mapping[str, Any] | None"
    ) -> "FCSTBConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.mode not in {"control", "distill"}:
            raise ValueError("mode FC-STB harus control atau distill")
        if result.temperature <= 0 or result.distillation_weight < 0:
            raise ValueError("temperature harus positif dan bobot distillation non-negatif")
        if not 0 <= result.minimum_teacher_gt_probability <= 1:
            raise ValueError("minimum_teacher_gt_probability harus di [0,1]")
        if result.mode == "distill" and not result.teacher_checkpoint:
            raise ValueError("mode distill memerlukan teacher_checkpoint AF2")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_fcstb_weights(model: nn.Module, weights: Any) -> dict[str, Any]:
    """Strictly transfer a completed STB/FC-STB checkpoint into the student."""

    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    if not isinstance(source, nn.Module):
        raise TypeError("Checkpoint FC-STB tidak mengekspos model")
    target_layers = getattr(model, "model", None)
    source_layers = getattr(source, "model", None)
    if not isinstance(target_layers, (nn.Sequential, nn.ModuleList)) or not isinstance(
        source_layers, (nn.Sequential, nn.ModuleList)
    ):
        raise TypeError("Checkpoint FC-STB tidak mengekspos daftar layer")
    if not isinstance(target_layers[-1], STBDetectHead) or not isinstance(
        source_layers[-1], STBDetectHead
    ):
        raise TypeError("FC-STB memerlukan checkpoint STB")
    result = target_layers.load_state_dict(source_layers.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer FC-STB tidak strict")
    target_head, source_head = target_layers[-1], source_layers[-1]
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    return {
        "source": type(source).__name__,
        "items": len(source_layers.state_dict()),
        "strict": True,
    }


class FCSTBDetectionModel(STBDetectionModel):
    """STB detector with a training-only AF2 distillation criterion.

    The teacher is deliberately not registered on this model. Saved checkpoints
    therefore contain exactly one STB detector and inference is one forward.
    """

    def __init__(
        self,
        cfg="yolo26.yaml",
        ch=3,
        nc=None,
        verbose=True,
        stb=None,
        fcstb: FCSTBConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.fcstb_config = FCSTBConfig.from_mapping(fcstb)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, stb=stb)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode or not len(self.model):
            return self
        head = self.model[-1]
        if not isinstance(head, STBDetectHead):
            return self
        # Backbone/neck and box heads are frozen both as parameters and as
        # running-buffer state. STB blocks and classification heads co-adapt.
        for layer in list(self.model)[:-1]:
            layer.eval()
        for block in head.blocks:
            block.train(True)
        for branch in (head.one2many, head.one2one):
            for module in branch["box_head"]:
                module.eval()
            for module in branch["cls_head"]:
                module.train(True)
        return self

    def apply_freeze_policy(self) -> dict[str, int]:
        head = self.model[-1]
        if not isinstance(head, STBDetectHead):
            raise TypeError(type(head).__name__)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in [
            head.blocks,
            *[branch["cls_head"] for branch in (head.one2many, head.one2one)],
        ]:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.train(self.training)
        return {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }

    @property
    def teacher_path(self) -> Path | None:
        value = self.fcstb_config.teacher_checkpoint
        return Path(value).expanduser().resolve() if value else None
