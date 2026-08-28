from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig

from .config import AF2ScaffoldConfig
from .modules import MultilevelTrainingScaffold


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class TrainingOnlyMultilevelDetectHead(nn.Module):
    """Native Detect head with removable train-only P3/P4/P5 scaffolds."""

    def __init__(self, base_head: nn.Module, config: AF2ScaffoldConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("AF2MTS1 memerlukan native Detect")
        self.base_head = base_head
        self.config = AF2ScaffoldConfig.from_mapping(config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.scaffold = MultilevelTrainingScaffold(channels, self.config.lowpass_kernel)
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

    def set_scaffold_strength(self, value: float) -> None:
        self.scaffold.set_strength(value)

    def _sync_runtime_attributes(self) -> None:
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    def forward(self, features):
        self._sync_runtime_attributes()
        return self.base_head(self.scaffold(list(features)))

    def fuse(self) -> None:
        self.base_head.fuse()


class AF2ScaffoldDetectionModel(AFABDetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        afab: AFABConfig | Mapping[str, Any] | None = None,
        scaffold: AF2ScaffoldConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.af2_scaffold_config = AF2ScaffoldConfig.from_mapping(scaffold)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.model[-1] = TrainingOnlyMultilevelDetectHead(
            self.model[-1], self.af2_scaffold_config
        )


def load_af2_scaffold_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    source_model = getattr(source, "model", None)
    target_model = getattr(model, "model", None)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    if not isinstance(target_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Target tidak mengekspos daftar layer model")
    target_head = target_model[-1]
    if not isinstance(target_head, TrainingOnlyMultilevelDetectHead):
        raise TypeError("Target bukan TrainingOnlyMultilevelDetectHead")

    model.load(source)
    source_head = source_model[-1]
    if isinstance(source_head, TrainingOnlyMultilevelDetectHead):
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


def strip_training_scaffold(model: nn.Module) -> nn.Module:
    """Remove the wrapper in-place, leaving the learned native AF2 detector."""

    layers = getattr(model, "model", None)
    if not isinstance(layers, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Model tidak mengekspos daftar layer")
    head = layers[-1]
    if not isinstance(head, TrainingOnlyMultilevelDetectHead):
        raise TypeError("Model tidak memiliki training scaffold")
    layers[-1] = head.base_head
    return model
