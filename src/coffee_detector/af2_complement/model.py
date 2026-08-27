from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig

from .config import AF2ComplementConfig
from .loss import AF2ComplementDetectionLoss
from .modules import FrequencySelectionResidual, SpaceFrequencySelectionResidual


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class SharedFeatureDetectHead(nn.Module):
    """Native Detect head preceded by one identity-initialized shared P3 adapter."""

    def __init__(self, base_head: nn.Module, config: AF2ComplementConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("AF2 complement memerlukan native Detect")
        self.base_head = base_head
        self.config = AF2ComplementConfig.from_mapping(config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("AF2 complement memerlukan P3/P4/P5")
        channel = channels[self.config.feature_level]
        if self.config.mode == "frequency_select":
            self.adapter = FrequencySelectionResidual(channel, self.config.lowpass_kernel)
        elif self.config.mode == "space_frequency":
            self.adapter = SpaceFrequencySelectionResidual(channel, self.config.lowpass_kernel)
        else:
            self.adapter = nn.Identity()

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

    def adapt(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        result = list(features)
        result[self.config.feature_level] = self.adapter(result[self.config.feature_level])
        return result

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        return self.base_head(self.adapt(features))

    def fuse(self) -> None:
        self.base_head.fuse()


class AF2ComplementDetectionModel(AFABDetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        afab: AFABConfig | Mapping[str, Any] | None = None,
        complement: AF2ComplementConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.af2_complement_config = AF2ComplementConfig.from_mapping(complement)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.model[-1] = SharedFeatureDetectHead(
            self.model[-1], self.af2_complement_config
        )

    def init_criterion(self):
        if self.af2_complement_config.mode != "bhcl":
            return super().init_criterion()
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=AF2ComplementDetectionLoss)
        return AF2ComplementDetectionLoss(self)


def load_af2_complement_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load an AF2 parent (or resume arm) while preserving the frozen frontend."""

    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    source_model = getattr(source, "model", None)
    target_model = getattr(model, "model", None)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    if not isinstance(target_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Target tidak mengekspos daftar layer model")
    target_head = target_model[-1]
    if not isinstance(target_head, SharedFeatureDetectHead):
        raise TypeError("Target bukan SharedFeatureDetectHead")

    model.load(source)
    source_head = source_model[-1]
    if isinstance(source_head, SharedFeatureDetectHead):
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
