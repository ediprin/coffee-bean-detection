from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig, afab_gate, minmax_spatial

from .config import AF2SPDSConfig
from .loss import AF2SPDSDetectionLoss


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class AuxiliaryReconstructionDetectHead(nn.Module):
    """Native Detect plus training-only read-only decoders on P3/P4/P5."""

    def __init__(self, base_head: nn.Module, config: AF2SPDSConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("AF2-SPDS memerlukan native Detect")
        self.base_head = base_head
        self.config = AF2SPDSConfig.from_mapping(config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("AF2-SPDS memerlukan tepat P3/P4/P5")
        self.decoders = nn.ModuleList(
            [nn.Conv2d(channel, self.config.decoder_channels, 1) for channel in channels]
        )
        self.last_auxiliary_predictions: list[torch.Tensor] | None = None

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
        self.last_auxiliary_predictions = (
            [decoder(feature) for decoder, feature in zip(self.decoders, features)]
            if self.training
            else None
        )
        # Crucial contract: features are never replaced or modified.
        return self.base_head(features)

    def fuse(self) -> None:
        self.base_head.fuse()


class AF2SPDSDetectionModel(AFABDetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        afab: AFABConfig | Mapping[str, Any] | None = None,
        spds: AF2SPDSConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.af2_spds_config = AF2SPDSConfig.from_mapping(spds)
        self.last_auxiliary_targets: dict[str, torch.Tensor] | None = None
        self.af2_spds_epoch = 0
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.model[-1] = AuxiliaryReconstructionDetectHead(
            self.model[-1], self.af2_spds_config
        )

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "afab", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            raw = x
            # Compute recovery once, then expose the actual normalized AF2 gate
            # separately from the RGB-modulated residual used by the first study.
            recovered = enhancer.recover(raw)
            gate = minmax_spatial(recovered, eps=enhancer.config.eps)
            enhanced = afab_gate(raw, recovered, eps=enhancer.config.eps)
            if self.training:
                self.last_auxiliary_targets = {
                    "rgb": raw.detach(),
                    "af2_signal": (enhanced - raw).detach(),
                    "af2_gate": gate.detach(),
                }
            else:
                self.last_auxiliary_targets = None
            # Bypass AFABDetectionModel.predict to avoid applying AF2 twice.
            from ultralytics.nn.tasks import DetectionModel

            return DetectionModel.predict(
                self,
                enhanced,
                profile=profile,
                visualize=visualize,
                augment=augment,
                embed=embed,
            )
        # DetectionModel calls predict while it is still constructing strides;
        # AFAB is deliberately attached only after that initialization pass.
        from ultralytics.nn.tasks import DetectionModel

        return DetectionModel.predict(
            self, x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=AF2SPDSDetectionLoss)
        return AF2SPDSDetectionLoss(self)


def load_af2_spds_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load native AF2 or resume AF2-SPDS weights without changing AF2 config."""

    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    source_model = getattr(source, "model", None)
    target_model = getattr(model, "model", None)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    if not isinstance(target_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Target tidak mengekspos daftar layer model")
    target_head = target_model[-1]
    if not isinstance(target_head, AuxiliaryReconstructionDetectHead):
        raise TypeError("Target bukan AuxiliaryReconstructionDetectHead")

    model.load(source)
    source_head = source_model[-1]
    if isinstance(source_head, AuxiliaryReconstructionDetectHead):
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


def strip_auxiliary_head(model: nn.Module) -> nn.Module:
    """Remove training-only decoders in-place and restore the native Detect head."""

    layers = getattr(model, "model", None)
    if not isinstance(layers, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Model tidak mengekspos daftar layer")
    head = layers[-1]
    if not isinstance(head, AuxiliaryReconstructionDetectHead):
        raise TypeError("Model tidak memiliki head auxiliary AF2-SPDS")
    native = head.base_head
    native.stride = head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(head, name):
            setattr(native, name, getattr(head, name))
    layers[-1] = native
    model.stride = native.stride
    model.af2_spds_config = None
    model.last_auxiliary_targets = None
    return model
