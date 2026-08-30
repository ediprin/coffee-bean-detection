from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.af2_complement.modules import SpaceFrequencySelectionResidual
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig, afab_gate, minmax_spatial

from .config import AF2CurriculumSFSConfig, curriculum_state


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


def multilevel_gate_loss(
    predictions: list[torch.Tensor], target: torch.Tensor
) -> torch.Tensor:
    if not predictions:
        raise ValueError("Prediksi auxiliary kosong")
    losses = []
    for prediction in predictions:
        resized = F.interpolate(
            target.float(), size=prediction.shape[-2:], mode="area"
        ).to(dtype=prediction.dtype)
        losses.append(F.smooth_l1_loss(prediction, resized, reduction="mean"))
    return torch.stack(losses).mean()


def aligned_auxiliary_scale(
    detection_gradient: torch.Tensor | None,
    auxiliary_gradient: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detach a nonnegative cosine gate so auxiliary updates cannot oppose detection."""

    reference = detection_gradient if detection_gradient is not None else auxiliary_gradient
    if reference is None:
        zero = torch.tensor(0.0)
        return zero, zero
    if detection_gradient is None or auxiliary_gradient is None:
        zero = reference.new_zeros(())
        return zero, zero
    first = detection_gradient.detach().float().reshape(-1)
    second = auxiliary_gradient.detach().float().reshape(-1)
    denominator = first.norm() * second.norm()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
        zero = reference.new_zeros(())
        return zero, zero
    cosine = torch.dot(first, second) / denominator.clamp_min(1.0e-12)
    cosine = torch.nan_to_num(cosine, nan=0.0, posinf=0.0, neginf=0.0)
    return cosine.clamp_min(0.0).to(reference.dtype), cosine.to(reference.dtype)


class AF2CurriculumSFSHead(nn.Module):
    """Native Detect head with a scheduled identity SFS and train-only cue decoder."""

    def __init__(self, base_head: nn.Module, config: AF2CurriculumSFSConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("AF2 curriculum-SFS memerlukan native Detect")
        self.base_head = base_head
        self.config = AF2CurriculumSFSConfig.from_mapping(config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("AF2 curriculum-SFS memerlukan P3/P4/P5")
        self.adapter = SpaceFrequencySelectionResidual(
            channels[self.config.feature_level], self.config.lowpass_kernel
        )
        self.decoders = nn.ModuleList([nn.Conv2d(channel, 3, 1) for channel in channels])
        self.sfs_strength = 0.0
        self.last_auxiliary_predictions: list[torch.Tensor] | None = None
        self.last_pre_adapter_features: list[torch.Tensor] | None = None

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
        if self.training:
            self.last_pre_adapter_features = list(features)
            self.last_auxiliary_predictions = [
                decoder(feature) for decoder, feature in zip(self.decoders, features)
            ]
        else:
            self.last_pre_adapter_features = None
            self.last_auxiliary_predictions = None
        adapted = list(features)
        level = self.config.feature_level
        source = adapted[level]
        selected = self.adapter(source)
        adapted[level] = source + float(self.sfs_strength) * (selected - source)
        return self.base_head(adapted)

    def fuse(self) -> None:
        self.base_head.fuse()


class AF2CurriculumE2ELoss:
    """Native E2E loss plus one target-prioritized, gradient-aligned cue loss."""

    def __init__(self, model: nn.Module) -> None:
        from ultralytics.utils.loss import E2ELoss

        self.native = E2ELoss(model)
        self.model = model
        self.head = model.model[-1]
        self.config = AF2CurriculumSFSConfig.from_mapping(model.af2_curriculum_config)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]):
        detection, items = self.native(preds, batch)
        state = curriculum_state(
            self.config,
            epoch=int(getattr(self.model, "af2_curriculum_epoch", 0)),
            epochs=int(getattr(self.model.args, "epochs", self.config.total_epochs)),
        )
        predictions = self.head.last_auxiliary_predictions
        target = self.model.last_af2_gate_target
        features = self.head.last_pre_adapter_features
        if predictions is None or target is None or features is None:
            if self.model.training:
                raise RuntimeError("Target/prediksi curriculum tidak tersedia")
            return detection, items

        auxiliary = multilevel_gate_loss(predictions, target)
        scale = auxiliary.new_zeros(())
        cosine = auxiliary.new_zeros(())
        if state.auxiliary_gain > 0.0:
            probe = features[self.config.feature_level]
            detection_gradient = torch.autograd.grad(
                detection.sum(), probe, retain_graph=True, allow_unused=True
            )[0]
            auxiliary_gradient = torch.autograd.grad(
                auxiliary, probe, retain_graph=True, allow_unused=True
            )[0]
            scale, cosine = aligned_auxiliary_scale(
                detection_gradient, auxiliary_gradient
            )
            effective_gain = float(state.auxiliary_gain) * scale
            detection[1] = detection[1] + effective_gain * auxiliary * int(batch["img"].shape[0])
        self.model.last_curriculum_diagnostics = {
            "phase": state.phase,
            "sfs_strength": float(state.sfs_strength),
            "scheduled_auxiliary_gain": float(state.auxiliary_gain),
            "gradient_cosine": float(cosine.detach().cpu()),
            "alignment_scale": float(scale.detach().cpu()),
            "auxiliary_loss": float(auxiliary.detach().cpu()),
        }
        return detection, items

    def update(self) -> None:
        self.native.update()


class AF2CurriculumSFSDetectionModel(AFABDetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        afab: AFABConfig | Mapping[str, Any] | None = None,
        curriculum: AF2CurriculumSFSConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.af2_curriculum_config = AF2CurriculumSFSConfig.from_mapping(curriculum)
        self.af2_curriculum_epoch = 0
        self.last_af2_gate_target: torch.Tensor | None = None
        self.last_curriculum_diagnostics: dict[str, Any] | None = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.model[-1] = AF2CurriculumSFSHead(
            self.model[-1], self.af2_curriculum_config
        )

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        head = self.model[-1]
        state = curriculum_state(
            self.af2_curriculum_config,
            epoch=int(self.af2_curriculum_epoch),
            epochs=self.af2_curriculum_config.total_epochs,
        )
        head.sfs_strength = state.sfs_strength if self.training else 1.0
        enhancer = getattr(self, "afab", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            recovered = enhancer.recover(x)
            gate = minmax_spatial(recovered, eps=enhancer.config.eps)
            enhanced = afab_gate(x, recovered, eps=enhancer.config.eps)
            self.last_af2_gate_target = gate.detach() if self.training else None
            from ultralytics.nn.tasks import DetectionModel

            return DetectionModel.predict(
                self, enhanced, profile=profile, visualize=visualize,
                augment=augment, embed=embed,
            )
        from ultralytics.nn.tasks import DetectionModel

        return DetectionModel.predict(
            self, x, profile=profile, visualize=visualize,
            augment=augment, embed=embed,
        )

    def init_criterion(self):
        if not getattr(self, "end2end", False):
            raise RuntimeError("AF2 curriculum-SFS dikunci pada YOLO26 end-to-end")
        return AF2CurriculumE2ELoss(self)


def load_af2_curriculum_sfs_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    if not isinstance(source, nn.Module):
        raise TypeError("Checkpoint tidak mengekspos model torch")
    source_layers = getattr(source, "model", None)
    target_layers = getattr(model, "model", None)
    if not isinstance(source_layers, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Source tidak mengekspos daftar layer")
    if not isinstance(target_layers, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Target tidak mengekspos daftar layer")
    target_head = target_layers[-1]
    if not isinstance(target_head, AF2CurriculumSFSHead):
        raise TypeError("Target bukan AF2CurriculumSFSHead")

    source_head = source_layers[-1]
    if isinstance(source_head, AF2CurriculumSFSHead):
        result = model.load_state_dict(source.state_dict(), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Resume AF2 curriculum-SFS tidak exact")
        return {"resume": 1, "transferred_items": len(model.state_dict())}

    model.load(source)
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Source head bukan native Detect: {type(source_head).__name__}")
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Transfer native Detect tidak lengkap")
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    return {"resume": 0, "transferred_items": len(source.state_dict())}
