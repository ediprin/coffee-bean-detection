from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from coffee_detector.af2_complement.modules import SpaceFrequencySelectionResidual
from coffee_detector.af2_spds.loss import multilevel_reconstruction_loss
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig, afab_gate, minmax_spatial

from .config import AF2SFSCUEConfig


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class AF2SFSCUEDetectHead(nn.Module):
    """Training-only CUE decoders before an inference-time P3 SFS residual.

    The auxiliary decoders read the untouched P3/P4/P5 tensors.  Only after
    that observation is captured does the identity-initialized SFS adapter
    modify P3 for both native box and class branches.  This ordering prevents
    the reconstruction target from directly constraining the selector output.
    """

    def __init__(self, base_head: nn.Module, config: AF2SFSCUEConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("AF2-SFS-CUE memerlukan native Detect")
        self.base_head = base_head
        self.config = AF2SFSCUEConfig.from_mapping(config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("AF2-SFS-CUE memerlukan tepat P3/P4/P5")
        self.adapter = SpaceFrequencySelectionResidual(
            channels[self.config.feature_level], self.config.lowpass_kernel
        )
        self.decoders = nn.ModuleList(
            [nn.Conv2d(channel, self.config.decoder_channels, 1) for channel in channels]
        )
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
        adapted[level] = self.adapter(adapted[level])
        return self.base_head(adapted)

    def fuse(self) -> None:
        self.base_head.fuse()


class AF2SFSCUEDetectionLoss:
    """Native detection loss plus the frozen pure-AF2-gate CUE objective."""

    def __new__(
        cls, model: nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundAF2SFSCUEDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                self.head = model.model[-1]
                self.config = AF2SFSCUEConfig.from_mapping(model.af2_sfs_cue_config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                predictions = self.head.last_auxiliary_predictions
                if predictions is None:
                    if model.training:
                        raise RuntimeError("Prediksi CUE tidak tersedia saat training")
                    return assignments, loss, loss.detach()
                target = model.last_af2_gate_target
                if target is None:
                    raise RuntimeError("Target pure AF2 gate tidak tersedia")
                auxiliary = multilevel_reconstruction_loss(predictions, target)
                loss[1] = loss[1] + self.config.auxiliary_gain * auxiliary
                return assignments, loss, loss.detach()

        return _BoundAF2SFSCUEDetectionLoss()


class AF2SFSCUEDetectionModel(AFABDetectionModel):
    """AF2 direct model with pre-SFS cue supervision and a persistent P3 SFS."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        afab: AFABConfig | Mapping[str, Any] | None = None,
        sfs_cue: AF2SFSCUEConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.af2_sfs_cue_config = AF2SFSCUEConfig.from_mapping(sfs_cue)
        self.last_af2_gate_target: torch.Tensor | None = None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.model[-1] = AF2SFSCUEDetectHead(
            self.model[-1], self.af2_sfs_cue_config
        )

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "afab", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            recovered = enhancer.recover(x)
            gate = minmax_spatial(recovered, eps=enhancer.config.eps)
            enhanced = afab_gate(x, recovered, eps=enhancer.config.eps)
            self.last_af2_gate_target = gate.detach() if self.training else None
            # Bypass AFABDetectionModel.predict so AF2 is applied exactly once.
            from ultralytics.nn.tasks import DetectionModel

            return DetectionModel.predict(
                self,
                enhanced,
                profile=profile,
                visualize=visualize,
                augment=augment,
                embed=embed,
            )
        from ultralytics.nn.tasks import DetectionModel

        return DetectionModel.predict(
            self, x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=AF2SFSCUEDetectionLoss)
        return AF2SFSCUEDetectionLoss(self)


def _source_model(weights: Any) -> nn.Module:
    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    if not isinstance(source, nn.Module):
        raise TypeError("Checkpoint tidak mengekspos model torch")
    return source


def load_af2_sfs_cue_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Transfer all shape-compatible native detector tensors into the wrapper."""

    source = _source_model(weights)
    source_layers = getattr(source, "model", None)
    target_layers = getattr(model, "model", None)
    if not isinstance(source_layers, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Source tidak mengekspos daftar layer")
    if not isinstance(target_layers, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Target tidak mengekspos daftar layer")
    target_head = target_layers[-1]
    if not isinstance(target_head, AF2SFSCUEDetectHead):
        raise TypeError("Target bukan AF2SFSCUEDetectHead")

    source_head = source_layers[-1]
    if isinstance(source_head, AF2SFSCUEDetectHead):
        result = model.load_state_dict(source.state_dict(), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Resume AF2-SFS-CUE tidak exact")
        return {
            "source_items": len(source.state_dict()),
            "target_items": len(model.state_dict()),
            "transferred_items": len(model.state_dict()),
            "missing_after_partial_load": 0,
        }

    source_state = source.state_dict()
    target_state = model.state_dict()
    head_index = len(target_layers) - 1
    wrapped_prefix = f"model.{head_index}.base_head."
    native_prefix = f"model.{head_index}."
    transfer: dict[str, torch.Tensor] = {}
    for target_key, target_value in target_state.items():
        source_key = target_key
        if target_key.startswith(wrapped_prefix):
            source_key = native_prefix + target_key[len(wrapped_prefix):]
        source_value = source_state.get(source_key)
        if source_value is not None and source_value.shape == target_value.shape:
            transfer[target_key] = source_value
    result = model.load_state_dict(transfer, strict=False)
    return {
        "source_items": len(source_state),
        "target_items": len(target_state),
        "transferred_items": len(transfer),
        "missing_after_partial_load": len(result.missing_keys),
    }


def canonical_native_state(model: AF2SFSCUEDetectionModel) -> dict[str, torch.Tensor]:
    """Return detector-only state under the same keys as native AF2DIRECT."""

    layers = model.model
    head_index = len(layers) - 1
    wrapped_prefix = f"model.{head_index}.base_head."
    native_prefix = f"model.{head_index}."
    result: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if f"model.{head_index}.adapter." in key or f"model.{head_index}.decoders." in key:
            continue
        canonical = native_prefix + key[len(wrapped_prefix):] if key.startswith(wrapped_prefix) else key
        result[canonical] = value
    return result
