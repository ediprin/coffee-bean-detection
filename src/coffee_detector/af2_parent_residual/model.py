from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.igem.model import ClassAwareReferenceLevel, _first_conv_channels
from coffee_detector.safpn_alignment.model import SAFPNClassificationCorrection

from .config import AF2ParentResidualConfig


class AF2ParentResidualDetectHead(nn.Module):
    """Frozen native AF2 head plus a trainable classification-only residual.

    ``conditioning=zero`` retains the complete residual architecture while
    hiding P3/P4/P5 information. It is therefore the matched optimization and
    capacity control for ``conditioning=feature``.
    """

    def __init__(self, base_head: nn.Module, config: AF2ParentResidualConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect" or not getattr(base_head, "end2end", False):
            raise TypeError("AF2 parent residual memerlukan native YOLO26 end-to-end Detect")
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        if len(channels) != 3:
            raise ValueError("AF2 parent residual memerlukan P3/P4/P5")
        self.base_head = base_head
        self.config = config
        if config.family == "saf":
            self.residual = SAFPNClassificationCorrection(
                channels, int(base_head.nc), config.saf
            )
        else:
            self.residual = nn.ModuleList(
                [
                    ClassAwareReferenceLevel(channel, int(base_head.nc), config.igem)
                    for channel in channels
                ]
            )
        self.last_diagnostics: dict[str, torch.Tensor] = {}
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

    @property
    def residual_parameters(self):
        return self.residual.parameters()

    def _sync_runtime_attributes(self) -> None:
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    @staticmethod
    def _has_heads(branch: dict[str, nn.Module]) -> bool:
        return bool(branch.get("box_head")) and bool(branch.get("cls_head"))

    def _conditioned(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.config.conditioning == "zero":
            return [torch.zeros_like(value) for value in features]
        return features

    def _residual_outputs(
        self, features: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        conditioned = self._conditioned(features)
        if self.config.family == "saf":
            corrections, diagnostics = self.residual(conditioned)
            self.last_diagnostics = diagnostics
            return corrections, []
        corrections, masks = [], []
        for level, feature in zip(self.residual, conditioned):
            correction, mask = level(feature)
            corrections.append(float(self.config.igem.correction_scale) * correction)
            masks.append(mask)
        return corrections, masks

    def _forward_branch(
        self,
        features: list[torch.Tensor],
        branch: dict[str, nn.Module],
        *,
        expose_masks: bool,
    ) -> dict[str, Any]:
        boxes = [branch["box_head"][i](features[i]) for i in range(self.nl)]
        native_scores = [branch["cls_head"][i](features[i]) for i in range(self.nl)]
        corrections, masks = self._residual_outputs(features)
        batch = features[0].shape[0]
        output: dict[str, Any] = {
            "boxes": torch.cat(
                [value.view(batch, 4 * self.reg_max, -1) for value in boxes], dim=-1
            ),
            "scores": torch.cat(
                [
                    (native + correction).view(batch, self.nc, -1)
                    for native, correction in zip(native_scores, corrections)
                ],
                dim=-1,
            ),
            "feats": features,
        }
        if expose_masks and masks:
            output["parent_residual_mask_logits"] = masks
        return output

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if self.training:
            return {
                "one2many": self._forward_branch(features, self.one2many, expose_masks=True),
                "one2one": self._forward_branch(
                    [value.detach() for value in features],
                    self.one2one,
                    expose_masks=False,
                ),
            }
        one2many = (
            self._forward_branch(features, self.one2many, expose_masks=False)
            if self._has_heads(self.one2many)
            else None
        )
        one2one = self._forward_branch(
            [value.detach() for value in features], self.one2one, expose_masks=False
        )
        predictions = {"one2one": one2one}
        if one2many is not None:
            predictions["one2many"] = one2many
        inference = self.base_head._inference(one2one)
        output = self.base_head.postprocess(inference.permute(0, 2, 1))
        return output if self.export else (output, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


class AF2ParentResidualDetectionModel(AFABDetectionModel):
    def __init__(
        self,
        cfg="yolo26.yaml",
        ch=3,
        nc=None,
        verbose=True,
        afab=None,
        parent_residual=None,
    ) -> None:
        self.parent_residual_config = AF2ParentResidualConfig.from_mapping(parent_residual)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.model[-1] = AF2ParentResidualDetectHead(
            self.model[-1], self.parent_residual_config
        )
        freeze_for_parent_residual(self)

    def init_criterion(self):
        if self.parent_residual_config.family == "igem":
            from ultralytics.utils.loss import E2ELoss

            from .loss import AF2ParentResidualDetectionLoss

            return E2ELoss(self, loss_fn=AF2ParentResidualDetectionLoss)
        return super().init_criterion()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and len(self.model):
            head = self.model[-1]
            for layer in list(self.model)[:-1]:
                layer.eval()
            head.base_head.eval()
            head.residual.train(True)
        return self


def freeze_for_parent_residual(model: nn.Module) -> dict[str, int]:
    layers = getattr(model, "model", None)
    if not isinstance(layers, (nn.Sequential, nn.ModuleList)) or not isinstance(
        layers[-1], AF2ParentResidualDetectHead
    ):
        raise TypeError("freeze_for_parent_residual memerlukan head yang sesuai")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in layers[-1].residual.parameters():
        parameter.requires_grad_(True)
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def load_af2_parent_residual_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Strictly load a completed AF2 parent or resume this exact architecture."""

    model.load(weights)
    source_has_af2 = getattr(weights, "afab", None) is not None
    source_model = getattr(weights, "model", None)
    target = getattr(model, "model", model)
    if not isinstance(source_model, (nn.Sequential, nn.ModuleList)):
        raise TypeError("Checkpoint tidak mengekspos daftar layer model")
    source_head, target_head = source_model[-1], target[-1]
    if not isinstance(target_head, AF2ParentResidualDetectHead):
        raise TypeError("Target bukan AF2ParentResidualDetectHead")
    if isinstance(source_head, AF2ParentResidualDetectHead):
        if source_head.config != target_head.config:
            raise RuntimeError("Resume arm/config tidak identik")
        target_head.load_state_dict(copy.deepcopy(source_head.state_dict()), strict=True)
        resume = 1
    else:
        if type(source_head).__name__ != "Detect" or not source_has_af2:
            raise TypeError("Source harus checkpoint AF2 dengan native Detect")
        result = target_head.base_head.load_state_dict(
            copy.deepcopy(source_head.state_dict()), strict=True
        )
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Transfer native head AF2 tidak lengkap")
        resume = 0
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
    freeze_for_parent_residual(model)
    return {
        "native_head_items": len(target_head.base_head.state_dict()),
        "resume": resume,
    }
