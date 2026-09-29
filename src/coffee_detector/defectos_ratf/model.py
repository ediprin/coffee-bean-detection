from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.j25_cwcf.model import _first_conv_channels

from .config import RATFConfig
from .operator import ratf_texture_cue


class RedundancyAwareTextureResidual(nn.Module):
    """Per-class score residual from texture information orthogonal to RGB."""

    def __init__(self, channels: int, classes: int, config: RATFConfig) -> None:
        super().__init__()
        self.channels = int(channels)
        self.classes = int(classes)
        self.config = RATFConfig.from_mapping(config)
        self.cue_encoder = nn.Conv2d(self.config.cue_channels, channels, 1, bias=False)
        self.class_gate = nn.Conv2d(channels, classes, 1, bias=True)
        self.class_residual = nn.Conv2d(channels, classes, 1, bias=True)
        nn.init.kaiming_uniform_(self.cue_encoder.weight, a=5**0.5)
        nn.init.zeros_(self.class_gate.weight)
        nn.init.zeros_(self.class_gate.bias)
        nn.init.zeros_(self.class_residual.weight)
        nn.init.zeros_(self.class_residual.bias)
        self.last_novel: torch.Tensor | None = None
        self.last_reference: torch.Tensor | None = None

    def decompose(
        self, feature: torch.Tensor, cue: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resized = F.interpolate(cue, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        texture = self.cue_encoder(resized)
        reference = feature
        coefficient = (texture * reference).sum(dim=1, keepdim=True) / (
            reference.square().sum(dim=1, keepdim=True)
            + self.config.orthogonality_epsilon
        )
        novel = texture - coefficient * reference
        return novel, reference

    def forward(self, feature: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
        novel, reference = self.decompose(feature, cue)
        self.last_novel = novel
        self.last_reference = reference
        gate = torch.sigmoid(self.class_gate(novel))
        residual = self.config.residual_logit_bound * torch.tanh(
            self.class_residual(novel)
        )
        return gate * residual


class RATFDetectHead(nn.Module):
    """Native Detect with RATF applied only to per-class logits."""

    def __init__(self, base_head: nn.Module, config: RATFConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("RATF1 memerlukan native Detect")
        self.base_head = base_head
        self.config = RATFConfig.from_mapping(config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.adapters = nn.ModuleList(
            [
                RedundancyAwareTextureResidual(channel, int(base_head.nc), self.config)
                for channel in channels
            ]
        )
        self.current_cue: torch.Tensor | None = None
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

    def set_cue(self, cue: torch.Tensor) -> None:
        self.current_cue = cue

    def _sync_runtime_attributes(self) -> None:
        for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
            if hasattr(self, name):
                setattr(self.base_head, name, getattr(self, name))

    def _forward_head(
        self,
        features: list[torch.Tensor],
        *,
        box_head: nn.Module,
        cls_head: nn.Module,
    ) -> dict[str, torch.Tensor]:
        if self.current_cue is None:
            raise RuntimeError("Cue RATF belum disetel")
        batch = features[0].shape[0]
        boxes, scores = [], []
        for index in range(self.nl):
            feature = features[index]
            native_score = cls_head[index](feature)
            score = native_score + self.adapters[index](feature, self.current_cue)
            boxes.append(box_head[index](feature).view(batch, 4 * self.reg_max, -1))
            scores.append(score.view(batch, self.nc, -1))
        return {
            "boxes": torch.cat(boxes, dim=-1),
            "scores": torch.cat(scores, dim=-1),
            "feats": features,
        }

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        one2many = self.base_head.one2many
        fused = one2many.get("box_head") is None or one2many.get("cls_head") is None
        predictions = None
        if not fused:
            predictions = self._forward_head(features, **one2many)
        if self.end2end:
            detached = [value.detach() for value in features]
            one2one = self._forward_head(detached, **self.base_head.one2one)
            predictions = (
                {"one2one": one2one}
                if fused
                else {"one2many": predictions, "one2one": one2one}
            )
        elif predictions is None:
            raise RuntimeError("Head RATF non-end2end tidak memiliki cabang inference")
        self.current_cue = None
        if self.training:
            if fused:
                raise RuntimeError("Model fused tidak boleh digunakan untuk training")
            return predictions
        inference = self.base_head._inference(
            predictions["one2one"] if self.end2end else predictions
        )
        if self.end2end:
            inference = self.base_head.postprocess(inference.permute(0, 2, 1))
        return inference if self.export else (inference, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


from ultralytics.nn.tasks import DetectionModel


class RATFDetectionModel(DetectionModel):
    """YOLO detector with RGB-native boxes and RATF classification scores."""

    def __init__(
        self,
        cfg: str | dict,
        *,
        ch: int = 3,
        nc: int = 12,
        verbose: bool = False,
        ratf: RATFConfig | Mapping[str, Any] | None = None,
        native_source: nn.Module | None = None,
    ) -> None:
        self.ratf_config = RATFConfig.from_mapping(ratf)
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        if native_source is not None:
            self.load(native_source)
        self.model[-1] = RATFDetectHead(self.model[-1], self.ratf_config)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        head = self.model[-1] if hasattr(self, "model") and len(self.model) else None
        if isinstance(x, torch.Tensor) and isinstance(head, RATFDetectHead):
            head.set_cue(ratf_texture_cue(x, self.ratf_config).detach())
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def build_ratf_model(
    model_yaml: str,
    *,
    nc: int,
    source: nn.Module | None,
    seed: int,
    config: RATFConfig | Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> RATFDetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return RATFDetectionModel(
            model_yaml,
            ch=3,
            nc=nc,
            verbose=verbose,
            ratf=config,
            native_source=source,
        )


def load_ratf_weights(model: nn.Module, weights: Any) -> None:
    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    if not isinstance(source, nn.Module):
        raise TypeError("Checkpoint RATF1 tidak mengekspos model torch")
    result = model.load_state_dict(source.float().state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Restore RATF1 tidak lengkap")
