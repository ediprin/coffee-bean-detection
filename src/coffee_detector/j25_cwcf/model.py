from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.data.prepare_coffee_standard_primary import J25_CLASSES

from .config import CWCFConfig
from .operator import chromatic_wavelet_cue


ATTRIBUTE_NAMES = (
    "moldy", "skin", "hole", "spotted", "brown_yellow", "black", "broken",
    "immature", "normal", "foreign", "fruit", "large", "small", "medium",
)


def build_j25_attribute_matrix() -> torch.Tensor:
    """Deterministic class-to-attribute factorization for the frozen J25 labels."""

    groups = {
        "moldy": {0},
        "skin": {1, 2, 16, 17, 18, 19, 20, 21},
        "hole": {3, 4},
        "spotted": {5},
        "brown_yellow": {6, 13},
        "black": {7, 8, 9},
        "broken": {7, 12},
        "immature": {10},
        "normal": {11},
        "foreign": {14, 22, 23, 24},
        "fruit": {15},
        "large": {16, 19, 22},
        "small": {17, 20, 23},
        "medium": {18, 21, 24},
    }
    matrix = torch.zeros(len(J25_CLASSES), len(ATTRIBUTE_NAMES), dtype=torch.float32)
    for attribute_index, name in enumerate(ATTRIBUTE_NAMES):
        for class_index in groups[name]:
            matrix[class_index, attribute_index] = 1.0
    return matrix


def _first_conv_channels(module: nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


class CueAffineResidual(nn.Module):
    """Zero-initialized cue-conditioned residual for one pyramid feature."""

    def __init__(self, channels: int, cue_channels: int) -> None:
        super().__init__()
        self.affine = nn.Conv2d(cue_channels, 2 * channels, 1, bias=True)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, feature: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(cue, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        scale, bias = self.affine(resized).chunk(2, dim=1)
        return feature + feature * torch.tanh(scale) + bias


class ChromaticWaveletDetectHead(nn.Module):
    """Native Detect with cue conditioning restricted to classification paths."""

    def __init__(self, base_head: nn.Module, config: CWCFConfig) -> None:
        super().__init__()
        if type(base_head).__name__ != "Detect":
            raise TypeError("CWCF memerlukan native Detect")
        self.base_head = base_head
        self.config = CWCFConfig.from_mapping(config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.adapters = nn.ModuleList(
            [CueAffineResidual(channel, self.config.cue_channels) for channel in channels]
        )
        self.attribute_heads = nn.ModuleList(
            [nn.Conv2d(channel, len(ATTRIBUTE_NAMES), 1) for channel in channels]
        )
        self.current_cue: torch.Tensor | None = None
        self.last_attribute_logits: torch.Tensor | None = None
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
        store_attributes: bool,
    ) -> dict[str, torch.Tensor]:
        if self.current_cue is None:
            raise RuntimeError("Cue CWCF belum disetel")
        batch = features[0].shape[0]
        boxes, scores, attributes = [], [], []
        for index in range(self.nl):
            feature = features[index]
            conditioned = self.adapters[index](feature, self.current_cue)
            boxes.append(box_head[index](feature).view(batch, 4 * self.reg_max, -1))
            scores.append(cls_head[index](conditioned).view(batch, self.nc, -1))
            if store_attributes:
                attributes.append(
                    self.attribute_heads[index](conditioned).view(
                        batch, len(ATTRIBUTE_NAMES), -1
                    )
                )
        if store_attributes:
            self.last_attribute_logits = torch.cat(attributes, dim=-1)
        return {
            "boxes": torch.cat(boxes, dim=-1),
            "scores": torch.cat(scores, dim=-1),
            "feats": features,
        }

    def forward(self, features: list[torch.Tensor]):
        self._sync_runtime_attributes()
        if not self.training:
            self.last_attribute_logits = None
        predictions = self._forward_head(
            features, **self.base_head.one2many, store_attributes=self.training
        )
        if self.end2end:
            detached = [value.detach() for value in features]
            one2one = self._forward_head(
                detached, **self.base_head.one2one, store_attributes=False
            )
            predictions = {"one2many": predictions, "one2one": one2one}
        self.current_cue = None
        if self.training:
            return predictions
        inference = self.base_head._inference(
            predictions["one2one"] if self.end2end else predictions
        )
        if self.end2end:
            inference = self.base_head.postprocess(inference.permute(0, 2, 1))
        return inference if self.export else (inference, predictions)

    def fuse(self) -> None:
        self.base_head.fuse()


def _aggregate_assigned_attributes(
    logits: torch.Tensor,
    foreground: torch.Tensor,
    target_gt_index: torch.Tensor,
    gt_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions, labels = [], []
    anchors = logits.permute(0, 2, 1).contiguous()
    for image_index in range(anchors.shape[0]):
        mask = foreground[image_index]
        if not bool(mask.any()):
            continue
        assigned = target_gt_index[image_index, mask].long()
        selected_logits = anchors[image_index, mask]
        for gt_index in torch.unique(assigned, sorted=True):
            predictions.append(selected_logits[assigned.eq(gt_index)].mean(dim=0))
            labels.append(gt_labels[image_index, gt_index])
    if not predictions:
        return logits.new_zeros((0, logits.shape[1])), gt_labels.new_zeros((0,))
    return torch.stack(predictions), torch.stack(labels).long()


class CWCFDetectionLoss:
    """Native detection loss plus object-level compositional supervision."""

    def __new__(cls, model: nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundCWCFDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                self.head = model.model[-1]
                self.config = CWCFConfig.from_mapping(model.cwcf_config)
                self.apply_auxiliary = tal_topk2 is None
                self.attribute_matrix = build_j25_attribute_matrix()

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                logits = self.head.last_attribute_logits
                self.head.last_attribute_logits = None
                if not self.apply_auxiliary or logits is None:
                    return assignments, loss, loss.detach()
                foreground, target_gt_index = assignments[:2]
                if not bool(foreground.any()):
                    return assignments, loss, loss.detach()
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                image_size = (
                    torch.tensor(
                        preds["feats"][0].shape[2:],
                        device=self.device,
                        dtype=pred_scores.dtype,
                    )
                    * self.stride[0]
                )
                targets = torch.cat(
                    (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
                    dim=1,
                )
                targets = self.preprocess(
                    targets.to(self.device),
                    pred_scores.shape[0],
                    scale_tensor=image_size[[1, 0, 1, 0]],
                )
                gt_labels = targets[..., 0].long()
                attribute_logits, labels = _aggregate_assigned_attributes(
                    logits, foreground, target_gt_index, gt_labels
                )
                if not labels.numel():
                    return assignments, loss, loss.detach()
                attribute_targets = self.attribute_matrix.to(
                    device=labels.device, dtype=attribute_logits.dtype
                )[labels]
                auxiliary = F.binary_cross_entropy_with_logits(
                    attribute_logits, attribute_targets
                )
                model.last_attribute_loss = auxiliary.detach()
                loss[1] = loss[1] + self.config.attribute_gain * auxiliary
                return assignments, loss, loss.detach()

        return _BoundCWCFDetectionLoss()


from ultralytics.nn.tasks import DetectionModel


class CWCFDetectionModel(DetectionModel):
    """YOLO detector with a J25-only chromatic-wavelet classification path."""

    def __init__(
        self,
        cfg: str | dict,
        *,
        ch: int = 3,
        nc: int = 25,
        verbose: bool = False,
        cwcf: CWCFConfig | Mapping[str, Any] | None = None,
        native_source: nn.Module | None = None,
    ) -> None:
        self.cwcf_config = CWCFConfig.from_mapping(cwcf)
        self.last_attribute_loss: torch.Tensor | None = None
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        # Transfer the official detector before changing the native head schema.
        if native_source is not None:
            self.load(native_source)
        self.model[-1] = ChromaticWaveletDetectHead(
            self.model[-1], self.cwcf_config
        )

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        head = self.model[-1] if hasattr(self, "model") and len(self.model) else None
        if isinstance(x, torch.Tensor) and isinstance(head, ChromaticWaveletDetectHead):
            head.set_cue(
                chromatic_wavelet_cue(x, clip=self.cwcf_config.cue_clip).detach()
            )
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=CWCFDetectionLoss)
        return CWCFDetectionLoss(self)


def build_cwcf_model(
    model_yaml: str,
    *,
    nc: int,
    source: nn.Module | None,
    seed: int,
    config: CWCFConfig | Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> CWCFDetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CWCFDetectionModel(
            model_yaml,
            ch=3,
            nc=nc,
            verbose=verbose,
            cwcf=config,
            native_source=source,
        )


def load_cwcf_weights(model: nn.Module, weights: Any) -> None:
    """Strictly restore a CWCF checkpoint produced by the same architecture."""

    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    if not isinstance(source, nn.Module):
        raise TypeError("Checkpoint CWCF tidak mengekspos model torch")
    result = model.load_state_dict(source.float().state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Restore CWCF tidak lengkap")
