from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.data.prepare_defectoscafeverde_grouped import NAMES
from coffee_detector.j25_cwcf.config import CWCFConfig
from coffee_detector.j25_cwcf.model import (
    ChromaticWaveletDetectHead,
    _aggregate_assigned_attributes,
    _first_conv_channels,
)
from coffee_detector.j25_cwcf.operator import chromatic_wavelet_cue


# The three shared families are taken directly from the paper's reported
# confusion analysis.  Atomic factors keep the twelve expert labels uniquely
# identifiable without pretending that the Spanish ontology is J25/SNI.
DEFECTOS_ATTRIBUTE_NAMES = (
    "sour_frozen_normal_color_family",
    "black_broca_dark_family",
    "broken_ear_peaberry_shape_family",
    "sour",
    "broca",
    "round_peaberry",
    "shell",
    "large_elephant",
    "pale_frozen",
    "black",
    "normal",
    "ear",
    "broken",
    "dry",
    "triangle",
)


def build_defectos_attribute_matrix() -> torch.Tensor:
    """Return the paper-grounded 12-class semantic factorization."""

    groups = {
        "sour_frozen_normal_color_family": {0, 5, 7},
        "black_broca_dark_family": {1, 6},
        "broken_ear_peaberry_shape_family": {2, 8, 9},
        "sour": {0},
        "broca": {1},
        "round_peaberry": {2},
        "shell": {3},
        "large_elephant": {4},
        "pale_frozen": {5},
        "black": {6},
        "normal": {7},
        "ear": {8},
        "broken": {9},
        "dry": {10},
        "triangle": {11},
    }
    matrix = torch.zeros(len(NAMES), len(DEFECTOS_ATTRIBUTE_NAMES))
    for attribute_index, name in enumerate(DEFECTOS_ATTRIBUTE_NAMES):
        for class_index in groups[name]:
            matrix[class_index, attribute_index] = 1.0
    return matrix


class DefectosChromaticWaveletDetectHead(ChromaticWaveletDetectHead):
    """CWCF score path with Defectos-specific training attributes."""

    def __init__(self, base_head: nn.Module, config: CWCFConfig) -> None:
        super().__init__(base_head, config)
        channels = tuple(_first_conv_channels(branch) for branch in base_head.cv2)
        self.attribute_count = len(DEFECTOS_ATTRIBUTE_NAMES)
        self.attribute_heads = nn.ModuleList(
            [nn.Conv2d(channel, self.attribute_count, 1) for channel in channels]
        )


class DefectosCWCFDetectionLoss:
    """Native detection objective plus paper-grounded object attributes."""

    def __new__(cls, model: nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundDefectosCWCFDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                self.head = model.model[-1]
                self.config = CWCFConfig.from_mapping(model.cwcf_config)
                self.apply_auxiliary = tal_topk2 is None
                self.attribute_matrix = build_defectos_attribute_matrix()

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
                    (
                        batch["batch_idx"].view(-1, 1),
                        batch["cls"].view(-1, 1),
                        batch["bboxes"],
                    ),
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

        return _BoundDefectosCWCFDetectionLoss()


from ultralytics.nn.tasks import DetectionModel


class DefectosCWCFDetectionModel(DetectionModel):
    """YOLO detector with RGB-native boxes and Defectos CWCF scores."""

    def __init__(
        self,
        cfg: str | dict,
        *,
        ch: int = 3,
        nc: int = 12,
        verbose: bool = False,
        cwcf: CWCFConfig | Mapping[str, Any] | None = None,
        native_source: nn.Module | None = None,
    ) -> None:
        self.cwcf_config = CWCFConfig.from_mapping(cwcf)
        if self.cwcf_config.explicit_composition:
            raise ValueError("Defectos DCWCF1 tidak memakai explicit composition")
        if self.cwcf_config.conditional_confusion_gain != 0.0:
            raise ValueError("Defectos DCWCF1 tidak memakai conditional J25 loss")
        self.last_attribute_loss: torch.Tensor | None = None
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        if native_source is not None:
            self.load(native_source)
        self.model[-1] = DefectosChromaticWaveletDetectHead(
            self.model[-1], self.cwcf_config
        )

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        head = self.model[-1] if hasattr(self, "model") and len(self.model) else None
        if isinstance(x, torch.Tensor) and isinstance(
            head, DefectosChromaticWaveletDetectHead
        ):
            head.set_cue(
                chromatic_wavelet_cue(x, clip=self.cwcf_config.cue_clip).detach()
            )
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=DefectosCWCFDetectionLoss)
        return DefectosCWCFDetectionLoss(self)


def build_defectos_cwcf_model(
    model_yaml: str,
    *,
    nc: int,
    source: nn.Module | None,
    seed: int,
    config: CWCFConfig | Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> DefectosCWCFDetectionModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return DefectosCWCFDetectionModel(
            model_yaml,
            ch=3,
            nc=nc,
            verbose=verbose,
            cwcf=config,
            native_source=source,
        )


def load_defectos_cwcf_weights(model: nn.Module, weights: Any) -> None:
    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    if not isinstance(source, nn.Module):
        raise TypeError("Checkpoint DCWCF1 tidak mengekspos model torch")
    result = model.load_state_dict(source.float().state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Restore DCWCF1 tidak lengkap")
