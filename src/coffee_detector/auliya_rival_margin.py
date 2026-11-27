"""Train-only strongest-rival hinge on assigned YOLO26-seg positives."""

from __future__ import annotations

import math

import torch
from ultralytics.nn.modules.head import Segment26
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.loss import E2ELoss, SegmentLoss26

RIVAL_ARM = "RGBMARGIN1"
RIVAL_PROTOCOL = "auliya-official-fold-a-rgbmargin1-v1"
RIVAL_WEIGHT = .10
RIVAL_MARGIN = math.log(2.)
RIVAL_CONTRACT = {
    "kind": "positive_anchor_strongest_wrong_class_logit_hinge",
    "relative_to_native_cls_gain": RIVAL_WEIGHT,
    "logit_margin": RIVAL_MARGIN,
    "branches": ["one2many", "one2one"],
}


def positive_rival_margin(scores: torch.Tensor, foreground: torch.Tensor,
                          target_gt_index: torch.Tensor, batch: dict) -> torch.Tensor:
    """Mean hinge on positives; ground-truth indices use native per-image padding."""
    # YOLO's preprocess() uses within-image offsets in the same order.
    image_index = batch["batch_idx"].reshape(-1).to(device=scores.device, dtype=torch.long)
    gt_classes = batch["cls"].reshape(-1).to(device=scores.device, dtype=torch.long)
    counts = torch.bincount(image_index, minlength=scores.shape[0])
    offsets = counts.cumsum(0) - counts
    labels = gt_classes[(target_gt_index.long() + offsets[:, None])[foreground]]
    logits = scores.transpose(1, 2)[foreground].float()
    true = logits.gather(1, labels[:, None]).squeeze(1)
    rivals = logits.scatter(1, labels[:, None], float("-inf")).amax(dim=1)
    return (RIVAL_MARGIN + rivals - true).relu().sum() / foreground.sum().clamp_min(1)


class RivalMarginSegmentLoss(SegmentLoss26):
    def get_assigned_targets_and_loss(self, preds, batch):
        assignment, loss, items = super().get_assigned_targets_and_loss(preds, batch)
        foreground, target_gt_index = assignment[:2]
        margin = positive_rival_margin(preds["scores"], foreground, target_gt_index, batch)
        cls = loss[1] + self.hyp.cls * RIVAL_WEIGHT * margin
        return assignment, torch.stack((loss[0], cls, loss[2])), {**items, "cls_loss": cls.detach()}


class RivalMarginSegmentationModel(SegmentationModel):
    """Same forward and state schema as the RGB baseline; swap only the criterion."""

    def __init__(self, cfg, *, ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        native = self.model[-1]
        if type(native) is not Segment26 or getattr(native, "one2one_cv2", None) is None:
            raise TypeError("Expected the native dual-branch YOLO26 segmentation head")
        self.frequency_arm = RIVAL_ARM

    def init_criterion(self):
        return E2ELoss(self, RivalMarginSegmentLoss)
