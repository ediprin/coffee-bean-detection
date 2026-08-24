from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import IGEMConfig, IGEMDetectHead


def rectangular_class_mask_targets(
    mask_logits: torch.Tensor,
    batch: dict,
    num_classes: int,
) -> torch.Tensor:
    """Rasterize train-time axis-aligned bbox masks into N+1 pixel targets.

    Original PCA-DB uses class-aware instance masks and aircraft-specific fine
    cross masks. Coffee transfer keeps only the coarse class-aware foreground
    principle and uses each training bbox rectangle. Background id is ``N``.

    If rectangles overlap, smaller boxes are painted last so small instances are
    not erased by larger regions. This deterministic overlap rule is a transfer
    choice, not a paper claim.
    """

    if mask_logits.ndim != 4 or mask_logits.shape[1] != num_classes + 1:
        raise ValueError("mask_logits harus [B,N+1,H,W]")
    batch_size, _channels, height, width = mask_logits.shape
    target = torch.full(
        (batch_size, height, width),
        int(num_classes),
        dtype=torch.long,
        device=mask_logits.device,
    )
    boxes = batch["bboxes"].to(mask_logits.device).reshape(-1, 4)
    labels = batch["cls"].to(mask_logits.device).reshape(-1).long()
    batch_index = batch["batch_idx"].to(mask_logits.device).reshape(-1).long()
    if not (len(boxes) == len(labels) == len(batch_index)):
        raise ValueError("bbox/cls/batch_idx tidak sejajar")
    if not len(boxes):
        return target

    areas = boxes[:, 2] * boxes[:, 3]
    # Large first, smaller later -> small object keeps its pixels in overlap.
    order = torch.argsort(areas, descending=True)
    for row in order.tolist():
        image_id = int(batch_index[row])
        class_id = int(labels[row])
        if image_id < 0 or image_id >= batch_size:
            raise ValueError("batch_idx di luar rentang")
        if class_id < 0 or class_id >= num_classes:
            raise ValueError("class id di luar rentang")
        xc, yc, bw, bh = (float(value) for value in boxes[row])
        x1 = max(0, min(width - 1, int(math.floor((xc - bw / 2.0) * width))))
        y1 = max(0, min(height - 1, int(math.floor((yc - bh / 2.0) * height))))
        x2 = max(x1 + 1, min(width, int(math.ceil((xc + bw / 2.0) * width))))
        y2 = max(y1 + 1, min(height, int(math.ceil((yc + bh / 2.0) * height))))
        target[image_id, y1:y2, x1:x2] = class_id
    return target


def multilevel_mask_loss(mask_logits: list[torch.Tensor], batch: dict, num_classes: int) -> torch.Tensor:
    if not mask_logits:
        raise ValueError("mask_logits kosong")
    losses = []
    for logits in mask_logits:
        target = rectangular_class_mask_targets(logits, batch, num_classes)
        losses.append(F.cross_entropy(logits, target, reduction="mean"))
    return torch.stack(losses).mean()


class IGEMDetectionLoss:
    """Native YOLO26 detection loss + coarse class-aware pixel CE."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundIGEMDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, IGEMDetectHead):
                    raise TypeError("IGEM loss memerlukan IGEMDetectHead")
                self.igem_head = head
                self.igem_config = IGEMConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                masks = preds.get("igem_mask_logits")
                if masks is None:
                    # One-to-one branch remains native detection loss.
                    return assignments, loss, loss.detach()
                auxiliary = multilevel_mask_loss(masks, batch, self.igem_head.nc)
                loss[1] = loss[1] + float(self.igem_config.mask_loss_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundIGEMDetectionLoss()
