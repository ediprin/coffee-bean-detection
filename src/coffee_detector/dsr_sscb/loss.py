from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import SSCBConfig, SSCBDetectHead


def rasterize_bbox_foreground(
    batch_idx: torch.Tensor,
    bboxes_xywh: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rasterize normalized xywh GT boxes to a union foreground mask.

    This is an explicit coffee-transfer supervisor replacing DSRDet's CLIP-
    attention semantic label generator. It must not be described as the paper's
    original shared-semantic target.
    """
    target = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
    if bboxes_xywh.numel() == 0:
        return target
    boxes = bboxes_xywh.detach().to(device=device, dtype=torch.float32)
    indices = batch_idx.detach().to(device=device, dtype=torch.long).reshape(-1)
    for index in range(boxes.shape[0]):
        b = int(indices[index])
        cx, cy, bw, bh = boxes[index].tolist()
        x1 = max(0, min(width - 1, int((cx - bw / 2.0) * width)))
        y1 = max(0, min(height - 1, int((cy - bh / 2.0) * height)))
        x2 = max(x1 + 1, min(width, int(torch.ceil(torch.tensor((cx + bw / 2.0) * width)).item())))
        y2 = max(y1 + 1, min(height, int(torch.ceil(torch.tensor((cy + bh / 2.0) * height)).item())))
        target[b, 0, y1:y2, x1:x2] = 1.0
    return target


def semantic_foreground_loss(
    semantic_logits: list[torch.Tensor], batch: dict, *, weight_balance: bool = True
) -> torch.Tensor:
    if not semantic_logits:
        raise ValueError("semantic_logits kosong")
    losses = []
    batch_size = semantic_logits[0].shape[0]
    for logits in semantic_logits:
        target = rasterize_bbox_foreground(
            batch["batch_idx"],
            batch["bboxes"],
            batch_size=batch_size,
            height=logits.shape[-2],
            width=logits.shape[-1],
            device=logits.device,
            dtype=logits.dtype,
        )
        if weight_balance:
            positives = target.sum().float()
            negatives = float(target.numel()) - positives
            pos_weight = (negatives / positives.clamp_min(1.0)).clamp(max=20.0)
            current = F.binary_cross_entropy_with_logits(
                logits.float(), target.float(), pos_weight=pos_weight, reduction="mean"
            )
        else:
            current = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="mean")
        losses.append(current)
    return torch.stack(losses).mean().to(dtype=semantic_logits[0].dtype)


class SSCBDetectionLoss:
    """Native detection loss plus bbox-supervised shared-foreground auxiliary."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundSSCBDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, SSCBDetectHead):
                    raise TypeError("SSCB loss memerlukan SSCBDetectHead")
                self.sscb_config = SSCBConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                semantic_logits = preds.get("sscb_semantic_logits")
                if not self.sscb_config.uses_semantics:
                    if semantic_logits is not None:
                        raise RuntimeError("M0 tidak boleh menghasilkan semantic auxiliary")
                    return assignments, loss, loss.detach()
                if semantic_logits is None:
                    raise RuntimeError("Semantic arm aktif tetapi semantic logits hilang")
                auxiliary = semantic_foreground_loss(semantic_logits, batch)
                loss[1] = loss[1] + float(self.sscb_config.semantic_aux_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundSSCBDetectionLoss()
