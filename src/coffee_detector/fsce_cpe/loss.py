from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import FSCECPEConfig, FSCECPEDetectHead


def aligned_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if box1.shape != box2.shape or box1.ndim != 2 or box1.shape[-1] != 4:
        raise ValueError("aligned_iou_xyxy memerlukan dua tensor [N,4] dengan shape sama")
    lt = torch.maximum(box1[:, :2], box2[:, :2])
    rb = torch.minimum(box1[:, 2:], box2[:, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box1[:, 2] - box1[:, 0]).clamp_min(0) * (box1[:, 3] - box1[:, 1]).clamp_min(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp_min(0) * (box2[:, 3] - box2[:, 1]).clamp_min(0)
    return inter / (area1 + area2 - inter).clamp_min(eps)


def cpe_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """FSCE CPE Eq. (2)-(3) after proposal-consistency filtering."""
    if embeddings.ndim != 2:
        raise ValueError("CPE embeddings harus [N,D]")
    labels = labels.to(device=embeddings.device, dtype=torch.long).reshape(-1)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Jumlah CPE embedding dan label berbeda")
    if temperature <= 0:
        raise ValueError("temperature harus positif")
    n = embeddings.shape[0]
    if n < 2:
        return embeddings.sum() * 0.0

    z = F.normalize(embeddings.float(), dim=1, eps=1e-8)
    logits = (z @ z.t()) / float(temperature)
    eye = torch.eye(n, device=embeddings.device, dtype=torch.bool)
    log_den = torch.logsumexp(logits.masked_fill(eye, -torch.inf), dim=1, keepdim=True)
    log_prob = (logits - log_den).masked_fill(eye, 0.0)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    counts = positives.sum(dim=1)
    per_anchor = torch.zeros(n, device=embeddings.device, dtype=log_prob.dtype)
    valid = counts > 0
    if bool(valid.any()):
        summed = (log_prob * positives.to(log_prob.dtype)).sum(dim=1)
        per_anchor[valid] = -summed[valid] / counts[valid].to(log_prob.dtype)
    return per_anchor.mean().to(dtype=embeddings.dtype)


class FSCECPEDetectionLoss:
    """Native Ultralytics loss plus training-only FSCE-style CPE."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundFSCECPEDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, FSCECPEDetectHead):
                    raise TypeError("FSCE-CPE loss memerlukan FSCECPEDetectHead")
                self.cpe_config = FSCECPEConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                embeddings = preds.get("cpe_embeddings")
                if embeddings is None:
                    return assignments, loss, loss.detach()

                fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assignments
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                if embeddings.shape[:2] != pred_scores.shape[:2]:
                    raise RuntimeError("Urutan CPE embeddings tidak sejajar dengan dense predictions")
                if not bool(fg_mask.any()):
                    return assignments, loss, loss.detach()

                batch_size = pred_scores.shape[0]
                image_size = (
                    torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_scores.dtype)
                    * self.stride[0]
                )
                targets = torch.cat(
                    (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
                )
                targets = self.preprocess(
                    targets.to(self.device), batch_size, scale_tensor=image_size[[1, 0, 1, 0]]
                )
                gt_labels = targets[..., 0].long()
                assigned_labels = gt_labels.gather(1, target_gt_idx.long())

                selected = fg_mask.clone()
                if self.cpe_config.iou_threshold > 0.0:
                    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
                    pred_bboxes_img = self.bbox_decode(anchor_points, pred_distri).detach() * stride_tensor
                    fg_iou = aligned_iou_xyxy(
                        pred_bboxes_img[fg_mask].float(), target_bboxes[fg_mask].detach().float()
                    )
                    keep = fg_iou > float(self.cpe_config.iou_threshold)
                    selected = torch.zeros_like(fg_mask)
                    selected[fg_mask] = keep

                if not bool(selected.any()):
                    return assignments, loss, loss.detach()

                auxiliary = cpe_supervised_contrastive_loss(
                    embeddings[selected],
                    assigned_labels[selected],
                    temperature=self.cpe_config.temperature,
                )
                loss[1] = loss[1] + float(self.cpe_config.loss_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundFSCECPEDetectionLoss()
