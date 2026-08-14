from __future__ import annotations

import torch
import torch.nn.functional as F

from coffee_detector.fsce_cpe.loss import aligned_iou_xyxy
from coffee_detector.fsce_cpe.model import FSCECPEDetectHead
from .config import CircleCPEConfig


def circle_pair_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.25,
    gamma: float = 256.0,
) -> torch.Tensor:
    """Circle Loss over class-labelled normalized embedding pairs.

    Positive and negative pairs are formed once with i<j to avoid duplicate
    symmetric pairs. Self-paced alpha weights are detached, matching the Circle
    Loss formulation where pair similarities farther from their optima receive
    larger weights.
    """
    if embeddings.ndim != 2:
        raise ValueError("Circle embeddings harus [N,D]")
    labels = labels.to(device=embeddings.device, dtype=torch.long).reshape(-1)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Jumlah embedding dan label berbeda")
    if not 0.0 < margin < 1.0:
        raise ValueError("margin harus berada pada (0,1)")
    if gamma <= 0:
        raise ValueError("gamma harus positif")
    n = embeddings.shape[0]
    if n < 2:
        return embeddings.sum() * 0.0

    z = F.normalize(embeddings.float(), dim=1, eps=1e-8)
    similarity = z @ z.t()
    upper = torch.triu(torch.ones((n, n), device=z.device, dtype=torch.bool), diagonal=1)
    same = labels[:, None].eq(labels[None, :])
    sp = similarity[same & upper]
    sn = similarity[(~same) & upper]
    if sp.numel() == 0 or sn.numel() == 0:
        return embeddings.sum() * 0.0

    alpha_p = (-sp.detach() + 1.0 + margin).clamp_min(0.0)
    alpha_n = (sn.detach() + margin).clamp_min(0.0)
    delta_p = 1.0 - margin
    delta_n = margin
    logit_p = -float(gamma) * alpha_p * (sp - delta_p)
    logit_n = float(gamma) * alpha_n * (sn - delta_n)
    loss = F.softplus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0))
    return loss.to(dtype=embeddings.dtype)


class CircleCPEDetectionLoss:
    """Native Ultralytics detection loss plus training-only Circle-CPE."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundCircleCPEDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                if not isinstance(detector[-1], FSCECPEDetectHead):
                    raise TypeError("Circle-CPE memerlukan frozen FSCECPEDetectHead infrastructure")
                self.circle_config = CircleCPEConfig.from_mapping(model.circle_cpe_config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                embeddings = preds.get("cpe_embeddings")
                if embeddings is None:
                    return assignments, loss, loss.detach()

                fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assignments
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                if embeddings.shape[:2] != pred_scores.shape[:2]:
                    raise RuntimeError("Circle embeddings tidak sejajar dengan dense predictions")
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
                if self.circle_config.iou_threshold > 0.0:
                    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
                    pred_bboxes_img = self.bbox_decode(anchor_points, pred_distri).detach() * stride_tensor
                    fg_iou = aligned_iou_xyxy(
                        pred_bboxes_img[fg_mask].float(), target_bboxes[fg_mask].detach().float()
                    )
                    keep = fg_iou > float(self.circle_config.iou_threshold)
                    selected = torch.zeros_like(fg_mask)
                    selected[fg_mask] = keep

                if not bool(selected.any()):
                    return assignments, loss, loss.detach()

                auxiliary = circle_pair_loss(
                    embeddings[selected],
                    assigned_labels[selected],
                    margin=self.circle_config.margin,
                    gamma=self.circle_config.gamma,
                )
                loss[1] = loss[1] + float(self.circle_config.loss_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundCircleCPEDetectionLoss()
