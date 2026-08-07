from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import DRNetRefinementConfig, DRNetRefinementDetectHead


def confusion_minimized_positive_loss(
    fine_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda1: float,
    lambda2: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Positive-assignment transfer of DRNet Algorithm 1 and Eqs. (5)-(6).

    DRNet defines CML on two-stage proposals including an explicit background
    subclass. YOLO26 has no background logit. Therefore this transfer applies
    the paper's separability-dependent weighting only to positively assigned
    dense samples; native YOLO classification loss continues to handle all
    positive/negative anchors. This is intentionally not claimed as a literal
    reproduction of Eq. (6).
    """
    if fine_logits.ndim != 2:
        raise ValueError("fine_logits harus [N,C]")
    labels = labels.to(device=fine_logits.device, dtype=torch.long).reshape(-1)
    if labels.shape[0] != fine_logits.shape[0]:
        raise ValueError("Jumlah fine logits dan label tidak sama")
    if not len(labels):
        zero = fine_logits.sum() * 0.0
        return zero, {"mean_separability": zero.detach(), "mean_weight": zero.detach()}

    probability = fine_logits.detach().sigmoid()
    row = torch.arange(len(labels), device=fine_logits.device)
    alpha = probability[row, labels]
    competitors = probability.clone()
    competitors[row, labels] = float("-inf")
    competitor = competitors.max(dim=1).values
    separability = (alpha - competitor).clamp(min=-1.0 + 1e-6, max=1.0)

    weight = torch.ones_like(separability)
    easy = separability > float(lambda1)
    wrong = separability < 0.0
    weight[easy] = torch.exp(-10.0 * (separability[easy] - float(lambda1)))
    weight[wrong] = -float(lambda2) * torch.log(separability[wrong] + 1.0) + 1.0

    targets = F.one_hot(labels, num_classes=fine_logits.shape[1]).to(fine_logits.dtype)
    per_class = F.binary_cross_entropy_with_logits(fine_logits, targets, reduction="none")
    per_sample = per_class.mean(dim=1)
    loss = (weight * per_sample).mean()
    details = {
        "mean_separability": separability.mean().detach(),
        "mean_weight": weight.mean().detach(),
        "hard_fraction": (separability < 0).float().mean().detach(),
    }
    return loss, details


class DRNetDetectionLoss:
    """Native Ultralytics detection loss plus optional adapted CML."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundDRNetDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, DRNetRefinementDetectHead):
                    raise TypeError("DRNet loss memerlukan DRNetRefinementDetectHead")
                self.dr_head = head
                self.dr_config = DRNetRefinementConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                fine_logits = preds.get("dr_fine_logits")
                if not self.dr_config.use_cml or fine_logits is None:
                    return assignments, loss, loss.detach()

                fg_mask, target_gt_idx = assignments[:2]
                if not bool(fg_mask.any()):
                    return assignments, loss, loss.detach()
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                if fine_logits.shape != pred_scores.shape:
                    raise RuntimeError("DRNet fine logits tidak sejajar dengan dense predictions")

                batch_size = pred_scores.shape[0]
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
                    1,
                )
                targets = self.preprocess(
                    targets.to(self.device),
                    batch_size,
                    scale_tensor=image_size[[1, 0, 1, 0]],
                )
                gt_labels = targets[..., 0].long()
                assigned_labels = gt_labels.gather(1, target_gt_idx.long())
                cml, _ = confusion_minimized_positive_loss(
                    fine_logits[fg_mask],
                    assigned_labels[fg_mask],
                    lambda1=self.dr_config.cml_lambda1,
                    lambda2=self.dr_config.cml_lambda2,
                )
                loss[1] = loss[1] + float(self.dr_config.cml_weight) * cml
                return assignments, loss, loss.detach()

        return _BoundDRNetDetectionLoss()
