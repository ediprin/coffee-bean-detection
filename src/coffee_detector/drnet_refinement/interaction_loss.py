from __future__ import annotations

import torch
import torch.nn.functional as F

from .interaction import DRNetInteractionConfig, DRNetInteractionDetectHead


class DRNetInteractionDetectionLoss:
    """Native YOLO detection loss plus positive-assignment coarse supervision.

    Official DRNet trains its bbox-head classifier on converted coarse labels
    and its fine classifier on the original fine labels. YOLO26 has no explicit
    proposal background class, so this transfer supervises the coarse branch on
    positively assigned one-to-many dense samples while native YOLO loss keeps
    handling foreground/background classification and localization.
    """

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundDRNetInteractionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, DRNetInteractionDetectHead):
                    raise TypeError("Interaction loss memerlukan DRNetInteractionDetectHead")
                self.dr_head = head
                self.dr_config = DRNetInteractionConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                coarse_logits = preds.get("dr_coarse_logits")
                if coarse_logits is None:
                    # One-to-one companion branch remains native.
                    return assignments, loss, loss.detach()

                fg_mask, target_gt_idx = assignments[:2]
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                if coarse_logits.shape[:2] != pred_scores.shape[:2]:
                    raise RuntimeError("Coarse logits tidak sejajar dengan dense predictions")
                if not bool(fg_mask.any()):
                    return assignments, loss, loss.detach()

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
                assigned_fine = gt_labels.gather(1, target_gt_idx.long())
                mapping = self.dr_head.class_to_group.to(self.device)
                assigned_coarse = mapping[assigned_fine]
                coarse_loss = F.cross_entropy(
                    coarse_logits[fg_mask], assigned_coarse[fg_mask], reduction="mean"
                )
                loss[1] = loss[1] + float(self.dr_config.coarse_loss_weight) * coarse_loss
                return assignments, loss, loss.detach()

        return _BoundDRNetInteractionLoss()
