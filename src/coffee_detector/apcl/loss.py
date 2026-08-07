from __future__ import annotations

import torch

from .model import APCLConfig, APCLDetectHead


class APCLDetectionLoss:
    """Native Ultralytics detection loss plus training-only APCL on positives."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundAPCLDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, APCLDetectHead):
                    raise TypeError("APCL loss memerlukan APCLDetectHead")
                self.apcl_head = head
                self.apcl_config = APCLConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                embeddings = preds.get("apcl_embeddings")
                if embeddings is None:
                    # One-to-one branch deliberately remains pure native loss.
                    return assignments, loss, loss.detach()

                fg_mask, target_gt_idx = assignments[:2]
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                if embeddings.shape[:2] != pred_scores.shape[:2]:
                    raise RuntimeError(
                        "Urutan/dimensi APCL embedding tidak sejajar dengan dense predictions"
                    )
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
                assigned_labels = gt_labels.gather(1, target_gt_idx.long())
                auxiliary = self.apcl_head.apcl.prototype_contrast.update_and_loss(
                    embeddings[fg_mask], assigned_labels[fg_mask]
                )
                loss[1] = loss[1] + float(self.apcl_config.loss_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundAPCLDetectionLoss()
