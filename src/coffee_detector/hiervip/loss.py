from __future__ import annotations

import torch

from .model import HierVIPConfig, HierVIPDetectHead


class HierVIPDetectionLoss:
    """Native YOLO26 detection loss plus training-only HierVIP HSC on positives."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundHierVIPDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, HierVIPDetectHead):
                    raise TypeError("HierVIP loss memerlukan HierVIPDetectHead")
                self.hiervip_head = head
                self.hiervip_config = HierVIPConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                embeddings = preds.get("hiervip_embeddings")
                if embeddings is None:
                    return assignments, loss, loss.detach()

                fg_mask, target_gt_idx = assignments[:2]
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                if embeddings.shape[:2] != pred_scores.shape[:2]:
                    raise RuntimeError("HierVIP embeddings tidak sejajar dengan dense predictions")
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
                auxiliary = self.hiervip_head.hiervip.prototype_tree.update_and_loss(
                    embeddings[fg_mask], assigned_labels[fg_mask]
                )
                loss[1] = loss[1] + float(self.hiervip_config.loss_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundHierVIPDetectionLoss()
