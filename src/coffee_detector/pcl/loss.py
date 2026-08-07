from __future__ import annotations

import torch

from .model import PCLConfig, PCLDetectHead


class PCLDetectionLoss:
    """Native Ultralytics detection loss plus training-only PCLDet ProtoCL."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundPCLDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, PCLDetectHead):
                    raise TypeError("PCL loss memerlukan PCLDetectHead")
                self.pcl_head = head
                self.pcl_config = PCLConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                embeddings = preds.get("pcl_embeddings")
                if embeddings is None:
                    # One-to-one companion branch remains native YOLO loss.
                    return assignments, loss, loss.detach()

                fg_mask, target_gt_idx = assignments[:2]
                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                if embeddings.shape[:2] != pred_scores.shape[:2]:
                    raise RuntimeError(
                        "Urutan/dimensi PCL embedding tidak sejajar dengan dense predictions"
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
                auxiliary = self.pcl_head.pcl.prototype_contrast.loss(
                    embeddings[fg_mask], assigned_labels[fg_mask]
                )
                loss[1] = loss[1] + float(self.pcl_config.loss_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundPCLDetectionLoss()
