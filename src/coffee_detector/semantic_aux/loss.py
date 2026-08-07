from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import SemanticAuxConfig, SemanticAuxDetectHead, semantic_task_spec


class SemanticAuxDetectionLoss:
    """Native YOLO loss + CE on independent semantic heads at positive assignments."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundSemanticAuxLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, SemanticAuxDetectHead):
                    raise TypeError("Semantic aux loss memerlukan SemanticAuxDetectHead")
                self.semantic_config = SemanticAuxConfig.from_mapping(head.config)
                spec = semantic_task_spec(self.semantic_config.tasks)
                self.task_values = {task: spec[task]["values"] for task in self.semantic_config.tasks}
                self.task_mapping = {
                    task: torch.tensor(spec[task]["mapping"], dtype=torch.long)
                    for task in self.semantic_config.tasks
                }

            def get_assigned_targets_and_loss(self, preds, batch):
                semantic_logits = preds.pop("semantic_aux_logits", None)
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                if semantic_logits is None:
                    return assignments, loss, loss.detach()
                fg_mask, target_gt_idx = assignments[:2]
                if not bool(fg_mask.any()):
                    return assignments, loss, loss.detach()

                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
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
                assigned_leaf = gt_labels.gather(1, target_gt_idx.long())

                auxiliary = pred_scores.sum() * 0.0
                active = 0
                for task in self.semantic_config.tasks:
                    logits = semantic_logits[task]
                    if logits.shape[:2] != pred_scores.shape[:2]:
                        raise RuntimeError(f"Semantic logits {task} tidak sejajar dengan assignments")
                    mapping = self.task_mapping[task].to(self.device)
                    group_target = mapping[assigned_leaf]
                    valid = fg_mask & group_target.ge(0)
                    if not bool(valid.any()):
                        continue
                    auxiliary = auxiliary + F.cross_entropy(logits[valid], group_target[valid])
                    active += 1
                if active:
                    auxiliary = auxiliary / float(active)
                    loss[1] = loss[1] + float(self.semantic_config.auxiliary_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundSemanticAuxLoss()
