"""Native YOLO26 loss plus IGEM geometry supervision for SGFR."""

from __future__ import annotations

import torch

from coffee_detector.igem.loss import multilevel_mask_loss

from .model import SGFRConfig, SGFRDetectHead


class SGFRDetectionLoss:
    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundSGFRDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                head = detector[-1]
                if not isinstance(head, SGFRDetectHead):
                    raise TypeError("SGFR loss memerlukan SGFRDetectHead")
                self.sgfr_head = head
                self.sgfr_config = SGFRConfig.from_mapping(head.config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                masks = preds.get("sgfr_mask_logits")
                if masks is not None:
                    auxiliary = multilevel_mask_loss(masks, batch, self.sgfr_head.nc)
                    loss[1] = loss[1] + float(self.sgfr_config.mask_loss_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundSGFRDetectionLoss()

