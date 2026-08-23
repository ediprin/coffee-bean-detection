from __future__ import annotations

from coffee_detector.igem.loss import multilevel_mask_loss


class AF2ParentResidualDetectionLoss:
    """Native YOLO loss plus IGEM's frozen 0.05 auxiliary mask objective."""

    def __new__(cls, model, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                detector = getattr(model, "model", model)
                self.head = detector[-1]

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                masks = preds.get("parent_residual_mask_logits")
                if masks is not None:
                    auxiliary = multilevel_mask_loss(masks, batch, self.head.nc)
                    loss[1] = loss[1] + float(self.head.config.igem.mask_loss_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundLoss()
