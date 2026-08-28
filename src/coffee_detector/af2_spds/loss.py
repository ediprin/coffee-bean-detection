from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import AF2SPDSConfig


def multilevel_reconstruction_loss(
    predictions: list[torch.Tensor], target: torch.Tensor
) -> torch.Tensor:
    """Mean Smooth-L1 reconstruction over P3/P4/P5 resolutions."""

    if not predictions:
        raise ValueError("Prediksi auxiliary kosong")
    if target.ndim != 4 or target.shape[1] != 3:
        raise ValueError("Target auxiliary harus BCHW RGB 3-channel")
    losses = []
    for prediction in predictions:
        if prediction.ndim != 4 or prediction.shape[:2] != target.shape[:2]:
            raise ValueError("Prediksi auxiliary tidak kompatibel dengan target")
        resized = F.interpolate(
            target.float(), size=prediction.shape[-2:], mode="area"
        ).to(dtype=prediction.dtype)
        losses.append(F.smooth_l1_loss(prediction, resized, reduction="mean"))
    return torch.stack(losses).mean()


class AF2SPDSDetectionLoss:
    """Native YOLO loss plus a non-invasive multilevel reconstruction loss."""

    def __new__(
        cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundAF2SPDSDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                head = model.model[-1]
                self.head = head
                self.spds = AF2SPDSConfig.from_mapping(model.af2_spds_config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                predictions = self.head.last_auxiliary_predictions
                if predictions is None:
                    if model.training:
                        raise RuntimeError("Auxiliary predictions tidak tersedia saat training")
                    # Ultralytics computes validation loss from eval-mode raw
                    # predictions. Auxiliary decoders are intentionally absent
                    # in eval/inference, so validation reports native detection
                    # loss only and never changes checkpoint selection metrics.
                    return assignments, loss, loss.detach()

                if self.spds.target == "none":
                    # Same decoder capacity and compute as treatment arms, but no
                    # learning signal: a matched deep-supervision control.
                    auxiliary = torch.stack([value.mean() * 0.0 for value in predictions]).sum()
                else:
                    targets = model.last_auxiliary_targets
                    if targets is None or self.spds.target not in targets:
                        raise RuntimeError("Target auxiliary tidak tersedia")
                    auxiliary = multilevel_reconstruction_loss(
                        predictions, targets[self.spds.target]
                    )
                loss[1] = loss[1] + float(self.spds.auxiliary_gain) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundAF2SPDSDetectionLoss()
