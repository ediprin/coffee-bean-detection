from __future__ import annotations

import torch

from coffee_detector.dida_af2.loss import DIDABranchLoss, weak_to_strong_consistency

from .config import RAFCConfig


class RAFCE2ELoss:
    """Native raw detection plus classification-only Fourier consistency."""

    def __init__(self, model: torch.nn.Module, config: RAFCConfig) -> None:
        self.config = config
        self.one2many = DIDABranchLoss(model, tal_topk=10)
        self.one2one = DIDABranchLoss(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = 0.2
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1
        self.last_stats: dict[str, float | int] = {}

    def _branch(self, criterion, raw_preds, fourier_preds, batch):
        raw_loss = criterion.loss(raw_preds, batch)
        raw_gt = criterion.last_gt_logits
        # This call supplies GT assignment and logits for the transformed view.
        # Its native box/class/DFL loss is deliberately excluded from total.
        criterion.loss(fourier_preds, batch)
        fourier_gt = criterion.last_gt_logits
        if raw_gt is None or fourier_gt is None:
            raise RuntimeError("Capture GT logits RAFC tidak tersedia")
        consistency, matched = weak_to_strong_consistency(
            raw_gt, fourier_gt, temperature=self.config.temperature
        )
        return raw_loss, consistency, matched

    def __call__(self, preds, batch):
        # Native validation path: decoded tensor + ordinary raw branch mapping.
        if (
            isinstance(preds, tuple)
            and len(preds) == 2
            and isinstance(preds[0], torch.Tensor)
            and isinstance(preds[1], dict)
        ):
            raw = preds[1]
            many = self.one2many.loss(raw["one2many"], batch)
            one = self.one2one.loss(raw["one2one"], batch)
            return self.o2m * many[0] + self.o2o * one[0], one[1]
        if (
            not isinstance(preds, tuple)
            or len(preds) != 2
            or not all(isinstance(view, dict) for view in preds)
        ):
            raise TypeError("RAFC memerlukan pasangan prediction raw/Fourier saat training")
        raw, fourier = preds
        raw = self.one2many.parse_output(raw)
        fourier = self.one2many.parse_output(fourier)
        many = self._branch(
            self.one2many, raw["one2many"], fourier["one2many"], batch
        )
        one = self._branch(
            self.one2one, raw["one2one"], fourier["one2one"], batch
        )
        native = self.o2m * many[0][0] + self.o2o * one[0][0]
        consistency = self.o2m * many[1] + self.o2o * one[1]
        batch_size = raw["one2one"]["boxes"].shape[0]
        total = native.clone()
        total[1] = (
            total[1]
            + float(self.config.consistency_gain) * consistency * batch_size
        )
        items = one[0][1].clone()
        items[1] = items[1] + float(self.config.consistency_gain) * consistency.detach()
        self.last_stats = {
            "consistency_loss": float(consistency.detach()),
            "matched_o2m_gt": int(many[2]),
            "matched_o2o_gt": int(one[2]),
            "o2m_weight": float(self.o2m),
            "o2o_weight": float(self.o2o),
        }
        return total, items

    def update(self) -> None:
        self.updates += 1
        self.o2m = max(
            1 - self.updates / max(self.one2one.hyp.epochs - 1, 1), 0
        ) * (self.o2m_copy - self.final_o2m) + self.final_o2m
        self.o2o = max(self.total - self.o2m, 0)
