from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .config import DIDAAF2Config


@dataclass
class GTLogits:
    keys: torch.Tensor
    labels: torch.Tensor
    logits: torch.Tensor


def aggregate_positive_logits(
    scores: torch.Tensor,
    foreground: torch.Tensor,
    target_gt_index: torch.Tensor,
    labels: torch.Tensor,
) -> GTLogits:
    """Average native positive-anchor logits per (image, padded GT index)."""

    if scores.ndim != 3 or foreground.shape != scores.shape[:2]:
        raise ValueError("Shape scores/foreground tidak sejajar")
    if target_gt_index.shape != foreground.shape or labels.shape != foreground.shape:
        raise ValueError("Shape GT index/labels tidak sejajar")
    device = scores.device
    records: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
    for image_index in range(scores.shape[0]):
        positive = foreground[image_index]
        if not bool(positive.any()):
            continue
        gt_indices = target_gt_index[image_index, positive].long()
        image_scores = scores[image_index, positive]
        image_labels = labels[image_index, positive].long()
        for gt_index in torch.unique(gt_indices, sorted=True):
            keep = gt_indices.eq(gt_index)
            gt_labels = image_labels[keep]
            if not bool(gt_labels.eq(gt_labels[0]).all()):
                raise RuntimeError("Satu GT memperoleh label assignment yang tidak konsisten")
            records.append(
                (image_index, int(gt_index), gt_labels[0], image_scores[keep].mean(dim=0))
            )
    if not records:
        return GTLogits(
            keys=torch.empty((0, 2), device=device, dtype=torch.long),
            labels=torch.empty((0,), device=device, dtype=torch.long),
            logits=scores.new_empty((0, scores.shape[-1])),
        )
    return GTLogits(
        keys=torch.tensor([[row[0], row[1]] for row in records], device=device),
        labels=torch.stack([row[2] for row in records]).long(),
        logits=torch.stack([row[3] for row in records]),
    )


def smooth_topk_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
    topk: int,
) -> torch.Tensor:
    """GT-balanced smooth separation from the strongest dynamic class rivals."""

    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("Logits/labels FG harus [N,C] dan [N]")
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    if logits.shape[1] < 2:
        raise ValueError("FG margin memerlukan setidaknya dua kelas")
    labels = labels.to(device=logits.device, dtype=torch.long)
    true_logits = logits.gather(1, labels[:, None])
    rivals = logits.masked_fill(
        F.one_hot(labels, num_classes=logits.shape[1]).bool(), float("-inf")
    )
    strongest = rivals.topk(min(int(topk), logits.shape[1] - 1), dim=1).values
    log_rival_sum = torch.logsumexp(strongest - true_logits + float(margin), dim=1)
    return F.softplus(log_rival_sum).mean()


def match_gt_logits(weak: GTLogits, strong: GTLogits) -> tuple[torch.Tensor, torch.Tensor]:
    """Match two branch-specific assignment sets by (image, GT), never anchor index."""

    if weak.keys.numel() == 0 or strong.keys.numel() == 0:
        width = weak.logits.shape[-1] if weak.logits.ndim == 2 else strong.logits.shape[-1]
        empty = weak.logits.new_empty((0, width))
        return empty, empty
    strong_lookup = {
        (int(key[0]), int(key[1])): index for index, key in enumerate(strong.keys.detach().cpu())
    }
    weak_indices: list[int] = []
    strong_indices: list[int] = []
    for index, key in enumerate(weak.keys.detach().cpu()):
        other = strong_lookup.get((int(key[0]), int(key[1])))
        if other is not None:
            if int(weak.labels[index]) != int(strong.labels[other]):
                raise RuntimeError("Label GT berbeda di antara dua view")
            weak_indices.append(index)
            strong_indices.append(other)
    if not weak_indices:
        empty = weak.logits.new_empty((0, weak.logits.shape[-1]))
        return empty, empty
    return weak.logits[weak_indices], strong.logits[strong_indices]


def weak_to_strong_consistency(
    weak: GTLogits, strong: GTLogits, *, temperature: float
) -> tuple[torch.Tensor, int]:
    weak_logits, strong_logits = match_gt_logits(weak, strong)
    if weak_logits.shape[0] == 0:
        return strong.logits.sum() * 0.0, 0
    teacher = F.softmax(weak_logits.detach() / float(temperature), dim=-1)
    student = F.log_softmax(strong_logits / float(temperature), dim=-1)
    loss = F.kl_div(student, teacher, reduction="batchmean") * float(temperature) ** 2
    return loss, int(weak_logits.shape[0])


class DIDABranchLoss:
    """Factory around native YOLO assignment/loss with GT-level logit capture."""

    def __new__(cls, model: torch.nn.Module, **kwargs: Any):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundDIDABranchLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, **kwargs)
                self.last_gt_logits: GTLogits | None = None

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                foreground, target_gt_index = assignments[:2]
                scores = preds["scores"].permute(0, 2, 1).contiguous()
                batch_size = scores.shape[0]
                image_size = (
                    torch.tensor(
                        preds["feats"][0].shape[2:],
                        device=self.device,
                        dtype=scores.dtype,
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
                padded = self.preprocess(
                    targets.to(self.device),
                    batch_size,
                    scale_tensor=image_size[[1, 0, 1, 0]],
                )
                gt_labels = padded[..., 0].long()
                assigned_labels = gt_labels.gather(1, target_gt_index.long())
                self.last_gt_logits = aggregate_positive_logits(
                    scores, foreground, target_gt_index, assigned_labels
                )
                return assignments, loss, loss.detach()

        return _BoundDIDABranchLoss()


class DIDAE2ELoss:
    """Paired AF2 objective with native E2E weighting and training-only auxiliaries."""

    def __init__(self, model: torch.nn.Module, config: DIDAAF2Config) -> None:
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

    @staticmethod
    def _branch(
        criterion,
        weak_preds,
        strong_preds,
        batch,
        config: DIDAAF2Config,
    ):
        weak_loss = criterion.loss(weak_preds, batch)
        weak_gt = criterion.last_gt_logits
        strong_loss = criterion.loss(strong_preds, batch)
        strong_gt = criterion.last_gt_logits
        if weak_gt is None or strong_gt is None:
            raise RuntimeError("Capture GT logits DIDA tidak tersedia")
        native = 0.5 * (weak_loss[0] + strong_loss[0])
        dg, matched = weak_to_strong_consistency(
            weak_gt, strong_gt, temperature=config.temperature
        )
        fg_weak = smooth_topk_margin_loss(
            weak_gt.logits,
            weak_gt.labels,
            margin=config.margin,
            topk=config.topk,
        )
        fg_strong = smooth_topk_margin_loss(
            strong_gt.logits,
            strong_gt.labels,
            margin=config.margin,
            topk=config.topk,
        )
        fg = 0.5 * (fg_weak + fg_strong)
        return native, dg, fg, matched, weak_loss[1], strong_loss[1]

    def __call__(self, preds, batch):
        if not isinstance(preds, tuple) or len(preds) != 2:
            raise TypeError("DIDA-AF2 memerlukan pasangan prediksi weak/strong")
        weak = self.one2many.parse_output(preds[0])
        strong = self.one2many.parse_output(preds[1])
        many = self._branch(
            self.one2many,
            weak["one2many"],
            strong["one2many"],
            batch,
            self.config,
        )
        one = self._branch(
            self.one2one,
            weak["one2one"],
            strong["one2one"],
            batch,
            self.config,
        )
        native = self.o2m * many[0] + self.o2o * one[0]
        dg = self.o2m * many[1] + self.o2o * one[1]
        fg = self.o2m * many[2] + self.o2o * one[2]
        batch_size = weak["one2one"]["boxes"].shape[0]
        total = native.clone()
        if self.config.dg_enabled:
            total[1] = total[1] + float(self.config.dg_gain) * dg * batch_size
        if self.config.fg_enabled:
            total[1] = total[1] + float(self.config.fg_gain) * fg * batch_size
        items = 0.5 * (one[4] + one[5])
        if self.config.dg_enabled:
            items[1] = items[1] + float(self.config.dg_gain) * dg.detach()
        if self.config.fg_enabled:
            items[1] = items[1] + float(self.config.fg_gain) * fg.detach()
        self.last_stats = {
            "dg_loss": float(dg.detach()),
            "fg_loss": float(fg.detach()),
            "matched_o2m_gt": int(many[3]),
            "matched_o2o_gt": int(one[3]),
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
