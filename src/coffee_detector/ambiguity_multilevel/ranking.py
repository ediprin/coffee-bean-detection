from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .model import (
    AmbiguityMultilevelConfig,
    AmbiguityMultilevelDetectionModel,
    load_ambiguity_multilevel_detector_weights,
)


@dataclass(frozen=True)
class HardCompetitorRankingConfig:
    """Training-only ranking term for the one-to-one classification branch."""

    weight: float = 0.25

    @classmethod
    def from_mapping(
        cls, payload: "HardCompetitorRankingConfig" | dict[str, Any] | None
    ) -> "HardCompetitorRankingConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if not 0.0 < result.weight <= 1.0:
            raise ValueError("hard_competitor_ranking.weight harus berada pada (0,1]")
        return result


def hard_competitor_softplus_loss(
    logits: torch.Tensor,
    target_classes: torch.Tensor,
) -> torch.Tensor:
    """Push the assigned class above the strongest competing wrong class.

    This loss is class-generic: no validation-derived confusion pairs are
    encoded.  The competitor is selected dynamically from the current logits.
    """

    if logits.ndim != 2:
        raise ValueError(f"logits harus [N,C], dapat {tuple(logits.shape)}")
    if target_classes.ndim != 1 or len(target_classes) != len(logits):
        raise ValueError("target_classes harus [N] dan sejajar dengan logits")
    if not len(logits):
        return logits.sum() * 0.0
    if logits.shape[1] < 2:
        raise ValueError("hard-competitor ranking memerlukan minimal dua kelas")

    target_classes = target_classes.long()
    true_logits = logits.gather(1, target_classes[:, None]).squeeze(1)
    competitor_logits = logits.masked_fill(
        F.one_hot(target_classes, num_classes=logits.shape[1]).bool(),
        torch.finfo(logits.dtype).min,
    ).max(dim=1).values
    return F.softplus(competitor_logits - true_logits).mean()


class HardCompetitorDetectionLoss:
    """Mixin-style factory wrapper around the pinned Ultralytics v8DetectionLoss."""

    @staticmethod
    def build(model: torch.nn.Module, *, weight: float, tal_topk: int, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        frozen_weight = float(weight)

        class _Loss(v8DetectionLoss):
            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, native_loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                fg_mask, target_gt_idx, _, _, _ = assignments
                if not bool(fg_mask.any()):
                    return assignments, native_loss, native_loss.detach()

                pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
                dtype = pred_scores.dtype
                batch_size = pred_scores.shape[0]
                imgsz = (
                    torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
                    * self.stride[0]
                )
                targets = torch.cat(
                    (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
                    1,
                )
                targets = self.preprocess(
                    targets.to(self.device),
                    batch_size,
                    scale_tensor=imgsz[[1, 0, 1, 0]],
                )
                gt_labels = targets[..., 0].long()
                batch_index = torch.arange(batch_size, device=self.device)[:, None].expand_as(target_gt_idx)
                assigned_labels = gt_labels[
                    batch_index,
                    target_gt_idx.clamp(min=0, max=max(gt_labels.shape[1] - 1, 0)),
                ]
                positive_logits = pred_scores[fg_mask]
                positive_labels = assigned_labels[fg_mask]
                rank_loss = hard_competitor_softplus_loss(positive_logits, positive_labels)

                # Native loss has already applied hyp.cls to its classification component.
                # Apply the same gain to the auxiliary ranking term, then fold it into
                # classification so Ultralytics logging remains box/cls/dfl compatible.
                loss = native_loss.clone()
                loss[1] = loss[1] + frozen_weight * self.hyp.cls * rank_loss
                return assignments, loss, loss.detach()

        return _Loss(model, tal_topk=tal_topk, tal_topk2=tal_topk2)


class HardCompetitorE2ELoss:
    """Native YOLO26 E2E loss with HCR only on the final one-to-one branch."""

    def __init__(self, model: torch.nn.Module, *, weight: float):
        from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

        base = E2ELoss(model)
        self.one2many = base.one2many
        self.one2one = HardCompetitorDetectionLoss.build(
            model,
            weight=weight,
            tal_topk=7,
            tal_topk2=1,
        )
        self.updates = base.updates
        self.total = base.total
        self.o2m = base.o2m
        self.o2o = base.o2o
        self.o2m_copy = base.o2m_copy
        self.final_o2m = base.final_o2m
        # Keep a direct reference only for explicit protocol assertions.
        self.native_loss_type = v8DetectionLoss
        self.hard_competitor_weight = float(weight)

    def __call__(self, preds, batch):
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (
            self.o2m_copy - self.final_o2m
        ) + self.final_o2m


class AmbiguityMultilevelRankingDetectionModel(AmbiguityMultilevelDetectionModel):
    """ACMC1 forward architecture with a training-only hard-competitor criterion."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        ambiguity_multilevel: AmbiguityMultilevelConfig | dict[str, Any] | None = None,
        hard_competitor_ranking: HardCompetitorRankingConfig | dict[str, Any] | None = None,
    ) -> None:
        self.hard_competitor_ranking_config = HardCompetitorRankingConfig.from_mapping(
            hard_competitor_ranking
        )
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            ambiguity_multilevel=ambiguity_multilevel,
        )

    def init_criterion(self):
        return HardCompetitorE2ELoss(
            self,
            weight=self.hard_competitor_ranking_config.weight,
        )


def make_ambiguity_multilevel_ranking_trainer(
    ambiguity_config: AmbiguityMultilevelConfig | dict[str, Any],
    ranking_config: HardCompetitorRankingConfig | dict[str, Any],
    *,
    d0_checkpoint: str | Path | None = None,
):
    """Build ACMC1-HCR trainer while preserving native D0 initialization."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen_acmc = AmbiguityMultilevelConfig.from_mapping(ambiguity_config)
    frozen_ranking = HardCompetitorRankingConfig.from_mapping(ranking_config)
    bound_checkpoint = (
        Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint is not None else None
    )

    class AmbiguityMultilevelRankingTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AmbiguityMultilevelRankingDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    ambiguity_multilevel=frozen_acmc,
                    hard_competitor_ranking=frozen_ranking,
                )
            )
            if bound_checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(bound_checkpoint)).model
                transfer = load_ambiguity_multilevel_detector_weights(model, source)
            elif weights:
                transfer = load_ambiguity_multilevel_detector_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"ACMC1-HCR NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    AmbiguityMultilevelRankingTrainer.__name__ = "AmbiguityMultilevelRankingTrainer"
    return AmbiguityMultilevelRankingTrainer
