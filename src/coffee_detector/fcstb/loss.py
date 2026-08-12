from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .model import FCSTBConfig


def gt_bounded_logit_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    minimum_gt_probability: float,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """KL distillation only where the frozen teacher predicts the GT class."""

    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("Student/teacher logits harus sejajar [N,C]")
    labels = labels.to(device=student_logits.device, dtype=torch.long).reshape(-1)
    if labels.shape[0] != student_logits.shape[0]:
        raise ValueError("Label distillation tidak sejajar")
    teacher_probability = F.softmax(teacher_logits / float(temperature), dim=-1)
    gt_probability = teacher_probability.gather(1, labels[:, None]).squeeze(1)
    keep = teacher_probability.argmax(dim=-1).eq(labels) & gt_probability.ge(
        float(minimum_gt_probability)
    )
    if not bool(keep.any()):
        return student_logits.sum() * 0.0, {
            "positive_anchors": int(labels.numel()),
            "teacher_correct_anchors": 0,
            "teacher_correct_fraction": 0.0,
        }
    per_anchor = F.kl_div(
        F.log_softmax(student_logits[keep] / float(temperature), dim=-1),
        teacher_probability[keep],
        reduction="none",
    ).sum(dim=-1)
    # Confidence weighting suppresses weak yet technically top-1 teacher cues.
    weight = gt_probability[keep].detach()
    loss = (per_anchor * weight).sum() / weight.sum().clamp_min(1e-8)
    loss = loss * float(temperature) ** 2
    return loss, {
        "positive_anchors": int(labels.numel()),
        "teacher_correct_anchors": int(keep.sum()),
        "teacher_correct_fraction": float(keep.float().mean()),
    }


class FCSTBBranchLoss:
    """Factory binding native YOLO loss to one optional teacher branch."""

    def __new__(cls, model: torch.nn.Module, config: FCSTBConfig, **kwargs: Any):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundFCSTBBranchLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, **kwargs)
                self.fcstb_config = config
                self.teacher_branch: dict[str, torch.Tensor] | None = None
                self.last_distillation_stats: dict[str, int | float] = {}

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                teacher = self.teacher_branch
                if self.fcstb_config.mode == "control" or teacher is None:
                    return assignments, loss, loss.detach()
                fg_mask, target_gt_idx = assignments[:2]
                if not bool(fg_mask.any()):
                    return assignments, loss, loss.detach()
                student_scores = preds["scores"].permute(0, 2, 1).contiguous()
                teacher_scores = teacher["scores"].permute(0, 2, 1).contiguous()
                if teacher_scores.shape != student_scores.shape:
                    raise RuntimeError(
                        f"Anchor teacher {tuple(teacher_scores.shape)} tidak sejajar "
                        f"dengan student {tuple(student_scores.shape)}"
                    )
                batch_size = student_scores.shape[0]
                image_size = (
                    torch.tensor(
                        preds["feats"][0].shape[2:],
                        device=self.device,
                        dtype=student_scores.dtype,
                    )
                    * self.stride[0]
                )
                targets = torch.cat(
                    (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
                    1,
                )
                targets = self.preprocess(
                    targets.to(self.device),
                    batch_size,
                    scale_tensor=image_size[[1, 0, 1, 0]],
                )
                labels = targets[..., 0].long().gather(1, target_gt_idx.long())
                auxiliary, stats = gt_bounded_logit_distillation(
                    student_scores[fg_mask],
                    teacher_scores[fg_mask].detach(),
                    labels[fg_mask],
                    temperature=self.fcstb_config.temperature,
                    minimum_gt_probability=self.fcstb_config.minimum_teacher_gt_probability,
                )
                self.last_distillation_stats = stats
                loss[1] = loss[1] + float(self.fcstb_config.distillation_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundFCSTBBranchLoss()


class FCSTBE2ELoss:
    """YOLO26 E2E loss with one AF2 teacher forward per training batch."""

    def __init__(self, model: torch.nn.Module, config: FCSTBConfig) -> None:
        self.config = config
        self.one2many = FCSTBBranchLoss(model, config, tal_topk=10)
        self.one2one = FCSTBBranchLoss(model, config, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = 0.2
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1
        self.teacher = self._load_teacher(model) if config.mode == "distill" else None

    def _load_teacher(self, model: torch.nn.Module) -> torch.nn.Module:
        from ultralytics import YOLO

        path = model.teacher_path
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Checkpoint teacher AF2 tidak ditemukan: {path}")
        teacher = YOLO(str(path)).model.to(next(model.parameters()).device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher

    @staticmethod
    def _teacher_raw(output: Any) -> dict[str, dict[str, torch.Tensor]]:
        if not isinstance(output, tuple) or len(output) < 2 or not isinstance(output[1], dict):
            raise TypeError("Teacher AF2 tidak menyediakan raw prediction branches")
        raw = output[1]
        if "one2one" not in raw:
            raise KeyError("Teacher AF2 kehilangan one2one")
        return raw

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]):
        parsed = self.one2many.parse_output(preds)
        if self.teacher is not None:
            self.teacher.eval()
            with torch.inference_mode():
                raw = self._teacher_raw(self.teacher(batch["img"]))
            self.one2many.teacher_branch = raw.get("one2many", raw["one2one"])
            self.one2one.teacher_branch = raw["one2one"]
        else:
            self.one2many.teacher_branch = None
            self.one2one.teacher_branch = None
        loss_many = self.one2many.loss(parsed["one2many"], batch)
        loss_one = self.one2one.loss(parsed["one2one"], batch)
        return loss_many[0] * self.o2m + loss_one[0] * self.o2o, loss_one[1]

    def update(self) -> None:
        self.updates += 1
        self.o2m = max(1 - self.updates / max(self.one2one.hyp.epochs - 1, 1), 0) * (
            self.o2m_copy - self.final_o2m
        ) + self.final_o2m
        self.o2o = max(self.total - self.o2m, 0)
