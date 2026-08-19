from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer
from coffee_detector.stb.model import STBDetectHead

from .config import STBGuidedConfig


def gt_bounded_cross_head_kl(
    cross_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    minimum_gt_probability: float,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Distill teacher class distributions through a frozen teacher head.

    The student *native classification head* is not given the teacher loss.
    Instead, non-detached student P3/P4/P5 features are passed through the
    frozen STB1 inference classification head and only those cross-head logits
    receive the KL objective. This is a CrossKD-inspired adaptation that keeps
    GT supervision and teacher supervision on different prediction paths.
    """

    if cross_logits.shape != teacher_logits.shape or cross_logits.ndim != 2:
        raise ValueError("Cross-head/teacher logits harus sejajar [N,C]")
    labels = labels.to(device=cross_logits.device, dtype=torch.long).reshape(-1)
    if labels.shape[0] != cross_logits.shape[0]:
        raise ValueError("Label distillation tidak sejajar")

    t = float(temperature)
    teacher_probability = F.softmax(teacher_logits / t, dim=-1)
    gt_probability = teacher_probability.gather(1, labels[:, None]).squeeze(1)
    keep = teacher_probability.argmax(dim=-1).eq(labels) & gt_probability.ge(
        float(minimum_gt_probability)
    )

    if not bool(keep.any()):
        return cross_logits.sum() * 0.0, {
            "positive_anchors": int(labels.numel()),
            "teacher_correct_anchors": 0,
            "teacher_correct_fraction": 0.0,
        }

    per_anchor = F.kl_div(
        F.log_softmax(cross_logits[keep] / t, dim=-1),
        teacher_probability[keep],
        reduction="none",
    ).sum(dim=-1)
    # Reuse the frozen confidence-bounded rule from the earlier reverse
    # distillation experiment, but change the optimization path to cross-head.
    weight = gt_probability[keep].detach()
    loss = (per_anchor * weight).sum() / weight.sum().clamp_min(1.0e-8)
    loss = loss * t**2
    return loss, {
        "positive_anchors": int(labels.numel()),
        "teacher_correct_anchors": int(keep.sum()),
        "teacher_correct_fraction": float(keep.float().mean()),
    }


def positive_consistency_kl(
    clean_logits: torch.Tensor,
    shifted_logits: torch.Tensor,
    foreground_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """One-way clean->shifted class consistency on the same positive anchors."""

    if clean_logits.shape != shifted_logits.shape or clean_logits.ndim != 3:
        raise ValueError("clean/shifted logits harus sejajar [B,A,C]")
    if foreground_mask.shape != clean_logits.shape[:2]:
        raise ValueError("foreground mask tidak sejajar dengan anchor logits")
    if not bool(foreground_mask.any()):
        return shifted_logits.sum() * 0.0
    t = float(temperature)
    clean = clean_logits[foreground_mask].detach()
    shifted = shifted_logits[foreground_mask]
    target = F.softmax(clean / t, dim=-1)
    return (
        F.kl_div(F.log_softmax(shifted / t, dim=-1), target, reduction="batchmean")
        * t**2
    )


def cross_head_class_scores(
    teacher_head: STBDetectHead,
    student_features: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    branch_name: str = "one2one",
) -> torch.Tensor:
    """Run student neck features through the frozen STB1 class pathway.

    No `no_grad` context is used here: teacher parameters are frozen, but the
    computational graph must remain connected to student features.
    """

    if not isinstance(teacher_head, STBDetectHead):
        raise TypeError(f"Teacher head harus STBDetectHead, diterima {type(teacher_head).__name__}")
    if len(student_features) != teacher_head.nl or len(student_features) != len(teacher_head.blocks):
        raise ValueError("Jumlah level student tidak sejajar dengan STB1 teacher")
    branch = getattr(teacher_head, branch_name, None)
    if not isinstance(branch, dict) or not branch.get("cls_head"):
        raise RuntimeError(f"Teacher kehilangan classification branch {branch_name}")

    enhanced = [
        block(feature) for block, feature in zip(teacher_head.blocks, student_features)
    ]
    batch = enhanced[0].shape[0]
    logits = [
        branch["cls_head"][index](enhanced[index]) for index in range(teacher_head.nl)
    ]
    return torch.cat(
        [value.view(batch, teacher_head.nc, -1) for value in logits], dim=-1
    )


class STBGuidedBranchLoss:
    """Bind native YOLO loss to an optional cross-head classification KD term."""

    def __new__(
        cls,
        model: torch.nn.Module,
        config: STBGuidedConfig,
        *,
        enable_crosskd: bool,
        **kwargs: Any,
    ):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundSTBGuidedBranchLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, **kwargs)
                self.guided_config = config
                self.enable_crosskd = bool(enable_crosskd)
                self.teacher_head: STBDetectHead | None = None
                self.teacher_scores: torch.Tensor | None = None
                self.last_distillation_stats: dict[str, int | float] = {}
                self.last_foreground_mask: torch.Tensor | None = None

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                fg_mask, target_gt_idx = assignments[:2]
                self.last_foreground_mask = fg_mask.detach()

                if (
                    not self.enable_crosskd
                    or self.teacher_head is None
                    or self.teacher_scores is None
                    or not bool(fg_mask.any())
                ):
                    return assignments, loss, loss.detach()

                # YOLO26 one2one features are detached by design. Therefore the
                # authorized CrossKD path is attached to non-detached one2many
                # P3/P4/P5 features, while using the teacher's one2one inference
                # classification head as the frozen cross-head target pathway.
                cross_scores = cross_head_class_scores(
                    self.teacher_head,
                    preds["feats"],
                    branch_name="one2one",
                ).permute(0, 2, 1).contiguous()
                teacher_scores = self.teacher_scores.permute(0, 2, 1).contiguous()
                if cross_scores.shape != teacher_scores.shape:
                    raise RuntimeError(
                        f"Cross-head {tuple(cross_scores.shape)} tidak sejajar dengan "
                        f"teacher {tuple(teacher_scores.shape)}"
                    )

                batch_size = cross_scores.shape[0]
                image_size = (
                    torch.tensor(
                        preds["feats"][0].shape[2:],
                        device=self.device,
                        dtype=cross_scores.dtype,
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
                labels = targets[..., 0].long().gather(1, target_gt_idx.long())
                auxiliary, stats = gt_bounded_cross_head_kl(
                    cross_scores[fg_mask],
                    teacher_scores[fg_mask].detach(),
                    labels[fg_mask],
                    temperature=self.guided_config.temperature,
                    minimum_gt_probability=self.guided_config.minimum_teacher_gt_probability,
                )
                self.last_distillation_stats = stats
                loss[1] = loss[1] + float(self.guided_config.distillation_weight) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundSTBGuidedBranchLoss()


class STBGuidedE2ELoss:
    """YOLO26 E2E loss for S2 CrossKD and the later S3 AF2 robustness stage."""

    def __init__(self, model: torch.nn.Module, config: STBGuidedConfig) -> None:
        self.model = model
        self.config = config
        self.one2many = STBGuidedBranchLoss(
            model,
            config,
            enable_crosskd=True,
            tal_topk=10,
        )
        self.one2one = STBGuidedBranchLoss(
            model,
            config,
            enable_crosskd=False,
            tal_topk=7,
            tal_topk2=1,
        )
        self.robust_many = None
        self.robust_one = None
        self.af2 = None
        if config.mode == "crosskd_af2":
            self.robust_many = STBGuidedBranchLoss(
                model,
                config,
                enable_crosskd=False,
                tal_topk=10,
            )
            self.robust_one = STBGuidedBranchLoss(
                model,
                config,
                enable_crosskd=False,
                tal_topk=7,
                tal_topk2=1,
            )
            self.af2 = AFABInputEnhancer(
                AFABConfig(
                    mode="af2",
                    patch_size=32,
                    overlap=0.50,
                    gamma=0.10,
                    angular_bins=360,
                    chunk_size=128,
                    eps=1.0e-8,
                )
            )

        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = 0.2
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1
        self.teacher = self._load_teacher(model)

    def _load_teacher(self, model: torch.nn.Module) -> torch.nn.Module:
        from ultralytics import YOLO

        path = model.teacher_path
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Checkpoint STB1 teacher tidak ditemukan: {path}")
        # Importing STB above also guarantees the custom class is available for
        # checkpoint deserialization before Ultralytics opens the file.
        teacher = YOLO(str(path)).model.to(next(model.parameters()).device).eval()
        head = getattr(teacher, "model", [None])[-1]
        if not isinstance(head, STBDetectHead):
            raise TypeError(f"Teacher checkpoint bukan STB1: {type(head).__name__}")
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher

    @staticmethod
    def _teacher_raw(output: Any) -> dict[str, dict[str, torch.Tensor]]:
        if not isinstance(output, tuple) or len(output) < 2 or not isinstance(output[1], dict):
            raise TypeError("STB1 teacher tidak menyediakan raw prediction branches")
        raw = output[1]
        if "one2one" not in raw:
            raise KeyError("STB1 teacher kehilangan one2one")
        return raw

    def _clean_loss(self, preds: Any, batch: dict[str, torch.Tensor]):
        parsed = self.one2many.parse_output(preds)
        self.teacher.eval()
        with torch.no_grad():
            teacher_raw = self._teacher_raw(self.teacher(batch["img"]))
        teacher_head = self.teacher.model[-1]
        self.one2many.teacher_head = teacher_head
        self.one2many.teacher_scores = teacher_raw["one2one"]["scores"]
        loss_many = self.one2many.loss(parsed["one2many"], batch)
        loss_one = self.one2one.loss(parsed["one2one"], batch)
        total = loss_many[0] * self.o2m + loss_one[0] * self.o2o
        return parsed, total, loss_one[1]

    def _robust_loss(
        self,
        clean_parsed: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.af2 is None or self.robust_many is None or self.robust_one is None:
            zero = clean_parsed["one2many"]["scores"].sum() * 0.0
            return zero, zero.detach(), zero

        self.af2 = self.af2.to(batch["img"].device)
        with torch.no_grad():
            af2_image = self.af2(batch["img"])
        # Tensor input calls predict(), not loss(), so this is a second student
        # forward rather than recursive criterion execution. WAV-L1 is still
        # applied by the deployed student frontend after AF2 creates the view.
        shifted_preds = self.model(af2_image)
        shifted_parsed = self.robust_many.parse_output(shifted_preds)
        shifted_many = self.robust_many.loss(shifted_parsed["one2many"], batch)
        shifted_one = self.robust_one.loss(shifted_parsed["one2one"], batch)
        shifted_total = shifted_many[0] * self.o2m + shifted_one[0] * self.o2o

        fg_mask = self.one2many.last_foreground_mask
        if fg_mask is None:
            consistency = shifted_total * 0.0
        else:
            clean_scores = clean_parsed["one2many"]["scores"].permute(0, 2, 1).contiguous()
            shifted_scores = shifted_parsed["one2many"]["scores"].permute(0, 2, 1).contiguous()
            consistency = positive_consistency_kl(
                clean_scores,
                shifted_scores,
                fg_mask,
                temperature=self.config.consistency_temperature,
            )
        return shifted_total, shifted_one[1], consistency

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]):
        clean_parsed, total, loss_items = self._clean_loss(preds, batch)
        if self.config.mode == "crosskd_af2":
            shifted_total, shifted_items, consistency = self._robust_loss(clean_parsed, batch)
            batch_size = int(batch["img"].shape[0])
            total = (
                total
                + float(self.config.af2_detection_weight) * shifted_total
                + float(self.config.consistency_weight) * consistency * batch_size
            )
            loss_items = loss_items + float(self.config.af2_detection_weight) * shifted_items
            loss_items = loss_items.clone()
            loss_items[1] = loss_items[1] + float(self.config.consistency_weight) * consistency.detach()
        return total, loss_items

    def update(self) -> None:
        self.updates += 1
        self.o2m = max(
            1 - self.updates / max(self.one2one.hyp.epochs - 1, 1),
            0,
        ) * (self.o2m_copy - self.final_o2m) + self.final_o2m
        self.o2o = max(self.total - self.o2m, 0)
