from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from coffee_detector.sni21_ontology import SNI21_CLASSES, load_sni21_ontology


ALLOWED_TASKS = (
    "entity_family",
    "primary_condition",
    "hole_count",
    "integrity_fraction",
    "surface_extent",
)
BLOCKED_TASKS = ("physical_size_mm", "relative_completeness", "positive_flag")


@dataclass(frozen=True)
class OntologyMarginalConfig:
    mode: str = "semantic"
    tasks: tuple[str, ...] = ALLOWED_TASKS
    auxiliary_gain: float = 0.20
    task_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)

    @classmethod
    def from_mapping(
        cls, payload: "OntologyMarginalConfig | dict[str, Any] | None"
    ) -> "OntologyMarginalConfig":
        if isinstance(payload, cls):
            return payload
        raw = dict(payload or {})
        if "tasks" in raw:
            raw["tasks"] = tuple(str(value) for value in raw["tasks"])
        if "task_weights" in raw:
            raw["task_weights"] = tuple(float(value) for value in raw["task_weights"])
        result = cls(**raw)
        if result.mode not in {"semantic", "identity_control"}:
            raise ValueError("mode harus semantic atau identity_control")
        if not result.tasks or len(set(result.tasks)) != len(result.tasks):
            raise ValueError("tasks harus unik dan tidak kosong")
        unknown = sorted(set(result.tasks) - set(ALLOWED_TASKS))
        if unknown:
            blocked = sorted(set(unknown) & set(BLOCKED_TASKS))
            if blocked:
                raise ValueError(f"Task belum diizinkan protokol: {blocked}")
            raise ValueError(f"Task ontologi tidak dikenal: {unknown}")
        if len(result.task_weights) != len(result.tasks):
            raise ValueError("task_weights harus sejajar dengan tasks")
        if result.auxiliary_gain < 0 or any(weight <= 0 for weight in result.task_weights):
            raise ValueError("Bobot ontology-marginal harus positif")
        return result

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        payload["task_weights"] = list(self.task_weights)
        return payload


def _task_mapping(ontology: dict, task: str) -> tuple[list[int], tuple[str, ...]]:
    observed = sorted(
        {
            str(row[task])
            for row in ontology["classes"].values()
            if row.get(task) is not None
        }
    )
    value_to_id = {value: index for index, value in enumerate(observed)}
    mapping = [
        value_to_id[str(ontology["classes"][class_name][task])]
        if ontology["classes"][class_name].get(task) is not None
        else -1
        for class_name in SNI21_CLASSES
    ]
    return mapping, tuple(observed)


class OntologyMarginalizer(nn.Module):
    """Project leaf probabilities onto ontology groups without extra parameters."""

    def __init__(
        self,
        config: OntologyMarginalConfig | dict[str, Any] | None = None,
        *,
        ontology_path: str | None = None,
    ) -> None:
        super().__init__()
        self.config = OntologyMarginalConfig.from_mapping(config)
        ontology = load_sni21_ontology(ontology_path)
        self.value_names: dict[str, tuple[str, ...]] = {}
        for task in self.config.tasks:
            mapping, values = _task_mapping(ontology, task)
            self.register_buffer(
                f"mapping_{task}", torch.tensor(mapping, dtype=torch.long), persistent=True
            )
            self.value_names[task] = values

    def _semantic_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mapping: torch.Tensor,
        value_count: int,
    ) -> torch.Tensor | None:
        targets = mapping[labels]
        valid = targets.ge(0)
        if not bool(valid.any()):
            return None
        leaf_log_probability = F.log_softmax(logits[valid], dim=-1)
        grouped = []
        for value_id in range(value_count):
            members = mapping.eq(value_id)
            grouped.append(torch.logsumexp(leaf_log_probability[:, members], dim=1))
        group_log_probability = torch.stack(grouped, dim=1)
        return F.nll_loss(group_log_probability, targets[valid])

    @staticmethod
    def _identity_loss(
        logits: torch.Tensor, labels: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor | None:
        valid = mapping[labels].ge(0)
        if not bool(valid.any()):
            return None
        return F.cross_entropy(logits[valid], labels[valid])

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.ndim != 2 or logits.shape[1] != len(SNI21_CLASSES):
            raise ValueError(f"Logits harus [N,21], diterima {tuple(logits.shape)}")
        labels = labels.to(device=logits.device, dtype=torch.long).reshape(-1)
        if labels.shape[0] != logits.shape[0]:
            raise ValueError("Jumlah labels tidak sama dengan jumlah logits")
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= len(SNI21_CLASSES)):
            raise ValueError("Label leaf di luar rentang SNI-21")

        details: dict[str, torch.Tensor] = {}
        weighted = logits.sum() * 0.0
        weight_sum = 0.0
        for task, weight in zip(self.config.tasks, self.config.task_weights):
            mapping = getattr(self, f"mapping_{task}").to(logits.device)
            if self.config.mode == "semantic":
                task_loss = self._semantic_loss(
                    logits, labels, mapping, len(self.value_names[task])
                )
            else:
                task_loss = self._identity_loss(logits, labels, mapping)
            if task_loss is None:
                continue
            details[task] = task_loss
            weighted = weighted + float(weight) * task_loss
            weight_sum += float(weight)
        if weight_sum:
            weighted = weighted / weight_sum
        return weighted, details


class OntologyDetectionLoss:
    """Ultralytics v8DetectionLoss wrapper with zero-parameter ontology loss."""

    def __new__(cls, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from ultralytics.utils.loss import v8DetectionLoss

        class _BoundOntologyDetectionLoss(v8DetectionLoss):
            def __init__(self):
                super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
                self.ontology = OntologyMarginalizer(model.ontology_marginal_config)

            def get_assigned_targets_and_loss(self, preds, batch):
                assignments, loss, _ = super().get_assigned_targets_and_loss(preds, batch)
                fg_mask, target_gt_idx = assignments[:2]
                if bool(fg_mask.any()):
                    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
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
                    auxiliary, _ = self.ontology(
                        pred_scores[fg_mask], assigned_labels[fg_mask]
                    )
                    loss[1] = loss[1] + float(
                        model.ontology_marginal_config.auxiliary_gain
                    ) * auxiliary
                return assignments, loss, loss.detach()

        return _BoundOntologyDetectionLoss()
