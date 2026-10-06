from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DualViewFusionConfig:
    classes: int = 12
    hidden: int = 24
    residual_limit: float = 2.0

    @classmethod
    def from_mapping(cls, values: Mapping) -> "DualViewFusionConfig":
        result = cls(
            classes=int(values.get("classes", 12)),
            hidden=int(values.get("hidden", 24)),
            residual_limit=float(values.get("residual_limit", 2.0)),
        )
        if result.classes < 2 or result.hidden < 4:
            raise ValueError("Konfigurasi DVF1 terlalu kecil")
        if not 0.0 < result.residual_limit <= 4.0:
            raise ValueError("residual_limit harus berada di (0,4]")
        return result

    def to_dict(self) -> dict:
        return asdict(self)


def paper_base_logits(view_logits: torch.Tensor) -> torch.Tensor:
    """Select the paper's maximum-confidence view without order bias on ties."""

    if view_logits.ndim != 3 or view_logits.shape[1] != 2:
        raise ValueError("view_logits harus berbentuk [B,2,C]")
    confidence = view_logits.sigmoid().amax(dim=2)
    left, right = view_logits[:, 0], view_logits[:, 1]
    left_wins = confidence[:, 0] > confidence[:, 1]
    right_wins = confidence[:, 1] > confidence[:, 0]
    average = (left + right) * 0.5
    return torch.where(
        left_wins[:, None],
        left,
        torch.where(right_wins[:, None], right, average),
    )


class SymmetricDualViewFuser(nn.Module):
    """Order-invariant bounded residual over the paper max-confidence rule."""

    def __init__(self, config: DualViewFusionConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.classes),
            nn.Linear(config.classes, config.hidden),
            nn.GELU(),
        )
        feature_dim = config.hidden * 2 + config.classes * 3
        self.residual = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.classes),
        )
        # Exact identity at initialization. Earlier layers still receive
        # gradients after the first optimizer update changes this final layer.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, view_logits: torch.Tensor) -> torch.Tensor:
        if view_logits.ndim != 3 or view_logits.shape[1:] != (
            2,
            self.config.classes,
        ):
            raise ValueError(
                f"view_logits harus [B,2,{self.config.classes}], "
                f"ditemukan {tuple(view_logits.shape)}"
            )
        left, right = view_logits[:, 0], view_logits[:, 1]
        left_encoded, right_encoded = self.encoder(left), self.encoder(right)
        symmetric = torch.cat(
            (
                (left_encoded + right_encoded) * 0.5,
                (left_encoded - right_encoded).abs(),
                torch.maximum(left, right),
                (left + right) * 0.5,
                (left - right).abs(),
            ),
            dim=1,
        )
        correction = self.config.residual_limit * torch.tanh(
            self.residual(symmetric)
        )
        return paper_base_logits(view_logits) + correction


@dataclass(frozen=True)
class SelectiveDualViewFusionConfig:
    """Configuration for the train-only selective DVF2 correction expert."""

    classes: int = 12
    hidden: int = 32
    expert_hidden: int = 48
    gate_hidden: int = 24
    residual_limit: float = 2.0
    gate_threshold: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping) -> "SelectiveDualViewFusionConfig":
        result = cls(
            classes=int(values.get("classes", 12)),
            hidden=int(values.get("hidden", 32)),
            expert_hidden=int(values.get("expert_hidden", 48)),
            gate_hidden=int(values.get("gate_hidden", 24)),
            residual_limit=float(values.get("residual_limit", 2.0)),
            gate_threshold=float(values.get("gate_threshold", 0.5)),
        )
        if min(result.classes, result.hidden, result.expert_hidden, result.gate_hidden) < 2:
            raise ValueError("Konfigurasi DVF2 terlalu kecil")
        if not 0.0 < result.residual_limit <= 4.0:
            raise ValueError("residual_limit harus berada di (0,4]")
        if not 0.0 < result.gate_threshold < 1.0:
            raise ValueError("gate_threshold harus berada di (0,1)")
        return result

    def to_dict(self) -> dict:
        return asdict(self)


class SelectiveDualViewFuser(nn.Module):
    """Symmetric correction expert with an explicit paper-rule fallback gate.

    Training uses a differentiable soft gate. Evaluation uses the frozen 0.5
    decision: uncertain pairs retain the paper maximum-confidence endpoint
    exactly, while only gate-positive pairs receive a bounded correction.
    """

    def __init__(self, config: SelectiveDualViewFusionConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.classes),
            nn.Linear(config.classes, config.hidden),
            nn.GELU(),
        )
        # Two encoded symmetric summaries, four per-class evidence summaries,
        # and five pair-level uncertainty/agreement scalars.
        feature_dim = config.hidden * 2 + config.classes * 4 + 5
        self.expert = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.expert_hidden),
            nn.GELU(),
            nn.Linear(config.expert_hidden, config.classes),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.gate_hidden),
            nn.GELU(),
            nn.Linear(config.gate_hidden, 1),
        )
        # Exact paper-rule identity before training.
        nn.init.zeros_(self.expert[-1].weight)
        nn.init.zeros_(self.expert[-1].bias)

    def _features(self, view_logits: torch.Tensor) -> torch.Tensor:
        left, right = view_logits[:, 0], view_logits[:, 1]
        left_encoded, right_encoded = self.encoder(left), self.encoder(right)
        probabilities = view_logits.softmax(dim=2)
        left_probability, right_probability = probabilities[:, 0], probabilities[:, 1]
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=2)
        top2 = probabilities.topk(2, dim=2).values
        margin = top2[:, :, 0] - top2[:, :, 1]
        cosine = F.cosine_similarity(left_probability, right_probability, dim=1)
        uncertainty = torch.stack(
            (
                entropy.mean(dim=1),
                (entropy[:, 0] - entropy[:, 1]).abs(),
                margin.mean(dim=1),
                (margin[:, 0] - margin[:, 1]).abs(),
                cosine,
            ),
            dim=1,
        )
        return torch.cat(
            (
                (left_encoded + right_encoded) * 0.5,
                (left_encoded - right_encoded).abs(),
                torch.maximum(left, right),
                (left + right) * 0.5,
                (left - right).abs(),
                left_probability * right_probability,
                uncertainty,
            ),
            dim=1,
        )

    def forward(
        self, view_logits: torch.Tensor, *, return_details: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if view_logits.ndim != 3 or view_logits.shape[1:] != (
            2,
            self.config.classes,
        ):
            raise ValueError(
                f"view_logits harus [B,2,{self.config.classes}], "
                f"ditemukan {tuple(view_logits.shape)}"
            )
        base = paper_base_logits(view_logits)
        features = self._features(view_logits)
        gate_logit = self.gate(features).squeeze(1)
        gate_probability = gate_logit.sigmoid()
        gate = (
            gate_probability
            if self.training
            else (gate_probability > self.config.gate_threshold).to(view_logits.dtype)
        )
        correction = self.config.residual_limit * torch.tanh(self.expert(features))
        fused = base + gate[:, None] * correction
        if not return_details:
            return fused
        return fused, {
            "base": base,
            "gate_logit": gate_logit,
            "gate_probability": gate_probability,
            "effective_gate": gate,
            "correction": correction,
        }
