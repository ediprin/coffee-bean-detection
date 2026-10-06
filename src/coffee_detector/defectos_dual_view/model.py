from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn


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
