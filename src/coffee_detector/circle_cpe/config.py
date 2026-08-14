from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from coffee_detector.fsce_cpe.model import FSCECPEConfig


@dataclass(frozen=True)
class CircleCPEConfig:
    embedding_dim: int = 128
    iou_threshold: float = 0.7
    margin: float = 0.25
    gamma: float = 256.0
    loss_weight: float = 0.005

    @classmethod
    def from_mapping(cls, payload: "CircleCPEConfig | dict[str, Any] | None") -> "CircleCPEConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.embedding_dim <= 0:
            raise ValueError("embedding_dim harus positif")
        if not 0.0 <= result.iou_threshold < 1.0:
            raise ValueError("iou_threshold harus berada pada [0,1)")
        if not 0.0 < result.margin < 1.0:
            raise ValueError("margin harus berada pada (0,1)")
        if result.gamma <= 0:
            raise ValueError("gamma harus positif")
        if result.loss_weight < 0:
            raise ValueError("loss_weight tidak boleh negatif")
        return result

    def projection_config(self) -> FSCECPEConfig:
        # Temperature is unused by Circle-CPE. It is populated only because the
        # frozen FSCE-CPE projection wrapper stores FSCECPEConfig.
        return FSCECPEConfig(
            embedding_dim=self.embedding_dim,
            temperature=0.2,
            iou_threshold=self.iou_threshold,
            loss_weight=self.loss_weight,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
