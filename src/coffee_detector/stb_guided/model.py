from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from torch import nn

from coffee_detector.wav1_factorization.config import WAV1FactorizationConfig
from coffee_detector.wav1_factorization.model import (
    WAV1FactorizationDetectionModel,
    load_factorization_weights,
)

from .config import STBGuidedConfig


class STBGuidedDetectionModel(WAV1FactorizationDetectionModel):
    """Deployed WAV-L1 + native YOLO26 student with training-only guidance.

    STB1 and AF2 live in the criterion, not on the serialized model. Therefore
    S2 and S3 have exactly the same inference architecture as WAV-L1.
    """

    def __init__(
        self,
        cfg="yolo26.yaml",
        ch=3,
        nc=None,
        verbose=True,
        factorization: WAV1FactorizationConfig | Mapping[str, Any] | None = None,
        stb_guided: STBGuidedConfig | Mapping[str, Any] | None = None,
    ) -> None:
        frozen_factorization = WAV1FactorizationConfig.from_mapping(factorization)
        if frozen_factorization.arm != "WAV_L1":
            raise ValueError("STB-guided student dikunci ke deployed WAV_L1 frontend")
        self.stb_guided_config = STBGuidedConfig.from_mapping(stb_guided)
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            factorization=frozen_factorization,
        )

    @property
    def teacher_path(self) -> Path | None:
        value = self.stb_guided_config.teacher_checkpoint
        return Path(value).expanduser().resolve() if value else None

    def init_criterion(self):
        if not getattr(self, "end2end", False):
            raise RuntimeError("STB-guided training dikunci untuk YOLO26 end-to-end")
        from .loss import STBGuidedE2ELoss

        return STBGuidedE2ELoss(self, self.stb_guided_config)


def load_stb_guided_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Reuse the exact WAV-L1/native-YOLO transfer contract."""

    return load_factorization_weights(model, weights)
