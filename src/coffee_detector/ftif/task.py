from __future__ import annotations

from typing import Any

import torch
from ultralytics.nn.tasks import DetectionModel

from .loss import FTIFDetectionLoss
from .model import FTIFConfig, inject_ftif


class FTIFDetectionModel(DetectionModel):
    """YOLO26 with frozen language priors feeding classification features."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        ftif: FTIFConfig | dict[str, Any] | None = None,
        text_embeddings: torch.Tensor | None = None,
    ) -> None:
        if text_embeddings is None:
            raise ValueError("FTIFDetectionModel memerlukan frozen text embeddings")
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.ftif_config = FTIFConfig.from_mapping(ftif)
        inject_ftif(self, self.ftif_config, text_embeddings)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=FTIFDetectionLoss)
        return FTIFDetectionLoss(self)
