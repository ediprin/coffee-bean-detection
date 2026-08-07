from __future__ import annotations

from typing import Any, Mapping

from ultralytics.nn.tasks import DetectionModel

from .loss import IGEMDetectionLoss
from .model import IGEMConfig, IGEMDetectHead


class IGEMDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        igem: IGEMConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.igem_config = IGEMConfig.from_mapping(igem)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.model[-1] = IGEMDetectHead(self.model[-1], self.igem_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=IGEMDetectionLoss)
        return IGEMDetectionLoss(self)
