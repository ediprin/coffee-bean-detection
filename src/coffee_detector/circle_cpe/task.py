from __future__ import annotations

from typing import Any

from ultralytics.nn.tasks import DetectionModel

from coffee_detector.fsce_cpe.model import inject_fsce_cpe
from .config import CircleCPEConfig
from .loss import CircleCPEDetectionLoss


class CircleCPEDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        circle_cpe: CircleCPEConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.circle_cpe_config = CircleCPEConfig.from_mapping(circle_cpe)
        inject_fsce_cpe(self, self.circle_cpe_config.projection_config())

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss
        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=CircleCPEDetectionLoss)
        return CircleCPEDetectionLoss(self)
