from __future__ import annotations

from typing import Any

from ultralytics.nn.tasks import DetectionModel

from .loss import APCLDetectionLoss
from .model import APCLConfig, inject_apcl


class APCLDetectionModel(DetectionModel):
    """YOLO26 with a training-only APCL projection/prototype branch."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        apcl: APCLConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.apcl_config = APCLConfig.from_mapping(apcl)
        inject_apcl(self, self.apcl_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=APCLDetectionLoss)
        return APCLDetectionLoss(self)
