from __future__ import annotations

from typing import Any

from ultralytics.nn.tasks import DetectionModel

from .loss import SSCBDetectionLoss
from .model import SSCBConfig, inject_sscb


class SSCBDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        sscb: SSCBConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.sscb_config = SSCBConfig.from_mapping(sscb)
        inject_sscb(self, self.sscb_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss
        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=SSCBDetectionLoss)
        return SSCBDetectionLoss(self)
