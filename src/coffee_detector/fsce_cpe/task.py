from __future__ import annotations

from typing import Any

from ultralytics.nn.tasks import DetectionModel

from .loss import FSCECPEDetectionLoss
from .model import FSCECPEConfig, inject_fsce_cpe


class FSCECPEDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        cpe: FSCECPEConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.cpe_config = FSCECPEConfig.from_mapping(cpe)
        inject_fsce_cpe(self, self.cpe_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss
        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=FSCECPEDetectionLoss)
        return FSCECPEDetectionLoss(self)
