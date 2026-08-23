from __future__ import annotations

from typing import Any

from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig
from coffee_detector.fsce_cpe.loss import FSCECPEDetectionLoss
from coffee_detector.fsce_cpe.model import FSCECPEConfig, inject_fsce_cpe


class AF2CPEDetectionModel(AFABDetectionModel):
    """Frozen AF2 input operator plus the CPE training-only auxiliary head."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        afab: AFABConfig | dict[str, Any] | None = None,
        cpe: FSCECPEConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, afab=afab)
        self.cpe_config = FSCECPEConfig.from_mapping(cpe)
        inject_fsce_cpe(self, self.cpe_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=FSCECPEDetectionLoss)
        return FSCECPEDetectionLoss(self)
