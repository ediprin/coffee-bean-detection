from __future__ import annotations

from typing import Any

from ultralytics.nn.tasks import DetectionModel

from .loss import PCLDetectionLoss
from .model import PCLConfig, inject_pcl


class PCLDetectionModel(DetectionModel):
    """YOLO26 with a training-only PCLDet learned-prototype branch."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        pcl: PCLConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.pcl_config = PCLConfig.from_mapping(pcl)
        inject_pcl(self, self.pcl_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=PCLDetectionLoss)
        return PCLDetectionLoss(self)
