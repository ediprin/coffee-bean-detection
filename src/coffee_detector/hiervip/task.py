from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ultralytics.nn.tasks import DetectionModel

from .loss import HierVIPDetectionLoss
from .model import HierarchySpec, HierVIPConfig, HierVIPDetectHead


class HierVIPDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        hiervip: HierVIPConfig | Mapping[str, Any] | None = None,
        hierarchy: HierarchySpec | None = None,
    ) -> None:
        self.hiervip_config = HierVIPConfig.from_mapping(hiervip)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if hierarchy is None:
            raise ValueError("HierVIPDetectionModel memerlukan frozen hierarchy")
        self.model[-1] = HierVIPDetectHead(self.model[-1], hierarchy, self.hiervip_config)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=HierVIPDetectionLoss)
        return HierVIPDetectionLoss(self)
