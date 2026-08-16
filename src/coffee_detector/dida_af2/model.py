from __future__ import annotations

from typing import Any

import torch

from coffee_detector.afab.model import AFABDetectionModel

from .config import DIDAAF2Config
from .loss import DIDAE2ELoss


class DIDAAF2DetectionModel(AFABDetectionModel):
    """AF2-YOLO26 with a paired training objective and unchanged inference."""

    def __init__(self, *args: Any, dida: DIDAAF2Config | dict | None = None, **kwargs: Any):
        self.dida_config = DIDAAF2Config.from_mapping(dida)
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        return DIDAE2ELoss(self, self.dida_config)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            if "img_style" not in batch:
                raise KeyError("Batch DIDA kehilangan img_style")
            preds = (self.forward(batch["img"]), self.forward(batch["img_style"]))
        return self.criterion(preds, batch)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        # Inference remains the exact deterministic AF2 path inherited from AFAB.
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )
