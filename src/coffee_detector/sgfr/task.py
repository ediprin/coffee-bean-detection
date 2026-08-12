from __future__ import annotations

from typing import Any, Mapping

from .loss import SGFRDetectionLoss
from .model import SGFRConfig, SGFRDetectHead, SGFRDetectionModel


class SGFRTaskModel(SGFRDetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        sgfr: SGFRConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, sgfr=sgfr)

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if not isinstance(self.model[-1], SGFRDetectHead):
            raise TypeError(type(self.model[-1]).__name__)
        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=SGFRDetectionLoss)
        return SGFRDetectionLoss(self)
