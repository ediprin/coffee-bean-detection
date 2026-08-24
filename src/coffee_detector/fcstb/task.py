from __future__ import annotations

from typing import Any, Mapping

from .loss import FCSTBE2ELoss
from .model import FCSTBConfig, FCSTBDetectionModel


class FCSTBTaskModel(FCSTBDetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        stb: Mapping[str, Any] | None = None,
        fcstb: FCSTBConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose, stb=stb, fcstb=fcstb)

    def init_criterion(self):
        if not getattr(self, "end2end", False):
            raise RuntimeError("FC-STB dikunci untuk YOLO26 end-to-end")
        return FCSTBE2ELoss(self, self.fcstb_config)
