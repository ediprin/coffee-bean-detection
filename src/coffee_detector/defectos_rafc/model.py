from __future__ import annotations

from typing import Any, Mapping

from ultralytics.nn.tasks import DetectionModel

from .config import RAFCConfig
from .loss import RAFCE2ELoss


class RAFCDetectionModel(DetectionModel):
    """Native YOLO detector with a training-only paired Fourier objective."""

    def __init__(
        self,
        *args: Any,
        rafc: RAFCConfig | Mapping[str, object] | None = None,
        **kwargs: Any,
    ) -> None:
        self.rafc_config = RAFCConfig.from_mapping(rafc)
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        return RAFCE2ELoss(self, self.rafc_config)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            if "img_fourier" not in batch:
                raise KeyError("Batch RAFC kehilangan img_fourier")
            preds = (self.forward(batch["img"]), self.forward(batch["img_fourier"]))
        return self.criterion(preds, batch)


def build_rafc_model(
    cfg: str,
    *,
    nc: int,
    source=None,
    seed: int,
    config: RAFCConfig | Mapping[str, object] | None = None,
    verbose: bool = False,
) -> RAFCDetectionModel:
    import torch

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RAFCDetectionModel(
            cfg, ch=3, nc=nc, verbose=verbose, rafc=RAFCConfig.from_mapping(config)
        )
        if source is not None:
            model.load(source)
    return model


def load_rafc_weights(model: RAFCDetectionModel, weights) -> dict:
    source = getattr(weights, "model", weights)
    if source is None:
        raise TypeError("Checkpoint RAFC tidak mengekspos model torch")
    return model.load(source)
