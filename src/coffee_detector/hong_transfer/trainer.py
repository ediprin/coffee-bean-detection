from __future__ import annotations

from typing import Any

from .model import HongTransferConfig, inject_hong_transfer


def make_hong_transfer_trainer(config: HongTransferConfig | dict[str, Any]):
    """Build a resume-safe trainer for the native end-to-end Hong transfer."""

    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang; jalankan pip install -e .") from error

    frozen = HongTransferConfig.from_mapping(config)

    class HongTransferTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            # A resumed checkpoint already contains the custom modules. Loading
            # it through a native graph would silently drop KDS/CDS/PConv state.
            if weights is not None and getattr(weights, "hong_transfer_config", None):
                model = self.set_model_names_for_load(weights)
                inject_hong_transfer(model, frozen)
                return model

            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            inject_hong_transfer(model, frozen)
            return model

    HongTransferTrainer.__name__ = "HongTransferTrainerFull"
    return HongTransferTrainer
