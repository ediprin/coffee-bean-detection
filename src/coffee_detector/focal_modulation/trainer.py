from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import (
    FocalModulationConfig,
    FocalModulationDetectionModel,
    load_focal_modulation_weights,
)


def make_focal_modulation_trainer(
    config: FocalModulationConfig | dict[str, Any], *, d0_checkpoint: str | Path | None = None
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = FocalModulationConfig.from_mapping(config)
    bound = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class FocalModulationTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                FocalModulationDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    focal_modulation=frozen,
                )
            )
            if bound is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_focal_modulation_weights(model, YOLO(str(bound)).model)
            elif weights:
                transfer = load_focal_modulation_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"FMH1 NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    FocalModulationTrainer.__name__ = "FocalModulationTrainer"
    return FocalModulationTrainer
