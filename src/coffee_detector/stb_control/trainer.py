from __future__ import annotations

from pathlib import Path
from typing import Any

from coffee_detector.stb import STBConfig

from .model import STBCapacityControlDetectionModel, load_stb_control_weights


def make_stb_control_trainer(
    config: STBConfig | dict[str, Any], *, d0_checkpoint: str | Path
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = STBConfig.from_mapping(config)
    bound = Path(d0_checkpoint).expanduser().resolve()

    class STBCapacityControlTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                STBCapacityControlDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    stb=frozen,
                )
            )
            if weights is not None:
                transfer = load_stb_control_weights(model, weights)
                print(f"CMC0 RESUME WEIGHT TRANSFER: {transfer}", flush=True)
            elif not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_stb_control_weights(model, YOLO(str(bound)).model)
                print(f"CMC0 NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    STBCapacityControlTrainer.__name__ = "STBCapacityControlTrainer"
    return STBCapacityControlTrainer
