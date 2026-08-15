from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import (
    GeometryConditionedDetectionModel,
    GeometryConditioningConfig,
    load_geometry_conditioned_weights,
)


def make_geometry_conditioning_trainer(
    config: GeometryConditioningConfig | dict[str, Any],
    *,
    d0_checkpoint: str | Path,
    signal_mode: str,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = GeometryConditioningConfig.from_mapping(config)
    bound = Path(d0_checkpoint).expanduser().resolve()
    if signal_mode not in {"geometry", "zero"}:
        raise ValueError("signal_mode harus geometry/zero")

    class GeometryConditioningTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                GeometryConditionedDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    geometry_conditioning=frozen,
                    signal_mode=signal_mode,
                    class_names=self.data["names"],
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_geometry_conditioned_weights(
                    model, YOLO(str(bound)).model
                )
            elif weights:
                transfer = load_geometry_conditioned_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                label = "GEO1" if signal_mode == "geometry" else "GEO-C0"
                print(f"{label} HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    GeometryConditioningTrainer.__name__ = (
        "GEO1Trainer" if signal_mode == "geometry" else "GEOC0Trainer"
    )
    return GeometryConditioningTrainer
