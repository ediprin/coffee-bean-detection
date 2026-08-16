from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import (
    GeometryFactorizationConfig,
    GeometryFactorizedDetectionModel,
    load_geometry_factorized_weights,
)


def make_geometry_factorization_trainer(
    config: GeometryFactorizationConfig | dict[str, Any],
    *,
    d0_checkpoint: str | Path,
    mode: str,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = GeometryFactorizationConfig.from_mapping(config)
    bound = Path(d0_checkpoint).expanduser().resolve()
    if mode not in {"shared60", "family35x3"}:
        raise ValueError("mode harus shared60/family35x3")

    class GeometryFactorizationTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                GeometryFactorizedDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    geometry_factorization=frozen,
                    mode=mode,
                    class_names=self.data["names"],
                )
            )
            if not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_geometry_factorized_weights(
                    model, YOLO(str(bound)).model
                )
            elif weights:
                transfer = load_geometry_factorized_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"GEO-FACT {mode} HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    GeometryFactorizationTrainer.__name__ = (
        "GEOShared60Trainer" if mode == "shared60" else "GEOFamily35x3Trainer"
    )
    return GeometryFactorizationTrainer
