from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import SFRSpatialConfig, SFRSpatialDetectionModel, load_sfr_spatial_weights


def make_sfr_spatial_trainer(
    config: SFRSpatialConfig | dict[str, Any], *, d0_checkpoint: str | Path | None = None
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = SFRSpatialConfig.from_mapping(config)
    bound_checkpoint = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class SFRSpatialTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                SFRSpatialDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    sfr_spatial=frozen,
                )
            )
            if bound_checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO
                source = YOLO(str(bound_checkpoint)).model
                transfer = load_sfr_spatial_weights(model, source)
            elif weights:
                transfer = load_sfr_spatial_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"SFR SPATIAL NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    SFRSpatialTrainer.__name__ = "SFRSpatialTrainer"
    return SFRSpatialTrainer
