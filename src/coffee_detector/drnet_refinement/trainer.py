from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import (
    DRNetRefinementConfig,
    DRNetRefinementDetectionModel,
    load_drnet_refinement_weights,
)


def make_drnet_refinement_trainer(
    config: DRNetRefinementConfig | dict[str, Any],
    *,
    d0_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = DRNetRefinementConfig.from_mapping(config)
    bound_checkpoint = (
        Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint is not None else None
    )

    class DRNetRefinementTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                DRNetRefinementDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    drnet_refinement=frozen,
                )
            )
            if bound_checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(bound_checkpoint)).model
                transfer = load_drnet_refinement_weights(model, source)
            elif weights:
                transfer = load_drnet_refinement_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"DRNET NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    suffix = "CML" if frozen.use_cml else "FGB"
    DRNetRefinementTrainer.__name__ = f"DRNetRefinementTrainer{suffix}"
    return DRNetRefinementTrainer
