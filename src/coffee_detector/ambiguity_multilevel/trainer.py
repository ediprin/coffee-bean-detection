from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import (
    AmbiguityMultilevelConfig,
    AmbiguityMultilevelDetectionModel,
    load_ambiguity_multilevel_detector_weights,
)


def make_ambiguity_multilevel_trainer(
    config: AmbiguityMultilevelConfig | dict[str, Any],
    *,
    d0_checkpoint: str | Path | None = None,
):
    """Use the native YOLO end-to-end detection loss; no auxiliary ROI loss."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = AmbiguityMultilevelConfig.from_mapping(config)
    bound_checkpoint = (
        Path(d0_checkpoint).expanduser().resolve()
        if d0_checkpoint is not None
        else None
    )

    class AmbiguityMultilevelTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AmbiguityMultilevelDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    ambiguity_multilevel=frozen,
                )
            )
            # Ultralytics may forward the original 80-class pretrained object
            # here even though the outer YOLO instance was explicitly loaded
            # from D0.  ACMC is defined against the audited 21-class D0 file,
            # so a fresh run must bind that exact file directly.  A resumed
            # run instead receives its serialized ACMC checkpoint in weights.
            if bound_checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                source = YOLO(str(bound_checkpoint)).model
                transfer = load_ambiguity_multilevel_detector_weights(model, source)
            elif weights:
                transfer = load_ambiguity_multilevel_detector_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"ACMC NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    AmbiguityMultilevelTrainer.__name__ = "AmbiguityMultilevelTrainer"
    return AmbiguityMultilevelTrainer
