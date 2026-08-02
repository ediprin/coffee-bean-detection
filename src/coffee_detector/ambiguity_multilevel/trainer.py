from __future__ import annotations

from typing import Any

from .model import (
    AmbiguityMultilevelConfig,
    AmbiguityMultilevelDetectionModel,
    load_ambiguity_multilevel_detector_weights,
)


def make_ambiguity_multilevel_trainer(
    config: AmbiguityMultilevelConfig | dict[str, Any],
):
    """Use the native YOLO end-to-end detection loss; no auxiliary ROI loss."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = AmbiguityMultilevelConfig.from_mapping(config)

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
            if weights:
                transfer = load_ambiguity_multilevel_detector_weights(model, weights)
                print(f"ACMC NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    AmbiguityMultilevelTrainer.__name__ = "AmbiguityMultilevelTrainer"
    return AmbiguityMultilevelTrainer
