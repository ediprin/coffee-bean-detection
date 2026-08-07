from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import (
    SemanticGuidedConfig,
    SemanticGuidedDetectionModel,
    load_semantic_guided_weights,
)


def make_semantic_guided_trainer(
    config: SemanticGuidedConfig | dict[str, Any], *, d0_checkpoint: str | Path | None = None
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = SemanticGuidedConfig.from_mapping(config)
    bound = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class SemanticGuidedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                SemanticGuidedDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    semantic_guided=frozen,
                )
            )
            if bound is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO
                transfer = load_semantic_guided_weights(model, YOLO(str(bound)).model)
            elif weights:
                transfer = load_semantic_guided_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"SEMANTIC GUIDED NATIVE HEAD TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    SemanticGuidedTrainer.__name__ = "SemanticGuidedTrainer"
    return SemanticGuidedTrainer
