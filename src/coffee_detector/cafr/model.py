from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import CAFRConfig
from .operator import CAFRInputEnhancer


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class CAFRDetectionModel(DetectionModel):
    """Native YOLO26 with parameter-free CAFR preprocessing on train and inference."""

    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, cafr=None):
        frozen = CAFRConfig.from_mapping(cafr)
        # Attach CAFR only after native DetectionModel initialization so YOLO's internal
        # stride-probing forward remains untouched.
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.cafr_config = frozen
        self.cafr = CAFRInputEnhancer(frozen)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "cafr", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def load_cafr_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load a native/CAFR checkpoint without changing the candidate CAFR config."""

    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    source_state = source.state_dict() if isinstance(source, nn.Module) else None
    model.load(weights)
    target_state = model.state_dict()
    if source_state is None:
        return {"source_items": 0, "target_items": len(target_state)}
    common = {
        key
        for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    return {
        "source_items": len(source_state),
        "target_items": len(target_state),
        "shape_compatible_items": len(common),
    }


def make_cafr_trainer(
    config: CAFRConfig | dict,
    *,
    d0_checkpoint: str | Path | None = None,
):
    """Build a standard Ultralytics detection trainer whose only graph change is CAFR."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = CAFRConfig.from_mapping(config)
    checkpoint = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class CAFRTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                CAFRDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    cafr=frozen,
                )
            )
            if checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_cafr_weights(model, YOLO(str(checkpoint)).model)
            elif weights:
                transfer = load_cafr_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"CAFR DETECTOR WEIGHT TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    CAFRTrainer.__name__ = "CAFRTrainer"
    return CAFRTrainer
