from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import WAV1FactorizationConfig
from .operator import WAV1FactorizationEnhancer


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class WAV1FactorizationDetectionModel(DetectionModel):
    """Native YOLO26 with one frozen parameter-free cue frontend."""

    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, factorization=None):
        frozen = WAV1FactorizationConfig.from_mapping(factorization)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.factorization_config = frozen
        self.factorization = WAV1FactorizationEnhancer(frozen)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "factorization", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def load_factorization_weights(model: nn.Module, weights: Any) -> dict[str, int]:
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


def make_factorization_trainer(
    config: WAV1FactorizationConfig | dict,
    *,
    d0_checkpoint: str | Path | None = None,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = WAV1FactorizationConfig.from_mapping(config)
    checkpoint = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class WAV1FactorizationTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                WAV1FactorizationDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    factorization=frozen,
                )
            )
            if checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_factorization_weights(model, YOLO(str(checkpoint)).model)
            elif weights:
                transfer = load_factorization_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"WAV1 FACTORIZATION WEIGHT TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    WAV1FactorizationTrainer.__name__ = "WAV1FactorizationTrainer"
    return WAV1FactorizationTrainer
