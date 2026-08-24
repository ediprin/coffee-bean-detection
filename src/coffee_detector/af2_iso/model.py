from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import AF2IsolatedConfig
from .operator import AF2IsolatedInputEnhancer


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class AF2IsolatedDetectionModel(DetectionModel):
    """Native YOLO26 with one parameter-free isolated AF2 frontend change."""

    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, af2_iso=None):
        frozen = AF2IsolatedConfig.from_mapping(af2_iso)
        # Attach after native initialization so stride probing remains untouched.
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.af2_iso_config = frozen
        self.af2_iso = AF2IsolatedInputEnhancer(frozen)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "af2_iso", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def load_af2_iso_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load native checkpoint weights without changing the isolated frontend config."""

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


def make_af2_iso_trainer(
    config: AF2IsolatedConfig | dict,
    *,
    d0_checkpoint: str | Path | None = None,
):
    """Build Ultralytics trainer whose only graph change is AF2 geometry."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = AF2IsolatedConfig.from_mapping(config)
    checkpoint = Path(d0_checkpoint).expanduser().resolve() if d0_checkpoint else None

    class AF2IsolatedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                AF2IsolatedDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    af2_iso=frozen,
                )
            )
            if checkpoint is not None and not getattr(self.args, "resume", False):
                from ultralytics import YOLO

                transfer = load_af2_iso_weights(model, YOLO(str(checkpoint)).model)
            elif weights:
                transfer = load_af2_iso_weights(model, weights)
            else:
                transfer = None
            if transfer is not None:
                print(f"AF2-ISO DETECTOR WEIGHT TRANSFER: {transfer}", flush=True)
            return model

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    AF2IsolatedTrainer.__name__ = "AF2IsolatedTrainer"
    return AF2IsolatedTrainer
