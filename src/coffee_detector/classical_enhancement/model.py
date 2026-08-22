from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .operator import CLAHEConfig, CLAHEInputEnhancer

try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class CLAHEDetectionModel(DetectionModel):
    """Native YOLO26 with deterministic LAB-luminance CLAHE before prediction."""

    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, clahe=None):
        frozen = CLAHEConfig.from_mapping(clahe)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.clahe_config = frozen
        self.clahe = CLAHEInputEnhancer(frozen)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "clahe", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def load_clahe_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load native/CLAHE checkpoint while preserving the frozen CLAHE config."""

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
