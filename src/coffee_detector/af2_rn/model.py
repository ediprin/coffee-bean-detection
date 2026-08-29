from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import AF2RNConfig
from .operator import AF2RNInputEnhancer

try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class AF2RNDetectionModel(DetectionModel):
    """Native detector with parameter-free AF2RN input preprocessing."""

    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, af2rn=None):
        frozen = AF2RNConfig.from_mapping(af2rn)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.af2rn_config = frozen
        self.af2rn = AF2RNInputEnhancer(frozen)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "af2rn", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def load_af2rn_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    source = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    source_state = source.state_dict() if isinstance(source, nn.Module) else None
    model.load(weights)
    target_state = model.state_dict()
    if source_state is None:
        return {"source_items": 0, "target_items": len(target_state)}
    common = {
        key for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    return {
        "source_items": len(source_state),
        "target_items": len(target_state),
        "shape_compatible_items": len(common),
    }
