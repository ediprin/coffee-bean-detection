from __future__ import annotations

from typing import Any

import torch
from torch import nn

from coffee_detector.afab import AFABConfig

from .config import AF2RConfig
from .operator import AF2ResidualGateEnhancer


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class AF2RDetectionModel(DetectionModel):
    """Native YOLO26 with an end-to-end raw-preserving AF2 front end."""

    def __init__(
        self,
        cfg="yolo26.yaml",
        ch=3,
        nc=None,
        verbose=True,
        afab=None,
        af2r=None,
    ):
        frozen_afab = AFABConfig.from_mapping(afab)
        frozen_af2r = AF2RConfig.from_mapping(af2r)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.afab_config = frozen_afab
        self.af2r_config = frozen_af2r
        self.af2r = AF2ResidualGateEnhancer(frozen_afab, frozen_af2r)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "af2r", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def load_af2r_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Transfer every compatible AF2 detector tensor; keep the new gate initialized."""

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
        "new_gate_items": len(target_state) - len(common),
    }
