from __future__ import annotations

from typing import Any

import torch
from torch import nn

from coffee_detector.afab import AFABConfig

from .operator import AF2ChannelCalibratedEnhancer


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class AF2CalibratedDetectionModel(DetectionModel):
    """Native YOLO26 with an end-to-end three-parameter AF2 calibrator."""

    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, afab=None):
        frozen = AFABConfig.from_mapping(afab)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.afab_config = frozen
        self.af2cal = AF2ChannelCalibratedEnhancer(frozen)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "af2cal", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def load_af2cal_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Transfer every AF2 detector tensor and retain zero calibration logits."""

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
        "new_calibration_items": len(target_state) - len(common),
    }
