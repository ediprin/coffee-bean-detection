from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .operator import AFABConfig, AFABInputEnhancer


try:
    from ultralytics.nn.tasks import DetectionModel
except ImportError:  # pragma: no cover
    DetectionModel = nn.Module  # type: ignore


class AFABDetectionModel(DetectionModel):
    """Native YOLO26 with deterministic LFDet-AFAB preprocessing on every tensor forward.

    The AFAB module is attached only after DetectionModel finishes its native
    stride-initialization forward pass. Thereafter BaseModel.loss -> forward(img)
    -> predict(img) and ordinary inference both pass through the same AFAB input
    operator, matching AFAB's inference-time role in LFDet.
    """

    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, afab=None):
        frozen = AFABConfig.from_mapping(afab)
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.afab_config = frozen
        self.afab = AFABInputEnhancer(frozen)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "afab", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )


def load_afab_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load native/AFAB checkpoint while retaining the candidate AFAB config."""
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
    # AFAB has no persistent buffers/parameters, so a native D0 source should
    # cover the complete learnable detector state.
    return {
        "source_items": len(source_state),
        "target_items": len(target_state),
        "shape_compatible_items": len(common),
    }
