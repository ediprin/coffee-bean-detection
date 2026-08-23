from __future__ import annotations

from typing import Any
import torch
from torch import nn

from ultralytics.nn.tasks import DetectionModel

from coffee_detector.af2_iso.config import AF2IsolatedConfig
from coffee_detector.af2_iso.operator import AF2IsolatedInputEnhancer

from coffee_detector.stb.model import STBConfig
from coffee_detector.stb_control.model import STBCapacityControlDetectHead


class AF2OrientCMC0DetectionModel(DetectionModel):
    """Native YOLO26 with AF2_ORIENT input enhancement and CMC0 capacity head."""

    def __init__(self, cfg="yolo26.yaml", ch=3, nc=None, verbose=True, af2_iso=None, stb=None):
        frozen_af2 = AF2IsolatedConfig.from_mapping(af2_iso)
        frozen_stb = STBConfig.from_mapping(stb)
        
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        
        # 1. AF2 Input Enhancer
        self.af2_iso_config = frozen_af2
        self.af2_iso = AF2IsolatedInputEnhancer(frozen_af2)
        
        # 2. CMC0 Head 
        self.stb_config = frozen_stb
        self.model[-1] = STBCapacityControlDetectHead(
            self.model[-1], self.stb_config
        )

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        enhancer = getattr(self, "af2_iso", None)
        if enhancer is not None and isinstance(x, torch.Tensor):
            x = enhancer(x)
        return super().predict(
            x, profile=profile, visualize=visualize, augment=augment, embed=embed
        )

def load_af2_orient_cmc0_weights(model: nn.Module, weights: Any) -> dict[str, int]:
    """Load weights handling both AF2 (unaltered parameters) and CMC0 (head strict transfer)."""
    model.load(weights)
    source = weights.model if hasattr(weights, "model") else weights
    target = model.model if hasattr(model, "model") else model
    
    if not isinstance(source, (nn.Sequential, nn.ModuleList)):
        return {"resume": 0}
        
    source_head = source[-1]
    target_head = target[-1]
    
    if not isinstance(target_head, STBCapacityControlDetectHead):
        raise TypeError("Target is not STBCapacityControlDetectHead")
        
    if isinstance(source_head, STBCapacityControlDetectHead):
        target_head.load_state_dict(source_head.state_dict(), strict=True)
        return {"native_head_items": len(target_head.base_head.state_dict()), "resume": 1}
        
    if type(source_head).__name__ != "Detect":
        raise TypeError(f"Control must start from native D0, not {type(source_head).__name__}")
        
    result = target_head.base_head.load_state_dict(source_head.state_dict(), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Incomplete native Detect transfer")
        
    target_head.stride = source_head.stride.detach().clone()
    target_head.base_head.stride = target_head.stride
    for name in ("max_det", "export", "format", "dynamic", "agnostic_nms"):
        if hasattr(source_head, name):
            value = getattr(source_head, name)
            setattr(target_head, name, value)
            setattr(target_head.base_head, name, value)
            
    return {"native_head_items": len(source_head.state_dict()), "resume": 0}
