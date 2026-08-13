from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from coffee_detector.stb import STBConfig, STBDetectionModel, load_stb_weights

from .model import (
    ClassificationChannelControl,
    STBCapacityControlDetectionModel,
    load_stb_control_weights,
)


def static_stb_capacity_control_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    image_size: int = 128,
) -> dict:
    from ultralytics import YOLO

    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    source = YOLO(str(d0_checkpoint)).model.eval()
    config = STBConfig()
    stb = STBDetectionModel(
        str(Path(model_yaml).resolve()), nc=nc, verbose=False, stb=config
    ).eval()
    control = STBCapacityControlDetectionModel(
        str(Path(model_yaml).resolve()), nc=nc, verbose=False, stb=config
    ).eval()
    load_stb_weights(stb, source)
    load_stb_control_weights(control, source)

    image = torch.rand(1, 3, image_size, image_size)
    with torch.inference_mode():
        native, stb_zero, control_zero = source(image), stb(image), control(image)
    identity = {
        "stb_boxes": torch.equal(native[1]["one2one"]["boxes"], stb_zero[1]["one2one"]["boxes"]),
        "stb_scores": torch.equal(native[1]["one2one"]["scores"], stb_zero[1]["one2one"]["scores"]),
        "control_boxes": torch.equal(native[1]["one2one"]["boxes"], control_zero[1]["one2one"]["boxes"]),
        "control_scores": torch.equal(native[1]["one2one"]["scores"], control_zero[1]["one2one"]["scores"]),
    }
    with torch.no_grad():
        for model in (stb, control):
            for block in model.model[-1].blocks:
                block.gate.fill_(0.1)
    with torch.inference_mode():
        stb_active, control_active = stb(image), control(image)
    active = {
        "stb_boxes_preserved": torch.equal(stb_zero[1]["one2one"]["boxes"], stb_active[1]["one2one"]["boxes"]),
        "stb_scores_changed": not torch.equal(stb_zero[1]["one2one"]["scores"], stb_active[1]["one2one"]["scores"]),
        "control_boxes_preserved": torch.equal(control_zero[1]["one2one"]["boxes"], control_active[1]["one2one"]["boxes"]),
        "control_scores_changed": not torch.equal(control_zero[1]["one2one"]["scores"], control_active[1]["one2one"]["scores"]),
    }
    stb_params = sum(parameter.numel() for parameter in stb.parameters())
    control_params = sum(parameter.numel() for parameter in control.parameters())
    relative_gap = abs(control_params - stb_params) / stb_params
    control_source = inspect.getsource(ClassificationChannelControl).lower()
    gates = {
        **identity,
        **active,
        "same_three_pyramid_levels": len(stb.model[-1].blocks) == len(control.model[-1].blocks) == 3,
        "same_two_block_depth": all(len(block.blocks) == 2 for block in control.model[-1].blocks),
        "parameter_gap_below_0_05_percent": relative_gap <= 0.0005,
        "control_has_no_spatial_attention": "attention" not in control_source,
        "control_has_no_spatial_convolution": "conv" not in control_source,
    }
    payload = {
        "protocol": "faruq-v3-stb-capacity-causal-control-static-v1",
        "models": {
            "STB1": {"parameters": stb_params},
            "CMC0": {"parameters": control_params},
        },
        "parameter_difference": control_params - stb_params,
        "parameter_relative_gap": relative_gap,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "training_executed": False,
        "test_images_accessed": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(output)
    return payload
