from __future__ import annotations

import json
from pathlib import Path

import torch

from coffee_detector.stb import STBConfig
from coffee_detector.stb.model import STBDetectionModel, load_stb_weights
from coffee_detector.stb_control.model import (
    STBCapacityControlDetectionModel,
    load_stb_control_weights,
)

from .model import STBSR1DetectionModel, load_stb_sr1_weights


def _params(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _one2one(output):
    return output[1]["one2one"]


def static_stb_sr1_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    image_size: int = 128,
) -> dict:
    """Verify identity start, classification-only wiring, and branch structure."""

    from ultralytics import YOLO

    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    source = YOLO(str(d0_checkpoint)).model.eval()
    config = STBConfig()
    stb = STBDetectionModel(str(Path(model_yaml).resolve()), nc=nc, verbose=False, stb=config).eval()
    cmc = STBCapacityControlDetectionModel(
        str(Path(model_yaml).resolve()), nc=nc, verbose=False, stb=config
    ).eval()
    sr1 = STBSR1DetectionModel(
        str(Path(model_yaml).resolve()), nc=nc, verbose=False, stb=config
    ).eval()
    load_stb_weights(stb, source)
    load_stb_control_weights(cmc, source)
    load_stb_sr1_weights(sr1, source)

    generator = torch.Generator().manual_seed(42)
    image = torch.rand(1, 3, image_size, image_size, generator=generator)
    with torch.inference_mode():
        native = source(image)
        zero = sr1(image)
    native_o2o, zero_o2o = _one2one(native), _one2one(zero)
    identity = {
        "zero_gate_boxes_bitwise_native": torch.equal(native_o2o["boxes"], zero_o2o["boxes"]),
        "zero_gate_scores_bitwise_native": torch.equal(native_o2o["scores"], zero_o2o["scores"]),
    }

    with torch.no_grad():
        for block in sr1.model[-1].blocks:
            block.channel_gate.fill_(0.1)
            block.spatial_gate.zero_()
    with torch.inference_mode():
        channel_active = sr1(image)
    channel_o2o = _one2one(channel_active)
    channel_checks = {
        "channel_gate_preserves_boxes": torch.equal(zero_o2o["boxes"], channel_o2o["boxes"]),
        "channel_gate_changes_scores": not torch.equal(zero_o2o["scores"], channel_o2o["scores"]),
    }

    with torch.no_grad():
        for block in sr1.model[-1].blocks:
            block.channel_gate.zero_()
            block.spatial_gate.fill_(0.1)
    with torch.inference_mode():
        spatial_active = sr1(image)
    spatial_o2o = _one2one(spatial_active)
    spatial_checks = {
        "spatial_gate_preserves_boxes": torch.equal(zero_o2o["boxes"], spatial_o2o["boxes"]),
        "spatial_gate_changes_scores": not torch.equal(zero_o2o["scores"], spatial_o2o["scores"]),
    }

    blocks = list(sr1.model[-1].blocks)
    spatial_module_names = [
        type(module).__name__.lower()
        for block in blocks
        for branch in (block.wmsa, block.swmsa)
        for module in branch.modules()
    ]
    structure = {
        "three_pyramid_levels": len(blocks) == 3,
        "each_level_has_two_cmc_blocks": all(len(block.channel_base.blocks) == 2 for block in blocks),
        "each_level_has_wmsa_and_swmsa": all(
            block.wmsa.attn.shift_size == [0, 0]
            and block.swmsa.attn.shift_size == [config.window_size // 2] * 2
            for block in blocks
        ),
        "spatial_branch_contains_no_mlp_module": not any("mlp" in name for name in spatial_module_names),
        "separate_channel_and_spatial_scalar_gates": all(
            block.channel_gate.ndim == 0 and block.spatial_gate.ndim == 0 for block in blocks
        ),
    }

    params = {
        "STB1": _params(stb),
        "CMC0": _params(cmc),
        "STBSR1": _params(sr1),
    }
    gates = {**identity, **channel_checks, **spatial_checks, **structure}
    payload = {
        "protocol": "faruq-v3-stb-sr1-static-v1",
        "question": "Can CMC-style channel representation plus attention-only spatial residual improve the STB family without altering native localization?",
        "models": {name: {"parameters": value} for name, value in params.items()},
        "overhead_vs_cmc0": params["STBSR1"] - params["CMC0"],
        "overhead_vs_stb1": params["STBSR1"] - params["STB1"],
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "training_executed": False,
        "test_images_accessed": False,
        "test_opened": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(output)
    return payload
