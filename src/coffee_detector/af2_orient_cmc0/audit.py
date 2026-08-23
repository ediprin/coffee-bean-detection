from __future__ import annotations

import json
from pathlib import Path

import torch

from coffee_detector.af2_iso import frozen_arm_config
from coffee_detector.af2_iso.model import AF2IsolatedDetectionModel, load_af2_iso_weights
from coffee_detector.stb import STBConfig

from .model import AF2OrientCMC0DetectionModel, load_af2_orient_cmc0_weights


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def static_af2_orient_cmc0_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    image_size: int = 128,
    tolerance: float = 1e-4,
) -> dict:
    """Verify AF2_ORIENT + CMC0 before any training.

    The relevant identity is conditional on AF2_ORIENT: with all CMC0 residual
    gates at zero, the combined model must reproduce AF2_ORIENT initialized from
    the same seed-matched D0 checkpoint. Activating only CMC0 must preserve the
    localization tensors while changing classification scores. Full-model
    comparisons use a small numerical tolerance because AF2 includes FFT/IFFT
    operations; this mirrors the repository's established AF2 composition audits.
    """

    from ultralytics import YOLO

    model_yaml = Path(model_yaml).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not model_yaml.is_file():
        raise FileNotFoundError(model_yaml)
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)

    source = YOLO(str(d0_checkpoint)).model.eval().cpu()
    source_head = source.model[-1]
    checkpoint_nc = int(getattr(source_head, "nc", nc))
    if checkpoint_nc != int(nc):
        raise RuntimeError(f"Checkpoint nc={checkpoint_nc}, expected nc={nc}")

    af2_config = frozen_arm_config("AF2_ORIENT")
    stb_config = STBConfig()

    af2 = AF2IsolatedDetectionModel(
        str(model_yaml), nc=nc, verbose=False, af2_iso=af2_config
    ).eval().cpu()
    combined = AF2OrientCMC0DetectionModel(
        str(model_yaml), nc=nc, verbose=False, af2_iso=af2_config, stb=stb_config
    ).eval().cpu()

    af2_transfer = load_af2_iso_weights(af2, source)
    combined_transfer = load_af2_orient_cmc0_weights(combined, source)

    torch.manual_seed(42)
    image = torch.rand(1, 3, image_size, image_size)
    with torch.inference_mode():
        enhanced = af2.af2_iso(image)
        af2_zero = af2(image)
        combined_zero = combined(image)

    af2_boxes = af2_zero[1]["one2one"]["boxes"]
    af2_scores = af2_zero[1]["one2one"]["scores"]
    combined_boxes = combined_zero[1]["one2one"]["boxes"]
    combined_scores = combined_zero[1]["one2one"]["scores"]
    zero_box_diff = _max_abs(af2_boxes, combined_boxes)
    zero_score_diff = _max_abs(af2_scores, combined_scores)

    gates_before = [float(block.gate.detach().cpu()) for block in combined.model[-1].blocks]
    with torch.no_grad():
        for block in combined.model[-1].blocks:
            block.gate.fill_(0.1)
    with torch.inference_mode():
        combined_active = combined(image)

    active_boxes = combined_active[1]["one2one"]["boxes"]
    active_scores = combined_active[1]["one2one"]["scores"]
    active_box_diff = _max_abs(combined_boxes, active_boxes)
    active_score_diff = _max_abs(combined_scores, active_scores)

    checks = {
        "af2_orient_changes_input": not torch.equal(image, enhanced),
        "cmc0_gates_zero_initialized": all(value == 0.0 for value in gates_before),
        "conditional_identity_boxes": zero_box_diff <= tolerance,
        "conditional_identity_scores": zero_score_diff <= tolerance,
        "cmc0_active_preserves_boxes": active_box_diff <= tolerance,
        "cmc0_active_changes_scores": active_score_diff > tolerance,
        "three_classification_levels": len(combined.model[-1].blocks) == 3,
    }

    payload = {
        "format": "coffee_detector.af2_orient_cmc0.static_audit.v1",
        "checkpoint": str(d0_checkpoint),
        "checkpoint_nc": checkpoint_nc,
        "af2_operator": af2_config.to_dict(),
        "cmc0": stb_config.to_dict(),
        "transfer": {"af2_orient": af2_transfer, "combined": combined_transfer},
        "numerics": {
            "zero_gate_box_max_abs_diff": zero_box_diff,
            "zero_gate_score_max_abs_diff": zero_score_diff,
            "active_gate_box_max_abs_diff": active_box_diff,
            "active_gate_score_max_abs_diff": active_score_diff,
            "tolerance": tolerance,
        },
        "checks": checks,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "training_executed": False,
        "test_images_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["summary"] = str(output)
    return payload
