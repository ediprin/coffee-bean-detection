from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import torch

from .model import (
    FocalModulationConfig,
    FocalModulationDetectionModel,
    FocalModulationDetectHead,
    load_focal_modulation_weights,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def static_focal_modulation_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    image_size: int = 128,
) -> dict:
    """Prove identity start and classification-only wiring before training."""

    from ultralytics import YOLO

    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)
    source = YOLO(str(d0_checkpoint)).model.eval()
    candidate = FocalModulationDetectionModel(
        str(Path(model_yaml).resolve()),
        nc=nc,
        verbose=False,
        focal_modulation=FocalModulationConfig(),
    ).eval()
    transfer = load_focal_modulation_weights(candidate, source)
    head = candidate.model[-1]
    if not isinstance(head, FocalModulationDetectHead):
        raise TypeError("Head FMH1 tidak terpasang")

    image = torch.rand(1, 3, image_size, image_size)
    with torch.inference_mode():
        native = source(image)
        zero = candidate(image)
    zero_boxes = torch.equal(native[1]["one2one"]["boxes"], zero[1]["one2one"]["boxes"])
    zero_scores = torch.equal(native[1]["one2one"]["scores"], zero[1]["one2one"]["scores"])
    zero_output = torch.equal(native[0], zero[0])

    with torch.no_grad():
        for block in head.blocks:
            block.gate.fill_(0.1)
    with torch.inference_mode():
        active = candidate(image)
    active_box_diff = float(
        (zero[1]["one2one"]["boxes"] - active[1]["one2one"]["boxes"]).abs().max()
    )
    active_score_diff = float(
        (zero[1]["one2one"]["scores"] - active[1]["one2one"]["scores"]).abs().max()
    )

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    training_output = candidate(image)
    loss = training_output["one2many"]["scores"].square().mean()
    loss.backward()
    focal_gradients = [
        parameter.grad
        for name, parameter in candidate.named_parameters()
        if ".blocks." in name and "gate" not in name
    ]
    finite_gradients = any(
        gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
        for gradient in focal_gradients
    )

    source_text = inspect.getsource(FocalModulationDetectHead).lower()
    kernels = [
        int(layer[0].kernel_size[0])
        for layer in head.blocks[0].layers[0].modulation.focal_layers
    ]
    gates = {
        "native_d0_head_strictly_transferred": transfer["native_head_items"] > 0,
        "zero_output_is_d0": zero_output,
        "zero_boxes_bitwise_equal": zero_boxes,
        "zero_scores_bitwise_equal": zero_scores,
        "active_modulation_changes_scores": active_score_diff > 0.0,
        "active_modulation_preserves_boxes": active_box_diff == 0.0,
        "finite_focal_gradients": bool(finite_gradients),
        "classification_only_p3_p4_p5": len(head.blocks) == 3,
        "paper_default_nested_kernels": kernels == [3, 5],
        "no_roi_align": "roi_align" not in source_text,
        "no_topk_candidate_selection": "topk" not in source_text,
        "no_box_decode_before_classification": "decode" not in source_text,
    }
    payload = {
        "protocol": "faruq-v3-fmh1-static-audit-v1",
        "reference_implementation": "microsoft/FocalNet classification/focalnet.py",
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "config": FocalModulationConfig().to_dict(),
        "parameters": {
            "total": sum(parameter.numel() for parameter in candidate.parameters()),
            "focal_modulation": sum(parameter.numel() for block in head.blocks for parameter in block.parameters()),
        },
        "nested_kernel_sizes": kernels,
        "zero_max_box_diff": 0.0 if zero_boxes else float("nan"),
        "zero_max_score_diff": 0.0 if zero_scores else float("nan"),
        "active_max_box_diff": active_box_diff,
        "active_max_score_diff": active_score_diff,
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
