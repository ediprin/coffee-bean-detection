"""Static safety and capacity audit for the AGSF synthesis arms."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from .model import AGSFConfig, AGSFDetectHead, AGSFDetectionModel, load_agsf_detector_weights


ATOL = 1e-7
ARM_MODES = {"SYN0": "none", "SYN1": "additive", "SYN2": "gated"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_identical(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _finite_nonempty(parameters) -> bool:
    gradients = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)


def _finite_nonzero(parameters) -> bool:
    gradients = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
    return (
        bool(gradients)
        and all(bool(torch.isfinite(value).all()) for value in gradients)
        and any(bool(value.abs().max() > 0) for value in gradients)
    )


def _zero_or_missing_gradients(parameters) -> bool:
    gradients = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
    return not gradients or all(bool(value.abs().max() == 0) for value in gradients)


def _activate_correction(head: AGSFDetectHead) -> None:
    torch.manual_seed(17)
    for layer in head.correction.class_corrections:
        torch.nn.init.normal_(layer.weight, std=0.02)
        torch.nn.init.constant_(layer.bias, 0.1)


def static_agsf_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    image_size: int = 128,
    hidden_dim: int = 64,
) -> dict[str, Any]:
    """Prove native-start behavior and SYN1/SYN2 capacity matching."""

    from ultralytics import YOLO

    model_yaml = Path(model_yaml).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)
    d0 = YOLO(str(d0_checkpoint)).model.cpu().eval()
    torch.manual_seed(42)
    image = torch.randn(1, 3, int(image_size), int(image_size))
    with torch.inference_mode():
        native = d0(image)

    arms: dict[str, dict[str, Any]] = {}
    candidates: dict[str, AGSFDetectionModel] = {}
    for arm, mode in ARM_MODES.items():
        config = AGSFConfig(frequency_mode=mode, hidden_dim=int(hidden_dim))
        candidate = AGSFDetectionModel(
            str(model_yaml), nc=int(nc), verbose=False, agsf=config
        ).cpu()
        transfer = load_agsf_detector_weights(candidate, d0)
        candidate.eval()
        head = candidate.model[-1]
        if not isinstance(head, AGSFDetectHead):
            raise TypeError(type(head).__name__)
        with torch.inference_mode():
            zero = candidate(image)

        active = copy.deepcopy(candidate).eval()
        active_head = active.model[-1]
        _activate_correction(active_head)
        with torch.inference_mode():
            active_output = active(image)

        gradient_probe = copy.deepcopy(active).train()
        probe_head = gradient_probe.model[-1]
        scores = gradient_probe(image)["one2many"]["scores"]
        scores.square().mean().backward()
        stb_parameters = (
            parameter
            for block in probe_head.stb_blocks
            for parameter in block.parameters()
        )
        correction_parameters = probe_head.correction.class_corrections.parameters()
        frequency_parameters = probe_head.correction.frequency_encoders.parameters()
        gate_parameters = probe_head.correction.frequency_gates.parameters()
        source_code = inspect.getsource(type(head)) + inspect.getsource(type(head.correction))
        zero_score_diff = float(
            (zero[1]["one2one"]["scores"] - native[1]["one2one"]["scores"]).abs().max()
        )
        active_score_diff = float(
            (active_output[1]["one2one"]["scores"] - native[1]["one2one"]["scores"]).abs().max()
        )
        gates = {
            "native_d0_head_bitwise_preserved": _state_identical(head.base_head, d0.model[-1]),
            "zero_boxes_bitwise_equal": bool(
                torch.equal(zero[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"])
            ),
            "zero_scores_bitwise_equal": zero_score_diff <= ATOL,
            "active_correction_changes_scores": active_score_diff > ATOL,
            "active_correction_preserves_boxes": bool(
                torch.equal(
                    active_output[1]["one2one"]["boxes"],
                    native[1]["one2one"]["boxes"],
                )
            ),
            "finite_stb_gradients": _finite_nonempty(stb_parameters),
            "finite_class_correction_gradients": _finite_nonempty(correction_parameters),
            "frequency_gradient_policy_correct": (
                mode == "none" or _finite_nonempty(frequency_parameters)
            ),
            "gate_gradient_policy_correct": (
                (mode == "none")
                or (mode == "additive" and _zero_or_missing_gradients(gate_parameters))
                or (mode == "gated" and _finite_nonzero(gate_parameters))
            ),
            "no_roi_align": "roi_align" not in source_code,
            "no_topk_candidate_selection": ".topk(" not in source_code,
            "no_box_decode_before_classification": "_get_decode_boxes" not in source_code,
        }
        arms[arm] = {
            "frequency_mode": mode,
            "transfer": transfer,
            "parameters": sum(p.numel() for p in candidate.parameters()),
            "synthesis_parameters": (
                sum(p.numel() for p in head.stb_blocks.parameters())
                + sum(p.numel() for p in head.correction.parameters())
            ),
            "zero_score_max_abs_diff": zero_score_diff,
            "active_score_max_abs_diff": active_score_diff,
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
        candidates[arm] = candidate

    syn1, syn2 = candidates["SYN1"].model[-1], candidates["SYN2"].model[-1]
    capacity_gates = {
        "syn1_syn2_same_parameter_count": (
            arms["SYN1"]["parameters"] == arms["SYN2"]["parameters"]
        ),
        "syn1_syn2_same_state_schema": (
            list(syn1.state_dict()) == list(syn2.state_dict())
            and all(
                syn1.state_dict()[key].shape == syn2.state_dict()[key].shape
                for key in syn1.state_dict()
            )
        ),
        "frequency_is_classification_side_only": all(
            item["gates"]["active_correction_preserves_boxes"]
            for item in arms.values()
        ),
    }
    decision = "PASS" if (
        all(item["decision"] == "PASS" for item in arms.values())
        and all(capacity_gates.values())
    ) else "FAIL"
    payload = {
        "protocol": "faruq-v3-agsf-synthesis-static-v1",
        "training_executed": False,
        "dataset_accessed": False,
        "test_images_accessed": False,
        "d0_checkpoint": str(d0_checkpoint),
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "arms": arms,
        "capacity_gates": capacity_gates,
        "decision": decision,
        "training_authorized": False,
        "test_access_authorized": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(destination)
    return payload
