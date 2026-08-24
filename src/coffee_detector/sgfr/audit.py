"""Static safety gate for SGFR staged frozen-residual synthesis."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from coffee_detector.stb.model import STBDetectHead

from .model import SGFRConfig, SGFRDetectHead, SGFRDetectionModel, load_sgfr_weights


ATOL = 1e-7
ARMS = {"SGC0": "control", "SGI1": "geometry", "SGF2": "frequency"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def _base_equal(candidate: SGFRDetectionModel, source: torch.nn.Module) -> bool:
    if len(candidate.model) != len(source.model):
        return False
    if not all(
        _module_equal(candidate.model[index], source.model[index])
        for index in range(len(candidate.model) - 1)
    ):
        return False
    head, source_head = candidate.model[-1], source.model[-1]
    return (
        isinstance(head, SGFRDetectHead)
        and isinstance(source_head, STBDetectHead)
        and _module_equal(head.base_head, source_head.base_head)
        and _module_equal(head.blocks, source_head.blocks)
    )


def _activate_geometry(head: SGFRDetectHead) -> None:
    torch.manual_seed(11)
    for level in head.geometry_levels:
        torch.nn.init.normal_(level.class_correction.weight, std=0.02)
        torch.nn.init.constant_(level.class_correction.bias, 0.05)


def _activate_frequency(head: SGFRDetectHead) -> None:
    torch.manual_seed(13)
    for level in head.frequency_levels:
        torch.nn.init.normal_(level.class_correction.weight, std=0.02)
        torch.nn.init.constant_(level.class_correction.bias, 0.05)


def _trainable_policy_valid(model: SGFRDetectionModel, stage: str) -> bool:
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if not names:
        return False
    if stage == "control":
        return all(
            "model.23.base_head" in name
            and (".cv3." in name or ".one2one_cv3." in name)
            for name in names
        )
    if stage == "geometry":
        return all("model.23.geometry_levels" in name for name in names)
    return all("model.23.frequency_levels" in name for name in names)


def static_sgfr_audit(
    model_yaml: str | Path,
    stb_checkpoint: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    image_size: int = 128,
) -> dict[str, Any]:
    from ultralytics import YOLO

    model_yaml = Path(model_yaml).expanduser().resolve()
    stb_checkpoint = Path(stb_checkpoint).expanduser().resolve()
    if not stb_checkpoint.is_file():
        raise FileNotFoundError(stb_checkpoint)
    source = YOLO(str(stb_checkpoint)).model.cpu().eval()
    if not isinstance(source.model[-1], STBDetectHead):
        raise TypeError("Static SGFR memerlukan checkpoint STB1")
    torch.manual_seed(42)
    image = torch.rand(1, 3, int(image_size), int(image_size))
    with torch.inference_mode():
        native = source(image)

    arms: dict[str, Any] = {}
    for code, stage in ARMS.items():
        candidate = SGFRDetectionModel(
            str(model_yaml), nc=int(nc), verbose=False, sgfr=SGFRConfig(stage=stage)
        ).cpu()
        transfer = load_sgfr_weights(candidate, source)
        candidate.eval()
        with torch.inference_mode():
            zero = candidate(image)

        active = copy.deepcopy(candidate).eval()
        active_head = active.model[-1]
        if stage == "geometry":
            _activate_geometry(active_head)
        elif stage == "frequency":
            _activate_frequency(active_head)
        with torch.inference_mode():
            active_output = active(image)

        candidate.apply_freeze_policy()
        candidate.train(True)
        head = candidate.model[-1]
        base_training_modes_frozen = (
            all(not layer.training for layer in list(candidate.model)[:-1])
            and not head.blocks.training
            and (
                stage == "control"
                or not head.base_head.training
            )
        )
        trainable = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
        source_code = inspect.getsource(SGFRDetectHead)
        zero_score_diff = float(
            (zero[1]["one2one"]["scores"] - native[1]["one2one"]["scores"])
            .abs()
            .max()
        )
        active_score_diff = float(
            (active_output[1]["one2one"]["scores"] - native[1]["one2one"]["scores"])
            .abs()
            .max()
        )
        gates = {
            "strict_stb_checkpoint_transfer": _base_equal(candidate, source),
            "zero_boxes_bitwise_equal": bool(
                torch.equal(zero[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"])
            ),
            "zero_scores_bitwise_equal": zero_score_diff <= ATOL,
            "active_residual_preserves_boxes": bool(
                torch.equal(
                    active_output[1]["one2one"]["boxes"],
                    native[1]["one2one"]["boxes"],
                )
            ),
            "active_residual_policy_correct": (
                active_score_diff <= ATOL if stage == "control" else active_score_diff > ATOL
            ),
            "trainable_parameter_policy_exact": _trainable_policy_valid(candidate, stage),
            "frozen_modules_forced_eval": base_training_modes_frozen,
            "has_trainable_parameters": trainable > 0,
            "no_roi_align": "roi_align" not in source_code,
            "no_topk_candidate_selection": ".topk(" not in source_code,
            "no_box_decode_before_classification": "_get_decode_boxes" not in source_code,
        }
        arms[code] = {
            "stage": stage,
            "transfer": transfer,
            "parameters": sum(p.numel() for p in candidate.parameters()),
            "trainable_parameters": trainable,
            "zero_score_max_abs_diff": zero_score_diff,
            "active_score_max_abs_diff": active_score_diff,
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }

    decision = "PASS" if all(value["decision"] == "PASS" for value in arms.values()) else "FAIL"
    payload = {
        "protocol": "faruq-v3-sgfr-static-v1",
        "training_executed": False,
        "dataset_accessed": False,
        "test_images_accessed": False,
        "stb_checkpoint": str(stb_checkpoint),
        "stb_checkpoint_sha256": _sha256(stb_checkpoint),
        "arms": arms,
        "decision": decision,
        "training_authorized": False,
        "test_access_authorized": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(destination)
    return payload


def audit_sgfr_checkpoint_invariance(
    source_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    *,
    stage: str,
) -> dict[str, Any]:
    """Verify that a completed stage changed only its authorized modules."""

    from ultralytics import YOLO

    if stage not in {"control", "geometry", "frequency"}:
        raise ValueError(stage)
    source_path = Path(source_checkpoint).expanduser().resolve()
    candidate_path = Path(candidate_checkpoint).expanduser().resolve()
    source = YOLO(str(source_path)).model.cpu().eval()
    candidate = YOLO(str(candidate_path)).model.cpu().eval()
    candidate_head = candidate.model[-1]
    if not isinstance(candidate_head, SGFRDetectHead):
        raise TypeError(type(candidate_head).__name__)
    source_head = source.model[-1]
    if isinstance(source_head, STBDetectHead):
        source_base, source_blocks = source_head.base_head, source_head.blocks
    elif isinstance(source_head, SGFRDetectHead):
        source_base, source_blocks = source_head.base_head, source_head.blocks
    else:
        raise TypeError(type(source_head).__name__)
    gates = {
        "backbone_neck_unchanged": (
            len(source.model) == len(candidate.model)
            and all(
                _module_equal(candidate.model[index], source.model[index])
                for index in range(len(candidate.model) - 1)
            )
        ),
        "stb_blocks_unchanged": _module_equal(candidate_head.blocks, source_blocks),
        "box_heads_unchanged": (
            _module_equal(candidate_head.one2many["box_head"], source_base.one2many["box_head"])
            and _module_equal(candidate_head.one2one["box_head"], source_base.one2one["box_head"])
        ),
    }
    if stage == "geometry":
        gates["native_head_unchanged"] = _module_equal(candidate_head.base_head, source_base)
    elif stage == "frequency":
        if not isinstance(source_head, SGFRDetectHead):
            raise TypeError("Frequency SGFR harus dimulai dari checkpoint geometry SGFR")
        gates["native_head_unchanged"] = _module_equal(candidate_head.base_head, source_base)
        gates["geometry_residual_unchanged"] = _module_equal(
            candidate_head.geometry_levels, source_head.geometry_levels
        )
    else:
        gates["classification_head_was_authorized"] = not _module_equal(
            candidate_head.one2many["cls_head"], source_base.one2many["cls_head"]
        )
    return {
        "stage": stage,
        "source_checkpoint": str(source_path),
        "candidate_checkpoint": str(candidate_path),
        "source_sha256": _sha256(source_path),
        "candidate_sha256": _sha256(candidate_path),
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
    }
