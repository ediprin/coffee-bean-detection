from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from coffee_detector.afab.operator import AFABConfig

from .config import AF2ScaffoldConfig
from .model import (
    AF2ScaffoldDetectionModel,
    TrainingOnlyMultilevelDetectHead,
    load_af2_scaffold_weights,
    strip_training_scaffold,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_CFG = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(value: str | int | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if isinstance(value, int) or str(value).isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(str(value))


def _parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _state_exact(left: nn.Module, right: nn.Module) -> bool:
    first, second = left.state_dict(), right.state_dict()
    return set(first) == set(second) and all(
        torch.equal(first[key].detach().cpu(), second[key].detach().cpu())
        for key in first
    )


def run_af2_scaffold_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str | int | torch.device = "cpu",
) -> dict:
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    resolved_device = _device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia")

    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model.to(resolved_device)
    if type(source.model[-1]).__name__ != "Detect":
        raise TypeError("Checkpoint AF2 parent harus memakai native Detect")
    afab = AFABConfig.from_mapping(getattr(source, "afab_config", {"mode": "af2"}))
    config = AF2ScaffoldConfig()
    torch.manual_seed(20260828)
    candidate = AF2ScaffoldDetectionModel(
        str(MODEL_CFG), nc=21, verbose=False, afab=afab, scaffold=config
    ).to(resolved_device)
    load_af2_scaffold_weights(candidate, source)
    head = candidate.model[-1]
    if not isinstance(head, TrainingOnlyMultilevelDetectHead):
        raise TypeError("Candidate kehilangan TrainingOnlyMultilevelDetectHead")

    synthetic_features = [
        torch.rand(2, channel, max(3, 12 // (2**level)), max(3, 12 // (2**level)), device=resolved_device)
        for level, channel in enumerate(head.scaffold.channels)
    ]
    head.scaffold.train()
    with torch.no_grad():
        initial_features = head.scaffold([value.clone() for value in synthetic_features])
    initial_train_features_exact = all(
        torch.equal(source_feature, adapted_feature)
        for source_feature, adapted_feature in zip(synthetic_features, initial_features)
    )

    gradients: list[bool] = []
    for adapter, channel in zip(head.scaffold.adapters, head.scaffold.channels):
        nn.init.constant_(adapter.output.weight, 0.02)
        feature = torch.rand(2, channel, 12, 12, device=resolved_device, requires_grad=True)
        loss = adapter(feature).square().mean()
        loss.backward()
        gradients.append(
            feature.grad is not None
            and bool(torch.isfinite(feature.grad).all())
            and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in adapter.parameters()
            )
        )
    head.set_scaffold_strength(1.0)
    head.scaffold.train()
    with torch.no_grad():
        active_features = head.scaffold([value.clone() for value in synthetic_features])
    active_changes_each_level = [
        not torch.equal(source_feature, adapted_feature)
        for source_feature, adapted_feature in zip(synthetic_features, active_features)
    ]
    head.scaffold.eval()
    with torch.no_grad():
        eval_features = head.scaffold([value.clone() for value in synthetic_features])
    eval_features_exact = all(
        torch.equal(source_feature, adapted_feature)
        for source_feature, adapted_feature in zip(synthetic_features, eval_features)
    )

    source_parameters = _parameters(source)
    candidate_parameters = _parameters(candidate)
    scaffold_parameters = _parameters(head.scaffold)
    forbidden = ("roi_align", "decoded_box", "topk_candidate", "test/images")
    implementation = inspect.getsource(TrainingOnlyMultilevelDetectHead).lower()
    no_forbidden_dependency = not any(token in implementation for token in forbidden)

    strip_training_scaffold(candidate)
    stripped_parameters = _parameters(candidate)
    stripped_schema_equal = set(candidate.state_dict()) == set(source.state_dict())
    stripped_state_exact = _state_exact(candidate, source)

    gates = {
        "source_is_native_af2_head": True,
        "source_af2_config_matches_candidate": afab.to_dict() == candidate.afab_config.to_dict(),
        "initial_train_features_exact": initial_train_features_exact,
        "all_three_levels_present": len(gradients) == 3,
        "all_three_levels_have_finite_gradients": all(gradients),
        "active_scaffold_changes_every_train_level": all(active_changes_each_level),
        "active_scaffold_bypasses_every_eval_level_exactly": eval_features_exact,
        "schedule_starts_exactly_one": config.strength(0) == 1.0,
        "schedule_zero_for_last_three_epochs": all(
            config.strength(epoch) == 0.0 for epoch in (27, 28, 29)
        ),
        "stripped_parameter_count_equals_af2": stripped_parameters == source_parameters,
        "training_parameter_delta_is_only_scaffold": candidate_parameters - source_parameters == scaffold_parameters,
        "stripped_state_schema_equals_af2": stripped_schema_equal,
        "stripped_detector_state_values_equal_af2": stripped_state_exact,
        "no_roi_decoded_box_or_test_dependency": no_forbidden_dependency,
        "test_accessed": False,
    }
    passed = all(value for key, value in gates.items() if key != "test_accessed") and not gates["test_accessed"]
    result = {
        "format": "coffee_detector.af2_scaffold.static_audit.v2",
        "arm": config.arm,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": config.to_dict(),
        "equivalence_method": "same_feature_identity_and_exact_stripped_state",
        "active_changes_each_level": active_changes_each_level,
        "parameters": {
            "source_af2": source_parameters,
            "training_candidate": candidate_parameters,
            "training_scaffold": scaffold_parameters,
            "stripped_candidate": stripped_parameters,
        },
        "gates": gates,
        "decision": "PASS" if passed else "FAIL",
        "training_authorized": passed,
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
