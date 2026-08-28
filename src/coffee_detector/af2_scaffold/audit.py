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


def _flatten(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result: list[torch.Tensor] = []
        for key in sorted(value):
            result.extend(_flatten(value[key]))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return []


def _equal(left: Any, right: Any) -> bool:
    a, b = _flatten(left), _flatten(right)
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _different(left: Any, right: Any) -> bool:
    return not _equal(left, right)


def _parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


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

    sample = torch.rand(1, 3, 64, 64, device=resolved_device)
    source.train()
    candidate.train()
    with torch.no_grad():
        source_train = source(sample.clone())
        candidate_train = candidate(sample.clone())
    initial_train_exact = _equal(source_train, candidate_train)

    source.eval()
    candidate.eval()
    with torch.inference_mode():
        source_eval = source(sample.clone())
        candidate_eval = candidate(sample.clone())
    initial_eval_exact = _equal(source_eval, candidate_eval)

    synthetic_features = [
        torch.rand(2, channel, max(3, 12 // (2**level)), max(3, 12 // (2**level)), device=resolved_device)
        for level, channel in enumerate(head.scaffold.channels)
    ]
    head.base_head.eval()
    head.scaffold.train()
    with torch.no_grad():
        inactive_head_output = head([value.clone() for value in synthetic_features])

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
    head.base_head.eval()
    head.scaffold.train()
    with torch.no_grad():
        active_head_output = head([value.clone() for value in synthetic_features])
    active_changes_train = _different(inactive_head_output, active_head_output)
    candidate.eval()
    with torch.inference_mode():
        active_eval = candidate(sample.clone())
    active_eval_still_exact = _equal(source_eval, active_eval)

    source_parameters = _parameters(source)
    candidate_parameters = _parameters(candidate)
    scaffold_parameters = _parameters(head.scaffold)
    forbidden = ("roi_align", "decoded_box", "topk_candidate", "test/images")
    implementation = inspect.getsource(TrainingOnlyMultilevelDetectHead).lower()
    no_forbidden_dependency = not any(token in implementation for token in forbidden)

    strip_training_scaffold(candidate)
    stripped_parameters = _parameters(candidate)
    stripped_schema_equal = set(candidate.state_dict()) == set(source.state_dict())
    candidate.eval()
    with torch.inference_mode():
        stripped_eval = candidate(sample.clone())
    stripped_output_exact = _equal(source_eval, stripped_eval)

    gates = {
        "source_is_native_af2_head": True,
        "initial_train_output_exact": initial_train_exact,
        "initial_eval_output_exact": initial_eval_exact,
        "all_three_levels_present": len(gradients) == 3,
        "all_three_levels_have_finite_gradients": all(gradients),
        "active_scaffold_changes_train_output": active_changes_train,
        "active_scaffold_never_changes_eval_output": active_eval_still_exact,
        "schedule_starts_exactly_one": config.strength(0) == 1.0,
        "schedule_zero_for_last_three_epochs": all(
            config.strength(epoch) == 0.0 for epoch in (27, 28, 29)
        ),
        "stripped_parameter_count_equals_af2": stripped_parameters == source_parameters,
        "training_parameter_delta_is_only_scaffold": candidate_parameters - source_parameters == scaffold_parameters,
        "stripped_state_schema_equals_af2": stripped_schema_equal,
        "stripped_output_exact": stripped_output_exact,
        "no_roi_decoded_box_or_test_dependency": no_forbidden_dependency,
        "test_accessed": False,
    }
    passed = all(value for key, value in gates.items() if key != "test_accessed") and not gates["test_accessed"]
    result = {
        "format": "coffee_detector.af2_scaffold.static_audit.v1",
        "arm": config.arm,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": config.to_dict(),
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
