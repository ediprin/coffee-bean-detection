"""Dedicated IGEM-only static authorization for AF2FS frozen-parent confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from coffee_detector.afab import AFABConfig

from .config import AF2ParentResidualConfig
from .model import (
    AF2ParentResidualDetectionModel,
    freeze_for_parent_residual,
    load_af2_parent_residual_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = ("AF2IGEM0", "AF2IGEM1")
CONFIGS = {code: REPO_ROOT / f"configs/af2_parent_residual/{code}.yaml" for code in ARMS}
ATOL = 5.0e-5
RTOL = 1.0e-5
AUDIT_REVISION = "2026-08-24c"
INIT_SEED = 20260824


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _schema(module: nn.Module) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def _numerically_preserved(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left, right, atol=ATOL, rtol=RTOL))


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def _activate_last_projection(model: AF2ParentResidualDetectionModel) -> None:
    with torch.no_grad():
        for level in model.model[-1].residual:
            level.class_correction.weight.fill_(1.0)


def _state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _parent_state(model: AF2ParentResidualDetectionModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if ".residual." not in key
    }


def _parent_transfer_exact(source: nn.Module, target: AF2ParentResidualDetectionModel) -> bool:
    source_layers = source.model
    target_layers = target.model
    if len(source_layers) != len(target_layers):
        return False
    for index in range(len(source_layers) - 1):
        if not _state_equal(_clone_state(source_layers[index]), _clone_state(target_layers[index])):
            return False
    return _state_equal(_clone_state(source_layers[-1]), _clone_state(target_layers[-1].base_head))


def _bn_contract(model: AF2ParentResidualDetectionModel) -> tuple[bool, bool]:
    parent_bn, residual_bn = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            (residual_bn if ".residual" in name else parent_bn).append(module)
    parent_frozen = bool(parent_bn) and all(not module.training for module in parent_bn)
    residual_training = bool(residual_bn) and all(module.training for module in residual_bn)
    return parent_frozen, residual_training


def run_af2_igem_parent_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 64,
) -> dict[str, Any]:
    """Authorize only the seed-matched AF2IGEM0/1 pair; SAF is not materialized."""

    from ultralytics import YOLO

    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payloads = {
        code: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for code, path in CONFIGS.items()
    }
    source = YOLO(str(checkpoint)).model.to(device).eval()
    if getattr(source, "afab", None) is None:
        raise RuntimeError("Checkpoint sumber bukan AF2")

    torch.manual_seed(INIT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INIT_SEED)
    image = torch.rand(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        native = source(image)
    native_boxes = native[1]["one2one"]["boxes"]
    native_scores = native[1]["one2one"]["scores"]

    records: dict[str, Any] = {}
    initial_residual_states: dict[str, dict[str, torch.Tensor]] = {}
    for code in ARMS:
        payload = payloads[code]
        config = AF2ParentResidualConfig.from_mapping(payload["parent_residual"])
        if config.family != "igem":
            raise RuntimeError(f"{code} bukan family IGEM")

        torch.manual_seed(INIT_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(INIT_SEED)
        model = AF2ParentResidualDetectionModel(
            str(REPO_ROOT / payload["model"]),
            nc=int(source.model[-1].nc),
            verbose=False,
            afab=AFABConfig.from_mapping(payload["afab"]),
            parent_residual=config,
        ).to(device)
        transfer = load_af2_parent_residual_weights(model, source)
        transfer_exact = _parent_transfer_exact(source, model)
        initial_residual_states[code] = _clone_state(model.model[-1].residual)

        model.eval()
        with torch.inference_mode():
            identity = model(image)
        before_boxes = identity[1]["one2one"]["boxes"].clone()
        before_scores = identity[1]["one2one"]["scores"].clone()
        initial_identity = _numerically_preserved(before_boxes, native_boxes) and _numerically_preserved(
            before_scores, native_scores
        )

        _activate_last_projection(model)
        with torch.inference_mode():
            active = model(image)
        active_boxes = active[1]["one2one"]["boxes"]
        active_scores = active[1]["one2one"]["scores"]
        active_box_max_abs_diff = _max_abs_diff(before_boxes, active_boxes)
        active_score_max_abs_diff = _max_abs_diff(before_scores, active_scores)

        freeze = freeze_for_parent_residual(model)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        only_residual_trainable = bool(trainable_names) and all("model.23.residual" in name for name in trainable_names)
        model.train(True)
        parent_bn_frozen, residual_bn_training = _bn_contract(model)
        parent_before_step = _parent_state(model)
        model.zero_grad(set_to_none=True)
        training_output = model(image)["one2many"]
        objective = training_output["scores"].square().mean() + sum(
            value.square().mean() for value in training_output["parent_residual_mask_logits"]
        )
        objective.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        frozen = [parameter for parameter in model.parameters() if not parameter.requires_grad]
        finite_gradients = all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable
        ) and any(bool(parameter.grad.abs().max() > 0) for parameter in trainable)
        parent_no_grad = all(parameter.grad is None for parameter in frozen)
        optimizer = torch.optim.SGD(trainable, lr=1.0e-4)
        optimizer.step()
        parent_unchanged = _state_equal(parent_before_step, _parent_state(model))

        records[code] = {
            "family": config.family,
            "conditioning": config.conditioning,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": freeze["trainable"],
            "trainable_names": trainable_names,
            "state_schema": _schema(model),
            "transfer": transfer,
            "parent_transfer_bitwise_exact": transfer_exact,
            "initial_af2_numerically_preserved": initial_identity,
            "active_boxes_bitwise_preserved": bool(torch.equal(before_boxes, active_boxes)),
            "active_boxes_numerically_preserved": _numerically_preserved(before_boxes, active_boxes),
            "active_box_max_abs_diff": active_box_max_abs_diff,
            "active_scores_numerically_preserved": _numerically_preserved(before_scores, active_scores),
            "active_score_max_abs_diff": active_score_max_abs_diff,
            "only_residual_trainable": only_residual_trainable,
            "parent_batchnorm_frozen": parent_bn_frozen,
            "residual_batchnorm_training": residual_bn_training,
            "finite_nonzero_residual_gradients": finite_gradients,
            "frozen_parent_has_no_gradients": parent_no_grad,
            "parent_state_unchanged_after_residual_step": parent_unchanged,
        }

    control, candidate = records["AF2IGEM0"], records["AF2IGEM1"]
    left_cfg = dict(payloads["AF2IGEM0"]["parent_residual"])
    right_cfg = dict(payloads["AF2IGEM1"]["parent_residual"])
    left_cfg.pop("conditioning", None)
    right_cfg.pop("conditioning", None)
    matched_initial_residual = _state_equal(
        initial_residual_states["AF2IGEM0"], initial_residual_states["AF2IGEM1"]
    )

    gates = {
        "source_is_af2": True,
        "same_model_yaml": payloads["AF2IGEM0"]["model"] == payloads["AF2IGEM1"]["model"],
        "same_af2_config": payloads["AF2IGEM0"]["afab"] == payloads["AF2IGEM1"]["afab"],
        "same_training_schedule": payloads["AF2IGEM0"]["train"] == payloads["AF2IGEM1"]["train"],
        "same_parameter_count": control["parameters"] == candidate["parameters"],
        "same_trainable_count": control["trainable_parameters"] == candidate["trainable_parameters"],
        "same_state_schema": control["state_schema"] == candidate["state_schema"],
        "same_initial_residual_state": matched_initial_residual,
        "only_conditioning_differs": left_cfg == right_cfg,
        "control_receives_zero_information": control["conditioning"] == "zero",
        "candidate_receives_features": candidate["conditioning"] == "feature",
        "control_parent_transfer_exact": control["parent_transfer_bitwise_exact"],
        "candidate_parent_transfer_exact": candidate["parent_transfer_bitwise_exact"],
        "control_initial_identity": control["initial_af2_numerically_preserved"],
        "candidate_initial_identity": candidate["initial_af2_numerically_preserved"],
        "control_boxes_preserved": control["active_boxes_numerically_preserved"],
        "candidate_boxes_preserved": candidate["active_boxes_numerically_preserved"],
        "control_zero_information_identity": control["active_score_max_abs_diff"] <= ATOL,
        "candidate_changes_scores": candidate["active_score_max_abs_diff"] > ATOL,
        "control_only_residual_trainable": control["only_residual_trainable"],
        "candidate_only_residual_trainable": candidate["only_residual_trainable"],
        "control_parent_bn_frozen": control["parent_batchnorm_frozen"],
        "candidate_parent_bn_frozen": candidate["parent_batchnorm_frozen"],
        "control_residual_bn_training": control["residual_batchnorm_training"],
        "candidate_residual_bn_training": candidate["residual_batchnorm_training"],
        "control_finite_gradients": control["finite_nonzero_residual_gradients"],
        "candidate_finite_gradients": candidate["finite_nonzero_residual_gradients"],
        "control_parent_no_grad": control["frozen_parent_has_no_gradients"],
        "candidate_parent_no_grad": candidate["frozen_parent_has_no_gradients"],
        "control_parent_state_unchanged": control["parent_state_unchanged_after_residual_step"],
        "candidate_parent_state_unchanged": candidate["parent_state_unchanged_after_residual_step"],
        "test_accessed": False,
    }
    decision = "PASS" if all(value for key, value in gates.items() if key != "test_accessed") and not gates["test_accessed"] else "FAIL"
    if decision != "PASS":
        compact = {
            "revision": AUDIT_REVISION,
            "failed_gates": {key: value for key, value in gates.items() if value is False and key != "test_accessed"},
            "control_box_diff": control["active_box_max_abs_diff"],
            "control_score_diff": control["active_score_max_abs_diff"],
            "candidate_box_diff": candidate["active_box_max_abs_diff"],
            "candidate_score_diff": candidate["active_score_max_abs_diff"],
        }
        print("IGEM STATIC AUDIT FAIL:", json.dumps(compact, sort_keys=True), flush=True)
    result = {
        "format": "coffee_detector.af2_parent_residual.igem_static_audit.v1",
        "audit_revision": AUDIT_REVISION,
        "numerical_tolerance": {"atol": ATOL, "rtol": RTOL},
        "activity_absolute_threshold": ATOL,
        "initialization_seed": INIT_SEED,
        "decision": decision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "records": records,
        "gates": gates,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
        "note": "Dedicated IGEM audit; shared SAF/IGEM legacy audit is not used for this confirmation.",
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
