"""Static gates for frozen-parent AF2 + FFAB2 residual training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab import AFABConfig
from .model import AF2FFAConfig, AF2FFADetectHead, load_af2_ffa_weights
from .parent_preserving import AF2FFAParentPreservingModel, adapter_parameter_names


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "AF2FFAPR0": REPO_ROOT / "configs/af2_ffa_parent_preserving/AF2FFAPR0_yolo26n_zero_parent_residual.yaml",
    "AF2FFAPR1": REPO_ROOT / "configs/af2_ffa_parent_preserving/AF2FFAPR1_yolo26n_spectral_parent_residual.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_without_adapters(model: AF2FFAParentPreservingModel) -> dict[str, torch.Tensor]:
    marker = f"model.{len(model.model) - 1}.adapters."
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith(marker)
    }


def _identical_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float((left.float() - right.float()).abs().max())


def _build(source, payload: dict[str, Any], device: str) -> AF2FFAParentPreservingModel:
    source_head = source.model[-1]
    model = AF2FFAParentPreservingModel(
        str(REPO_ROOT / payload["model"]),
        nc=int(source_head.nc),
        verbose=False,
        afab=AFABConfig.from_mapping(payload["afab"]),
        af2_ffa=AF2FFAConfig.from_mapping(payload["af2_ffa"]),
    ).to(device)
    load_af2_ffa_weights(model, source)
    model.freeze_parent()
    return model


def run_af2_ffa_parent_preserving_audit(
    parent_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 128,
) -> dict[str, Any]:
    """Prove exact AF2 start, frozen parent, box isolation, and live spectral gradient."""

    from ultralytics import YOLO

    checkpoint = Path(parent_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payloads = {
        code: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for code, path in CONFIGS.items()
    }
    source = YOLO(str(checkpoint)).model.to(device).eval()
    if type(source.model[-1]).__name__ != "Detect":
        raise RuntimeError("Parent AF2 harus memiliki native Detect head")

    torch.manual_seed(20260824)
    image = torch.rand(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        parent_output = source(image)

    records: dict[str, Any] = {}
    models: dict[str, AF2FFAParentPreservingModel] = {}
    for code, payload in payloads.items():
        model = _build(source, payload, device)
        models[code] = model
        model.eval()
        with torch.inference_mode():
            output0 = model(image)
        head = model.model[-1]
        if not isinstance(head, AF2FFADetectHead):
            raise TypeError(type(head).__name__)
        full_box_diff = _max_abs(
            output0[1]["one2one"]["boxes"], parent_output[1]["one2one"]["boxes"]
        )
        full_score_diff = _max_abs(
            output0[1]["one2one"]["scores"], parent_output[1]["one2one"]["scores"]
        )

        sizes = (16, 8, 4)
        features = [
            torch.rand(1, adapter.channels, size, size, device=device)
            for adapter, size in zip(head.adapters, sizes)
        ]
        descriptor = head.adapters[0].spectral_descriptor(features[0])
        trainable = adapter_parameter_names(model)
        all_trainable_are_adapters = bool(trainable) and all(".adapters." in name for name in trainable)
        bn_frozen = True
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
                bn_frozen = False
                break

        # Candidate must have a live residual gradient while its parent remains frozen.
        model.zero_grad(set_to_none=True)
        cls_head = head.one2many["cls_head"][0]
        scores = head._classification_scores(0, features[0], cls_head)
        scores.float().square().mean().backward()
        adapter_grads = [
            parameter.grad
            for adapter in head.adapters
            for parameter in adapter.parameters()
        ]
        finite_grad = all(
            grad is None or bool(torch.isfinite(grad).all()) for grad in adapter_grads
        )
        nonzero_adapter_grad = any(
            grad is not None and float(grad.detach().abs().sum()) > 0.0
            for grad in adapter_grads
        )
        parent_has_no_grad = all(
            parameter.grad is None
            for name, parameter in model.named_parameters()
            if ".adapters." not in name
        )

        parent_before = _state_without_adapters(model)
        if code == "AF2FFAPR1":
            optimizer = torch.optim.SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=1.0e-3,
            )
            optimizer.step()
        parent_after = _state_without_adapters(model)

        # Active spectral adapter must only move classification scores.
        model.eval()
        with torch.no_grad():
            for adapter in head.adapters:
                adapter.alpha.fill_(0.05)
                adapter.bias.fill_(0.20)
        with torch.inference_mode():
            active = head([item.clone() for item in features])
            native = head.base_head([item.clone() for item in features])
        active_boxes_equal = torch.equal(
            active[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"]
        )
        active_score_delta = float(
            (active[1]["one2one"]["scores"] - native[1]["one2one"]["scores"])
            .abs()
            .max()
        )

        records[code] = {
            "conditioning": payload["af2_ffa"]["conditioning"],
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "trainable_names": list(trainable),
            "descriptor_abs_sum": float(descriptor.detach().abs().sum()),
            "initial_full_box_max_abs_diff": full_box_diff,
            "initial_full_score_max_abs_diff": full_score_diff,
            "active_score_max_abs_delta": active_score_delta,
            "gates": {
                "exact_parent_start_numerically": max(full_box_diff, full_score_diff) <= 1.0e-6,
                "only_adapters_trainable": all_trainable_are_adapters,
                "parent_batchnorm_frozen": bn_frozen,
                "adapter_gradients_finite": finite_grad,
                "candidate_has_live_adapter_gradient": (
                    nonzero_adapter_grad if code == "AF2FFAPR1" else True
                ),
                "parent_receives_no_gradient": parent_has_no_grad,
                "parent_state_unchanged_after_candidate_step": _identical_state(parent_before, parent_after),
                "active_preserves_boxes_bitwise": active_boxes_equal,
                "active_candidate_changes_scores": (
                    active_score_delta > 0.0 if code == "AF2FFAPR1" else True
                ),
                "descriptor_semantics_correct": (
                    float(descriptor.detach().abs().sum()) == 0.0
                    if code == "AF2FFAPR0"
                    else float(descriptor.detach().abs().sum()) > 0.0
                ),
            },
        }

    zero, spectral = records["AF2FFAPR0"], records["AF2FFAPR1"]
    left_cfg = dict(payloads["AF2FFAPR0"]["af2_ffa"])
    right_cfg = dict(payloads["AF2FFAPR1"]["af2_ffa"])
    left_cfg.pop("conditioning", None)
    right_cfg.pop("conditioning", None)
    global_gates = {
        "same_model_yaml": payloads["AF2FFAPR0"]["model"] == payloads["AF2FFAPR1"]["model"],
        "same_af2_config": payloads["AF2FFAPR0"]["afab"] == payloads["AF2FFAPR1"]["afab"],
        "same_train_schedule": payloads["AF2FFAPR0"]["train"] == payloads["AF2FFAPR1"]["train"],
        "same_adapter_config_except_conditioning": left_cfg == right_cfg,
        "same_parameter_count": zero["total_parameters"] == spectral["total_parameters"],
        "same_trainable_count": zero["trainable_parameters"] == spectral["trainable_parameters"],
        "trainable_fraction_under_one_percent": (
            0 < spectral["trainable_parameters"] < spectral["total_parameters"] * 0.01
        ),
        "parent_residual_fusion": payloads["AF2FFAPR1"]["af2_ffa"]["fusion_mode"] == "parent_residual",
        "no_ambiguity_gate": payloads["AF2FFAPR1"]["af2_ffa"]["ambiguity_gate"] == "none",
        "thirty_epoch_continuation": int(payloads["AF2FFAPR1"]["train"]["epochs"]) == 30,
    }
    all_gates = list(global_gates.values()) + [
        value for record in records.values() for value in record["gates"].values()
    ]
    result = {
        "format": "coffee_detector.af2_ffa.parent_preserving_static_audit.v1",
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": _sha256(checkpoint),
        "records": records,
        "global_gates": global_gates,
        "decision": "PASS" if all(all_gates) else "FAIL",
        "training_authorized": bool(all(all_gates)),
        "test_access_authorized": False,
        "note": (
            "AF2FFAPR0 is an exact zero-information/frozen-parent negative control. "
            "With the frozen parent and zero initialization it is expected to remain the AF2 parent; "
            "it is not an active generic-residual capacity control."
        ),
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
