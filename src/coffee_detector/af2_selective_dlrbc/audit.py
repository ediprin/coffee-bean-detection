from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from coffee_detector.afab.operator import AFABConfig

from .model import (
    AF2SelectiveDLRBCConfig,
    AF2SelectiveDLRBCDetectionModel,
    SelectiveLowRankResidual,
    load_af2_selective_weights,
)


def _raw(model: torch.nn.Module, value: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    model.eval()
    with torch.inference_mode():
        output = model(value)
    if not isinstance(output, tuple) or not isinstance(output[1], dict):
        raise TypeError("Model tidak mengekspos raw dual-head output")
    return output[1]


def run_af2_selective_static_audit(
    model_yaml: str | Path,
    af2_checkpoint: str | Path,
    afab: AFABConfig | Mapping[str, Any],
    selected_class_ids: list[int] | tuple[int, ...],
    output: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    from ultralytics import YOLO

    selected = tuple(sorted(set(int(value) for value in selected_class_ids)))
    selective = AF2SelectiveDLRBCConfig(selected_class_ids=selected)
    source = YOLO(str(Path(af2_checkpoint).expanduser().resolve())).model
    candidate = AF2SelectiveDLRBCDetectionModel(
        str(Path(model_yaml).expanduser().resolve()),
        ch=3,
        nc=21,
        verbose=False,
        afab=afab,
        selective=selective,
    )
    transfer = load_af2_selective_weights(candidate, source)
    torch_device = torch.device(device)
    source.to(torch_device)
    candidate.to(torch_device)
    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64, device=torch_device).reshape(1, 3, 64, 64)
    source_raw = _raw(source, probe)
    zero_raw = _raw(candidate, probe)
    branches = ("one2many", "one2one")
    zero_boxes_equal = all(torch.equal(source_raw[name]["boxes"], zero_raw[name]["boxes"]) for name in branches)
    zero_scores_equal = all(torch.equal(source_raw[name]["scores"], zero_raw[name]["scores"]) for name in branches)

    modules = [module for module in candidate.modules() if isinstance(module, SelectiveLowRankResidual)]
    with torch.no_grad():
        for module in modules:
            module.gate[list(selected)] = 0.5
    active_raw = _raw(candidate, probe)
    selected_mask = torch.zeros(21, dtype=torch.bool, device=torch_device)
    selected_mask[list(selected)] = True
    unselected_mask = ~selected_mask
    active_selected = any(
        not torch.equal(
            zero_raw[name]["scores"][:, selected_mask],
            active_raw[name]["scores"][:, selected_mask],
        )
        for name in branches
    )
    unselected_equal = all(
        torch.equal(
            zero_raw[name]["scores"][:, unselected_mask],
            active_raw[name]["scores"][:, unselected_mask],
        )
        for name in branches
    )
    active_boxes_equal = all(torch.equal(zero_raw[name]["boxes"], active_raw[name]["boxes"]) for name in branches)

    gradient_module = modules[0]
    gradient_module.train()
    gradient_probe = torch.randn(
        2,
        gradient_module.residual.channels,
        8,
        8,
        device=torch_device,
        requires_grad=True,
    )
    loss = gradient_module(gradient_probe).square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in gradient_module.parameters()]
    finite_gradients = all(value is not None and torch.isfinite(value).all().item() for value in gradients)
    nonzero_gate_gradient = bool(
        gradient_module.gate.grad is not None
        and gradient_module.gate.grad[list(selected)].abs().sum().item() > 0.0
    )
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    gates = {
        "selected_class_count_bounded": 2 <= len(selected) <= 10,
        "zero_gate_is_exact_af2": zero_boxes_equal and zero_scores_equal,
        "active_residual_changes_selected_scores": active_selected,
        "active_residual_preserves_unselected_scores": unselected_equal,
        "active_residual_preserves_raw_boxes": active_boxes_equal,
        "finite_gradients": finite_gradients,
        "nonzero_gate_gradient": nonzero_gate_gradient,
        "candidate_adds_parameters": candidate_parameters > source_parameters,
        "test_not_accessed": True,
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.af2_selective_dlrbc.static_audit.v1",
        "protocol": "faruq-v3-af2-class-selective-dlrbc-seed42-v1",
        "selected_class_ids": list(selected),
        "source_parameters": source_parameters,
        "candidate_parameters": candidate_parameters,
        "added_parameters": candidate_parameters - source_parameters,
        "transfer": transfer,
        "gates": gates,
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
