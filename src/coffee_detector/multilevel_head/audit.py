from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .model import (
    MultilevelHeadConfig,
    MultilevelResidualDetectHead,
    inject_multilevel_head,
)


def _hash_state(module: nn.Module) -> dict[str, str]:
    output = {}
    for name, value in module.state_dict().items():
        payload = value.detach().cpu().contiguous().numpy().tobytes()
        output[name] = hashlib.sha256(payload).hexdigest()
    return output


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape(item) for item in value]
    return type(value).__name__


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


IDENTITY_ATOL = 1e-7


def _numerically_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Require strict absolute agreement while tolerating CPU decode round-off."""

    return bool(torch.allclose(left, right, rtol=0.0, atol=IDENTITY_ATOL))


def _raw_predictions_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _raw_predictions_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _raw_predictions_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _load_base(
    model_yaml: str | Path,
    *,
    nc: int,
    weights: str | Path | None,
) -> nn.Module:
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(model_yaml), nc=int(nc), verbose=False)
    if weights is not None:
        from ultralytics import YOLO

        model.load(YOLO(str(weights)).model)
    return model


def _inject_clone(
    base: nn.Module,
    mode: str,
    *,
    inference_weight: float,
    topk: int,
    seed: int = 123,
) -> nn.Module:
    model = copy.deepcopy(base)
    torch.manual_seed(seed)
    inject_multilevel_head(
        model,
        MultilevelHeadConfig(
            mode=mode,
            descriptor_dim=512,
            roi_size=3,
            topk=topk,
            inference_weight=inference_weight,
            box_expand=1.0,
        ),
    )
    return model


def _latency_ms(model: nn.Module, image: torch.Tensor, repeats: int = 8) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(2):
            model(image)
        started = time.perf_counter()
        for _ in range(repeats):
            model(image)
    return (time.perf_counter() - started) * 1000.0 / repeats


def static_multilevel_head_audit(
    model_yaml: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    weights: str | Path | None = None,
    image_size: int = 128,
    topk: int = 32,
) -> dict[str, Any]:
    if image_size < 64 or image_size % 32:
        raise ValueError("image_size static audit harus kelipatan 32 dan minimal 64")
    base = _load_base(model_yaml, nc=nc, weights=weights).cpu().eval()
    base_parameters = _parameter_count(base)
    base_head_hash = _hash_state(base.model[-1])
    control_zero = _inject_clone(
        base, "p5_control", inference_weight=0.0, topk=topk
    ).cpu().eval()
    fusion_zero = _inject_clone(
        base, "pyramid_fusion", inference_weight=0.0, topk=topk
    ).cpu().eval()
    control_active = _inject_clone(
        base, "p5_control", inference_weight=0.5, topk=topk
    ).cpu().eval()
    fusion_active = _inject_clone(
        base, "pyramid_fusion", inference_weight=0.5, topk=topk
    ).cpu().eval()

    variants = {
        "MHC0": control_zero,
        "MHF1": fusion_zero,
    }
    parameter_counts = {name: _parameter_count(model) for name, model in variants.items()}
    state_schemas = {
        name: {key: list(value.shape) for key, value in model.state_dict().items()}
        for name, model in variants.items()
    }
    native_heads_preserved = {
        name: _hash_state(model.model[-1].base_head) == base_head_hash
        for name, model in variants.items()
    }

    image = torch.randn(1, 3, image_size, image_size)
    with torch.inference_mode():
        base_output = base(image)
        control_zero_output = control_zero(image)
        fusion_zero_output = fusion_zero(image)
        control_active_output = control_active(image)
        fusion_active_output = fusion_active(image)
    base_final = base_output[0]
    control_zero_final = control_zero_output[0]
    fusion_zero_final = fusion_zero_output[0]
    control_active_final = control_active_output[0]
    fusion_active_final = fusion_active_output[0]
    identity = {
        "absolute_tolerance": IDENTITY_ATOL,
        "control_zero_vs_d0": _numerically_identical(
            control_zero_final, base_final
        ),
        "fusion_zero_vs_d0": _numerically_identical(
            fusion_zero_final, base_final
        ),
        "control_zero_vs_fusion_zero": _numerically_identical(
            control_zero_final, fusion_zero_final
        ),
        "control_zero_vs_d0_max_abs_diff": float(
            (control_zero_final - base_final).abs().max()
        ),
        "fusion_zero_vs_d0_max_abs_diff": float(
            (fusion_zero_final - base_final).abs().max()
        ),
        "control_zero_vs_fusion_zero_max_abs_diff": float(
            (control_zero_final - fusion_zero_final).abs().max()
        ),
        "control_zero_raw_vs_d0_bitwise": _raw_predictions_equal(
            control_zero_output[1], base_output[1]
        ),
        "fusion_zero_raw_vs_d0_bitwise": _raw_predictions_equal(
            fusion_zero_output[1], base_output[1]
        ),
        "active_control_vs_fusion_max_abs_diff": float(
            (control_active_final - fusion_active_final).abs().max()
        ),
    }

    training_contract = {}
    for name, model in variants.items():
        model.train()
        training_contract[name] = _shape(model(torch.randn(2, 3, image_size, image_size)))
        model.eval()

    gradient_model = fusion_active
    head = gradient_model.model[-1]
    if not isinstance(head, MultilevelResidualDetectHead):
        raise TypeError(type(head).__name__)
    features = [
        torch.randn(2, 64, image_size // 8, image_size // 8),
        torch.randn(2, 128, image_size // 16, image_size // 16),
        torch.randn(2, 256, image_size // 32, image_size // 32),
    ]
    rois = torch.tensor(
        [
            [0.0, 8.0, 8.0, 64.0, 72.0],
            [0.0, 30.0, 20.0, 100.0, 110.0],
            [1.0, 12.0, 16.0, 80.0, 90.0],
            [1.0, 40.0, 35.0, 115.0, 120.0],
        ]
    )
    labels = torch.tensor([1, 2, 3, 4])
    logits = head.refiner(features, rois, (8.0, 16.0, 32.0))
    direct_loss = F.cross_entropy(logits, labels)
    direct_loss.backward()
    refiner_gradients = [
        parameter.grad
        for parameter in head.refiner.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_refiner_gradients = bool(refiner_gradients) and all(
        bool(torch.isfinite(value).all()) for value in refiner_gradients
    )
    gradient_model.zero_grad(set_to_none=True)

    clone = _inject_clone(
        base, "pyramid_fusion", inference_weight=0.5, topk=topk
    ).cpu().eval()
    missing, unexpected = clone.load_state_dict(fusion_active.state_dict(), strict=False)
    with torch.inference_mode():
        roundtrip_output = clone(image)[0]
    state_roundtrip_equal = (
        not missing
        and not unexpected
        and bool(torch.equal(roundtrip_output, fusion_active_final))
    )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "multilevel_state.pt"
        torch.save(fusion_active.state_dict(), checkpoint)
        serialized_bytes = checkpoint.stat().st_size

    control_latency = _latency_ms(control_active, image)
    fusion_latency = _latency_ms(fusion_active, image)
    latency_ratio = fusion_latency / max(control_latency, 1e-12)
    added_fraction = (parameter_counts["MHF1"] - base_parameters) / base_parameters
    gates = {
        "same_parameter_count": parameter_counts["MHC0"] == parameter_counts["MHF1"],
        "same_state_dict_schema": state_schemas["MHC0"] == state_schemas["MHF1"],
        "native_heads_preserved": all(native_heads_preserved.values()),
        "control_zero_is_d0": identity["control_zero_vs_d0"],
        "fusion_zero_is_d0": identity["fusion_zero_vs_d0"],
        "zero_modes_identical": identity["control_zero_vs_fusion_zero"],
        "zero_raw_native_predictions_bitwise_equal": (
            identity["control_zero_raw_vs_d0_bitwise"]
            and identity["fusion_zero_raw_vs_d0_bitwise"]
        ),
        "active_modes_differ": identity["active_control_vs_fusion_max_abs_diff"] > 0,
        "same_training_contract": training_contract["MHC0"] == training_contract["MHF1"],
        "finite_refiner_gradients": finite_refiner_gradients,
        "state_roundtrip_equal": state_roundtrip_equal,
        "added_parameters_no_more_than_30_percent": added_fraction <= 0.30,
        "fusion_latency_no_more_than_25_percent_over_control": latency_ratio <= 1.25,
    }
    passed = all(gates.values())
    payload = {
        "protocol": "faruq-v3-multilevel-head-static-v1",
        "training_executed": False,
        "dataset_accessed": False,
        "test_images_accessed": False,
        "model_yaml": str(Path(model_yaml).resolve()),
        "weights": str(weights) if weights is not None else None,
        "nc": nc,
        "image_size": image_size,
        "topk": topk,
        "base_parameters": base_parameters,
        "parameter_counts": parameter_counts,
        "added_parameter_fraction": float(added_fraction),
        "state_dict_bytes": int(serialized_bytes),
        "state_schema_equal": state_schemas["MHC0"] == state_schemas["MHF1"],
        "native_heads_preserved": native_heads_preserved,
        "identity": identity,
        "training_contract": training_contract,
        "evaluation_contract": {
            "D0": _shape(base_output),
            "MHC0": _shape(control_zero_output),
            "MHF1": _shape(fusion_zero_output),
        },
        "direct_refiner_loss": float(direct_loss.detach()),
        "finite_refiner_gradients": finite_refiner_gradients,
        "state_roundtrip_equal": state_roundtrip_equal,
        "latency_ms_cpu_smoke": {
            "MHC0": float(control_latency),
            "MHF1": float(fusion_latency),
            "ratio": float(latency_ratio),
        },
        "gates": gates,
        "decision": "PASS" if passed else "FAIL",
        "next_action": (
            "AUTHORIZE_MULTILEVEL_HEAD_TRAINING_PROTOCOL"
            if passed
            else "STOP_MULTILEVEL_HEAD_STATIC_AUDIT"
        ),
        "training_authorized": False,
        "test_access_authorized": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static audit for capacity-matched multilevel YOLO26 head"
    )
    parser.add_argument("--model-yaml", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--nc", type=int, default=21)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=32)
    args = parser.parse_args()
    result = static_multilevel_head_audit(
        args.model_yaml,
        args.output,
        nc=args.nc,
        weights=args.weights,
        image_size=args.image_size,
        topk=args.topk,
    )
    print(json.dumps(result["gates"], indent=2, ensure_ascii=False))
    print("DECISION:", result["decision"])
    print("NEXT:", result["next_action"])
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
