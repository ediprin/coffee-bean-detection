from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .model import (
    FrozenResidualConfig,
    FrozenResidualDetectionModel,
    FrozenResidualDetectHead,
    freeze_native_detector,
    load_frozen_d0_weights,
)


ATOL = 1e-7


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_state(module: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            value.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for name, value in module.state_dict().items()
    }


def _raw_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        ignored = {"frozen_residual_indices", "frozen_residual_gate"}
        keys = set(left) - ignored
        return keys == (set(right) - ignored) and all(
            _raw_equal(left[key], right[key]) for key in keys
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _raw_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _native_batchnorm_eval(model: FrozenResidualDetectionModel) -> bool:
    head = model.model[-1]
    modules = list(model.model[:-1].modules()) + list(head.base_head.modules())
    batch_norms = [module for module in modules if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    return bool(batch_norms) and all(not module.training for module in batch_norms)


def static_frozen_residual_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    image_size: int = 128,
    topk: int = 32,
) -> dict[str, Any]:
    from ultralytics import YOLO

    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)
    d0 = YOLO(str(d0_checkpoint)).model.cpu().eval()
    config = FrozenResidualConfig(topk=topk, training_topk=topk)
    candidate = FrozenResidualDetectionModel(
        str(model_yaml), nc=nc, verbose=False, frozen_residual=config
    ).cpu()
    transfer = load_frozen_d0_weights(candidate, d0)
    counts = freeze_native_detector(candidate)
    candidate.eval()
    d0_head_hash = _hash_state(d0.model[-1])
    candidate_head_hash = _hash_state(candidate.model[-1].base_head)

    torch.manual_seed(42)
    image = torch.randn(1, 3, image_size, image_size)
    with torch.inference_mode():
        d0_output = d0(image)
        zero_output = candidate(image)
    max_difference = float((d0_output[0] - zero_output[0]).abs().max())

    candidate.train(True)
    bn_eval = _native_batchnorm_eval(candidate)
    trainable_names = [
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    ]
    native_frozen = all(
        (".refiner." in name or ".gate." in name) for name in trainable_names
    )

    head = candidate.model[-1]
    if not isinstance(head, FrozenResidualDetectHead):
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
    base_logits = torch.randn(4, nc)
    residual = head.refiner(features, rois, (8.0, 16.0, 32.0))
    final, gate, correction = head.apply_residual(base_logits, residual)
    direct_loss = F.cross_entropy(final, labels) + 0.25 * correction.square().mean()
    direct_loss.backward()
    refiner_gradients = [
        parameter.grad
        for parameter in head.refiner.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    gate_gradients = [
        parameter.grad
        for parameter in head.gate.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = bool(refiner_gradients) and bool(gate_gradients) and all(
        bool(torch.isfinite(value).all())
        for value in (*refiner_gradients, *gate_gradients)
    )
    candidate.zero_grad(set_to_none=True)

    active = copy.deepcopy(candidate).eval()
    active_head = active.model[-1]
    torch.manual_seed(7)
    nn.init.normal_(active_head.refiner.classifier.weight, std=0.02)
    nn.init.zeros_(active_head.refiner.classifier.bias)
    nn.init.zeros_(active_head.gate.linear.weight)
    nn.init.zeros_(active_head.gate.linear.bias)
    with torch.inference_mode():
        active_output = active(image)
    active_difference = float((active_output[0] - d0_output[0]).abs().max())
    active_raw_boxes_equal = bool(
        torch.equal(
            active_output[1]["one2one"]["boxes"],
            d0_output[1]["one2one"]["boxes"],
        )
    )

    clone = FrozenResidualDetectionModel(
        str(model_yaml), nc=nc, verbose=False, frozen_residual=config
    ).cpu().eval()
    missing, unexpected = clone.load_state_dict(candidate.state_dict(), strict=False)
    clone_head = clone.model[-1]
    clone_head.max_det = head.max_det
    clone_head.base_head.max_det = head.max_det
    with torch.inference_mode():
        clone_output = clone(image)[0]
    roundtrip = (
        not missing
        and not unexpected
        and bool(torch.allclose(clone_output, zero_output[0], rtol=0.0, atol=ATOL))
    )

    gates = {
        "native_d0_head_bitwise_preserved": d0_head_hash == candidate_head_hash,
        "zero_output_is_d0": max_difference <= ATOL,
        "zero_raw_native_predictions_bitwise_equal": _raw_equal(
            zero_output[1], d0_output[1]
        ),
        "only_refiner_and_gate_trainable": native_frozen and bool(trainable_names),
        "native_batchnorm_stays_eval": bn_eval,
        "zero_initialized_residual_classifier": bool(
            torch.count_nonzero(head.refiner.classifier.weight) == 0
            and torch.count_nonzero(head.refiner.classifier.bias) == 0
        ),
        "finite_refiner_and_gate_gradients": finite_gradients,
        "active_residual_changes_output": active_difference > ATOL,
        "active_residual_preserves_native_boxes": active_raw_boxes_equal,
        "state_dict_roundtrip": roundtrip,
    }
    passed = all(gates.values())
    payload = {
        "protocol": "faruq-v3-frozen-residual-static-v1",
        "training_executed": False,
        "dataset_accessed": False,
        "test_images_accessed": False,
        "d0_checkpoint": str(d0_checkpoint),
        "d0_checkpoint_sha256": _sha256_file(d0_checkpoint),
        "transfer": transfer,
        "parameter_counts": counts,
        "trainable_fraction": counts["trainable"] / counts["total"],
        "trainable_names": trainable_names,
        "identity_atol": ATOL,
        "zero_output_max_abs_diff": max_difference,
        "initial_gate_mean": float(gate.detach().mean()),
        "direct_loss": float(direct_loss.detach()),
        "active_output_max_abs_diff": active_difference,
        "gates": gates,
        "decision": "PASS" if passed else "FAIL",
        "next_action": (
            "AUTHORIZE_FRM1_SEED42_VALIDATION_SCREENING"
            if passed
            else "STOP_FRM1_STATIC_AUDIT"
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
    parser = argparse.ArgumentParser(description="Static D0-preservation audit for FRM1")
    parser.add_argument("--model-yaml", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nc", type=int, default=21)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=32)
    args = parser.parse_args()
    result = static_frozen_residual_audit(
        args.model_yaml,
        args.d0_checkpoint,
        args.output,
        nc=args.nc,
        image_size=args.image_size,
        topk=args.topk,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
