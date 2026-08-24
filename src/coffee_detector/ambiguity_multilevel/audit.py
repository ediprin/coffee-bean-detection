"""Static safety audit for the one-stage ACMC head."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from .model import (
    AmbiguityMultilevelConfig,
    AmbiguityMultilevelDetectHead,
    AmbiguityMultilevelDetectionModel,
    load_ambiguity_multilevel_detector_weights,
)


ATOL = 1e-7


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_state(module: torch.nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, value in module.state_dict().items()
    }


def audit_ambiguity_multilevel_static(
    model_yaml: str | Path,
    *,
    num_classes: int = 21,
    image_size: int = 128,
    config: AmbiguityMultilevelConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify D0 identity at zero correction without accessing data or training."""
    from ultralytics.nn.tasks import DetectionModel

    model_yaml = Path(model_yaml).resolve()
    source = DetectionModel(str(model_yaml), nc=int(num_classes), verbose=False).eval()
    candidate = AmbiguityMultilevelDetectionModel(
        str(model_yaml), nc=int(num_classes), verbose=False, ambiguity_multilevel=config
    ).eval()
    transfer = load_ambiguity_multilevel_detector_weights(candidate, source)
    head = candidate.model[-1]
    if not isinstance(head, AmbiguityMultilevelDetectHead):
        raise TypeError("Injeksi ACMC gagal")
    image = torch.randn(1, 3, int(image_size), int(image_size))
    with torch.inference_mode():
        native_output = source(image)
        candidate_output = candidate(image)
    source_head = source.model[-1]
    source_state = source_head.state_dict()
    candidate_state = head.base_head.state_dict()
    source_code = inspect.getsource(type(head)) + inspect.getsource(type(head.correction))
    return {
        "training_executed": False,
        "test_images_accessed": False,
        "native_head_transfer": transfer,
        "native_head_state_identical": (
            source_state.keys() == candidate_state.keys()
            and all(torch.equal(source_state[key], candidate_state[key]) for key in source_state)
        ),
        "initial_detection_identical": bool(
            torch.allclose(candidate_output[0], native_output[0], rtol=0.0, atol=1e-7)
        ),
        "box_tensor_identical": bool(
            torch.equal(candidate_output[1]["one2one"]["boxes"], native_output[1]["one2one"]["boxes"])
        ),
        "score_tensor_identical": bool(
            torch.equal(candidate_output[1]["one2one"]["scores"], native_output[1]["one2one"]["scores"])
        ),
        "has_roi_align": "roi_align" in source_code,
        "has_topk": ".topk(" in source_code,
        "has_box_decode": "_get_decode_boxes" in source_code,
        "trainable_parameters": sum(parameter.numel() for parameter in candidate.parameters()),
        "correction_parameters": sum(parameter.numel() for parameter in head.correction.parameters()),
    }


def static_ambiguity_multilevel_audit(
    model_yaml: str | Path,
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    nc: int = 21,
    image_size: int = 128,
    config: AmbiguityMultilevelConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit an actual D0 checkpoint without opening data or beginning training."""
    from ultralytics import YOLO

    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)
    d0 = YOLO(str(d0_checkpoint)).model.cpu().eval()
    candidate = AmbiguityMultilevelDetectionModel(
        str(model_yaml), nc=int(nc), verbose=False, ambiguity_multilevel=config
    ).cpu()
    transfer = load_ambiguity_multilevel_detector_weights(candidate, d0)
    candidate.eval()
    head = candidate.model[-1]
    if not isinstance(head, AmbiguityMultilevelDetectHead):
        raise TypeError(type(head).__name__)
    native_hash = _hash_state(d0.model[-1])
    wrapped_hash = _hash_state(head.base_head)
    torch.manual_seed(42)
    image = torch.randn(1, 3, int(image_size), int(image_size))
    with torch.inference_mode():
        d0_output = d0(image)
        zero_output = candidate(image)
    zero_difference = float((d0_output[0] - zero_output[0]).abs().max())

    # Use a disposable training clone.  BatchNorm updates from this gradient
    # smoke test must never contaminate the zero/active inference comparison.
    gradient_probe = copy.deepcopy(candidate).train()
    probe_head = gradient_probe.model[-1]
    train_output = gradient_probe(image)
    direct_loss = train_output["one2many"]["scores"].square().mean()
    direct_loss.backward()
    correction_gradients = [
        parameter.grad
        for parameter in probe_head.correction.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = bool(correction_gradients) and all(
        bool(torch.isfinite(value).all()) for value in correction_gradients
    )
    active = copy.deepcopy(candidate).eval()
    active_head = active.model[-1]
    torch.manual_seed(7)
    for layer in active_head.correction.class_corrections:
        torch.nn.init.normal_(layer.weight, std=0.02)
        # A non-zero bias makes the wiring probe independent of a chance-zero
        # projected feature response on its synthetic smoke image.  It is
        # applied only to this disposable audit clone, never to ACMC1.
        torch.nn.init.constant_(layer.bias, 0.1)
    with torch.inference_mode():
        active_output = active(image)
    # The deployed post-process can retain identical top detections for a
    # small score perturbation.  Audit the native class-score tensor itself:
    # this is the exact tensor changed by ACMC before YOLO's post-process.
    active_difference = float(
        (
            active_output[1]["one2one"]["scores"]
            - d0_output[1]["one2one"]["scores"]
        )
        .abs()
        .max()
    )
    source_code = inspect.getsource(type(head)) + inspect.getsource(type(head.correction))
    gates = {
        "native_d0_head_bitwise_preserved": native_hash == wrapped_hash,
        "zero_output_is_d0": zero_difference <= ATOL,
        "zero_boxes_bitwise_equal": bool(
            torch.equal(zero_output[1]["one2one"]["boxes"], d0_output[1]["one2one"]["boxes"])
        ),
        "zero_scores_bitwise_equal": bool(
            torch.equal(zero_output[1]["one2one"]["scores"], d0_output[1]["one2one"]["scores"])
        ),
        "no_roi_align": "roi_align" not in source_code,
        "no_topk_candidate_selection": ".topk(" not in source_code,
        "no_box_decode_before_classification": "_get_decode_boxes" not in source_code,
        "finite_correction_gradients": finite_gradients,
        "active_correction_changes_scores": active_difference > ATOL,
        "active_correction_preserves_boxes": bool(
            torch.equal(active_output[1]["one2one"]["boxes"], d0_output[1]["one2one"]["boxes"])
        ),
    }
    payload = {
        "protocol": "faruq-v3-acmc-static-v1",
        "training_executed": False,
        "dataset_accessed": False,
        "test_images_accessed": False,
        "d0_checkpoint": str(d0_checkpoint),
        "d0_checkpoint_sha256": _sha256_file(d0_checkpoint),
        "transfer": transfer,
        "parameter_counts": {
            "total": sum(parameter.numel() for parameter in candidate.parameters()),
            "correction": sum(parameter.numel() for parameter in head.correction.parameters()),
        },
        "zero_output_max_abs_diff": zero_difference,
        "active_output_max_abs_diff": active_difference,
        "gates": gates,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "training_authorized": False,
        "test_access_authorized": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Static D0-preservation audit for ACMC")
    parser.add_argument("--model-yaml", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nc", type=int, default=21)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args()
    result = static_ambiguity_multilevel_audit(
        args.model_yaml,
        args.d0_checkpoint,
        args.output,
        nc=args.nc,
        image_size=args.image_size,
        config={"hidden_dim": args.hidden_dim},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
