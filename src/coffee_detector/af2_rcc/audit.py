"""Static identity, causality, and safety audit for AF2-RCC."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab import AFABConfig

from .model import (
    AF2RCCConfig,
    AF2RCCDetectHead,
    AF2RCCDetectionModel,
    freeze_for_rcc,
    load_af2_rcc_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    code: REPO_ROOT / f"configs/af2_rcc/{code}_yolo26n.yaml"
    for code in ("AF2RCC0", "AF2RCC1")
}

# A wrapped and an unwrapped YOLO26 head may select different but equivalent
# CUDA kernels.  T4 FP32 reductions then differ by a few ulps even when the
# transferred state and zero calibration are exact.  Direct-head identity is
# still required bitwise below; full-model identity uses an explicit numerical
# tolerance that is much smaller than the active RCC correction.
FULL_MODEL_ATOL = 5.0e-5
FULL_MODEL_RTOL = 1.0e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_schema(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def _first_conv_channels(module: torch.nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d):
            return int(child.in_channels)
    raise TypeError(f"Tidak menemukan Conv2d pada {type(module).__name__}")


def run_af2_rcc_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 64,
) -> dict[str, Any]:
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
    nc = int(source.model[-1].nc)
    torch.manual_seed(20260822)
    image = torch.rand(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        native = source(image)

    records: dict[str, Any] = {}
    models: dict[str, AF2RCCDetectionModel] = {}
    for code, payload in payloads.items():
        config = AF2RCCConfig.from_mapping(payload["af2_rcc"])
        model = AF2RCCDetectionModel(
            str(REPO_ROOT / payload["model"]),
            nc=nc,
            verbose=False,
            afab=AFABConfig.from_mapping(payload["afab"]),
            af2_rcc=config,
        ).to(device)
        transfer = load_af2_rcc_weights(model, source)
        model.eval()
        with torch.inference_mode():
            identity = model(image)
        head = model.model[-1]
        boxes_equal = torch.equal(
            identity[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"]
        )
        scores_equal = torch.equal(
            identity[1]["one2one"]["scores"], native[1]["one2one"]["scores"]
        )
        boxes_close = torch.allclose(
            identity[1]["one2one"]["boxes"],
            native[1]["one2one"]["boxes"],
            atol=FULL_MODEL_ATOL,
            rtol=FULL_MODEL_RTOL,
        )
        scores_close = torch.allclose(
            identity[1]["one2one"]["scores"],
            native[1]["one2one"]["scores"],
            atol=FULL_MODEL_ATOL,
            rtol=FULL_MODEL_RTOL,
        )
        numerical_diff = max(
            float(
                (identity[1]["one2one"]["boxes"] - native[1]["one2one"]["boxes"])
                .float()
                .abs()
                .max()
            ),
            float(
                (identity[1]["one2one"]["scores"] - native[1]["one2one"]["scores"])
                .float()
                .abs()
                .max()
            ),
        )

        # Direct head probe proves exact score identity and geometry isolation.
        feature_channels = [
            _first_conv_channels(module) for module in head.base_head.cv2
        ]
        features = [
            torch.rand(1, channels, size, size, device=device)
            for channels, size in zip(feature_channels, (16, 8, 4))
        ]
        cue = torch.rand(1, 3, 128, 128, device=device)
        head.set_spatial_cue(cue)
        with torch.inference_mode():
            wrapped = head([item.clone() for item in features])
            native_head = head.base_head([item.clone() for item in features])
        direct_boxes_equal = torch.equal(
            wrapped[1]["one2one"]["boxes"], native_head[1]["one2one"]["boxes"]
        )
        direct_scores_equal = torch.equal(
            wrapped[1]["one2one"]["scores"], native_head[1]["one2one"]["scores"]
        )

        with torch.no_grad():
            for calibrator in head.calibrators:
                calibrator.weight.fill_(0.05)
        head.set_spatial_cue(cue)
        with torch.inference_mode():
            active = head([item.clone() for item in features])
        active_boxes_equal = torch.equal(
            active[1]["one2one"]["boxes"], native_head[1]["one2one"]["boxes"]
        )
        active_score_diff = float(
            (active[1]["one2one"]["scores"] - native_head[1]["one2one"]["scores"])
            .abs()
            .max()
        )
        with torch.no_grad():
            for calibrator in head.calibrators:
                calibrator.weight.zero_()
        freeze = freeze_for_rcc(model)
        model.train()
        model.zero_grad(set_to_none=True)
        scores = model(image)["one2many"]["scores"]
        scores.square().mean().backward()
        gradients = [p.grad for p in head.calibrators.parameters()]
        finite_nonzero_gradients = all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        ) and any(bool(gradient.abs().max() > 0) for gradient in gradients if gradient is not None)
        models[code] = model
        records[code] = {
            "conditioning": config.conditioning,
            "parameters": sum(p.numel() for p in model.parameters()),
            "state_schema": _state_schema(model),
            "trainable_parameters": freeze["trainable"],
            "transfer": transfer,
            "identity_full_model_bitwise": boxes_equal and scores_equal,
            "identity_full_model_numerically_close": boxes_close and scores_close,
            "identity_full_model_max_abs_diff": numerical_diff,
            "identity_direct_boxes_bitwise": direct_boxes_equal,
            "identity_direct_scores_bitwise": direct_scores_equal,
            "active_boxes_bitwise": active_boxes_equal,
            "active_score_max_abs_diff": active_score_diff,
            "finite_nonzero_calibration_gradients": finite_nonzero_gradients,
        }

    control, candidate = records["AF2RCC0"], records["AF2RCC1"]
    source_parameters = sum(p.numel() for p in source.parameters())
    added = candidate["parameters"] - source_parameters
    model_source = inspect.getsource(AF2RCCDetectionModel)
    head_source = inspect.getsource(AF2RCCDetectHead)
    gates = {
        "same_model_yaml": payloads["AF2RCC0"]["model"] == payloads["AF2RCC1"]["model"],
        "same_af2_config": payloads["AF2RCC0"]["afab"] == payloads["AF2RCC1"]["afab"],
        "same_training_schedule": payloads["AF2RCC0"]["train"] == payloads["AF2RCC1"]["train"],
        "only_conditioning_differs": {
            key: value for key, value in payloads["AF2RCC0"]["af2_rcc"].items() if key != "conditioning"
        } == {
            key: value for key, value in payloads["AF2RCC1"]["af2_rcc"].items() if key != "conditioning"
        },
        "same_parameter_count": control["parameters"] == candidate["parameters"],
        "same_state_schema": control["state_schema"] == candidate["state_schema"],
        "added_parameters_exactly_189": added == 3 * nc * 3 == 189,
        "trainable_parameters_exactly_189": candidate["trainable_parameters"] == 189,
        "zero_control_identity": control["identity_direct_scores_bitwise"],
        "candidate_initial_identity": candidate["identity_direct_scores_bitwise"],
        "full_model_numerically_identical": (
            control["identity_full_model_numerically_close"]
            and candidate["identity_full_model_numerically_close"]
        ),
        "candidate_receives_recovered_cue": payloads["AF2RCC1"]["af2_rcc"]["conditioning"] == "recovered",
        "active_changes_scores": candidate["active_score_max_abs_diff"] > 0.0,
        "active_preserves_boxes": candidate["active_boxes_bitwise"],
        "finite_nonzero_calibration_gradients": candidate["finite_nonzero_calibration_gradients"],
        "one_af2_recovery_call": model_source.count(".recover(") == 1,
        "no_additional_fft": "torch.fft" not in model_source and "torch.fft" not in head_source,
        "no_roi_or_decoded_box_dependency": "roi" not in head_source.lower() and "decode" not in head_source.lower(),
        "classification_path_only": candidate["active_boxes_bitwise"],
        "test_accessed": False,
    }
    decision = "PASS" if all(
        value for key, value in gates.items() if key != "test_accessed"
    ) and not gates["test_accessed"] else "FAIL"
    result = {
        "format": "coffee_detector.af2_rcc.static_audit.v1",
        "decision": decision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_parameters": source_parameters,
        "candidate_parameters": candidate["parameters"],
        "added_parameters": added,
        "full_model_identity_tolerance": {
            "atol": FULL_MODEL_ATOL,
            "rtol": FULL_MODEL_RTOL,
        },
        "records": records,
        "gates": gates,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit AF2-RCC")
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()
    result = run_af2_rcc_static_audit(
        args.af2_checkpoint, args.output, device=args.device, image_size=args.image_size
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
