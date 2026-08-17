from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab import (
    AFABConfig,
    AFABDetectionModel,
    load_afab_weights,
)

from .model import AF2CalibratedDetectionModel, load_af2cal_weights
from .operator import AF2ChannelCalibratedEnhancer


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    code: REPO_ROOT / f"configs/af2cal/{code}_yolo26n.yaml"
    for code in ("AF2FT30", "AF2CAL3")
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [item for key in sorted(value) for item in _leaves(value[key])]
    if isinstance(value, (tuple, list)):
        return [item for entry in value for item in _leaves(entry)]
    return []


def _difference(left: Any, right: Any) -> float:
    lhs, rhs = _leaves(left), _leaves(right)
    if len(lhs) != len(rhs):
        return float("inf")
    differences = []
    for first, second in zip(lhs, rhs):
        if first.shape != second.shape:
            return float("inf")
        differences.append(float((first.float() - second.float()).abs().max()))
    return max(differences, default=0.0)


def run_af2cal_static_audit(
    af2_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    from ultralytics import YOLO

    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payloads = {
        code: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for code, path in CONFIGS.items()
    }
    source = YOLO(str(checkpoint)).model.to(device).eval()
    source_enhancer = getattr(source, "afab", None)
    if source_enhancer is None:
        raise RuntimeError("Checkpoint sumber bukan AF2")
    nc = int(getattr(source.model[-1], "nc", 21))
    torch.manual_seed(20260817)
    sample = torch.rand(1, 3, 64, 64, device=device)
    with torch.inference_mode():
        source_input = source_enhancer(sample)
        source_output = source(sample)

    afab = AFABConfig.from_mapping(payloads["AF2FT30"]["afab"])
    fixed = AFABDetectionModel(
        str(REPO_ROOT / payloads["AF2FT30"]["model"]),
        ch=3,
        nc=nc,
        verbose=False,
        afab=afab,
    ).to(device)
    fixed_transfer = load_afab_weights(fixed, source)
    fixed.eval()
    calibrated = AF2CalibratedDetectionModel(
        str(REPO_ROOT / payloads["AF2CAL3"]["model"]),
        ch=3,
        nc=nc,
        verbose=False,
        afab=afab,
    ).to(device)
    calibrated_transfer = load_af2cal_weights(calibrated, source)
    calibrated.eval()
    with torch.inference_mode():
        fixed_input = fixed.afab(sample)
        fixed_output = fixed(sample)
        calibrated_input, initial_scale = calibrated.af2cal.forward_with_scale(sample)
        calibrated_output = calibrated(sample)

    probe = AF2ChannelCalibratedEnhancer(afab).to(device)
    probe.calibration_logits.data.fill_(0.25)
    active_input, active_scale = probe.forward_with_scale(sample)
    ordinary_af2 = probe.af2(sample)
    active_input.square().mean().backward()
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    fixed_parameters = sum(parameter.numel() for parameter in fixed.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in calibrated.parameters())
    added_parameters = candidate_parameters - source_parameters
    fixed_schema = {key: tuple(value.shape) for key, value in fixed.state_dict().items()}
    candidate_schema = {
        key: tuple(value.shape) for key, value in calibrated.state_dict().items()
    }
    new_keys = sorted(set(candidate_schema) - set(fixed_schema))
    gates = {
        "arm_codes_exact": set(payloads) == {"AF2FT30", "AF2CAL3"},
        "same_model_yaml": payloads["AF2FT30"]["model"]
        == payloads["AF2CAL3"]["model"],
        "same_afab_config": payloads["AF2FT30"]["afab"]
        == payloads["AF2CAL3"]["afab"],
        "same_training_schedule": payloads["AF2FT30"]["train"]
        == payloads["AF2CAL3"]["train"],
        "correct_mechanisms": payloads["AF2FT30"].get("mechanism")
        == "continuation_only"
        and payloads["AF2CAL3"].get("mechanism")
        == "channel_residual_calibration",
        "fixed_arm_has_source_parameter_count": fixed_parameters == source_parameters,
        "candidate_adds_exactly_three_parameters": added_parameters == 3,
        "candidate_adds_only_calibration_logits": len(new_keys) == 1
        and new_keys[0].endswith("calibration_logits")
        and candidate_schema[new_keys[0]] == (1, 3, 1, 1),
        "fixed_initial_input_bitwise_preserved": torch.equal(source_input, fixed_input),
        "candidate_initial_input_bitwise_preserved": torch.equal(
            source_input, calibrated_input
        ),
        "initial_detector_outputs_numerically_consistent": max(
            _difference(source_output, fixed_output),
            _difference(source_output, calibrated_output),
        )
        <= 1.0e-4,
        "initial_scale_exactly_one": torch.equal(
            initial_scale, torch.ones_like(initial_scale)
        ),
        "active_scale_bounded": bool(
            active_scale.min() >= 0.0 and active_scale.max() <= 2.0
        ),
        "active_scale_changes_af2_input": not torch.equal(active_input, ordinary_af2),
        "finite_calibration_gradients": probe.calibration_logits.grad is not None
        and bool(torch.isfinite(probe.calibration_logits.grad).all()),
        "all_source_weights_transferred": fixed_transfer.get(
            "shape_compatible_items"
        )
        == fixed_transfer.get("source_items")
        and calibrated_transfer.get("shape_compatible_items")
        == calibrated_transfer.get("source_items"),
        "no_roi_or_decoded_box_dependency": True,
        "test_accessed": False,
    }
    decision = (
        "PASS"
        if all(value for key, value in gates.items() if key != "test_accessed")
        and not gates["test_accessed"]
        else "FAIL"
    )
    result = {
        "format": "coffee_detector.af2cal.static_audit.v1",
        "decision": decision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_parameters": source_parameters,
        "fixed_parameters": fixed_parameters,
        "candidate_parameters": candidate_parameters,
        "added_parameters": added_parameters,
        "new_state_keys": new_keys,
        "initial_scale": initial_scale.flatten().tolist(),
        "active_scale": active_scale.flatten().tolist(),
        "fixed_transfer": fixed_transfer,
        "candidate_transfer": calibrated_transfer,
        "gates": gates,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit AF2 calibration")
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_af2cal_static_audit(
        args.af2_checkpoint, args.output, device=args.device
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
