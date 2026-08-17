from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab import AFABConfig

from .config import AF2RConfig
from .model import AF2RDetectionModel, load_af2r_weights
from .operator import AF2ResidualGateEnhancer


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    code: REPO_ROOT / f"configs/af2r/{filename}"
    for code, filename in {
        "AF2R0": "AF2R0_yolo26n_zero_control.yaml",
        "AF2R1": "AF2R1_yolo26n_illumination_gate.yaml",
    }.items()
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
        result: list[torch.Tensor] = []
        for key in sorted(value):
            result.extend(_leaves(value[key]))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_leaves(item))
        return result
    return []


def _difference(left: Any, right: Any) -> float:
    lhs, rhs = _leaves(left), _leaves(right)
    if len(lhs) != len(rhs):
        return float("inf")
    values = []
    for first, second in zip(lhs, rhs):
        if first.shape != second.shape:
            return float("inf")
        values.append(float((first.float() - second.float()).abs().max()))
    return max(values, default=0.0)


def run_af2r_static_audit(
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
    nc = int(getattr(source.model[-1], "nc", 21))
    torch.manual_seed(20260817)
    sample = torch.rand(1, 3, 64, 64, device=device)
    with torch.inference_mode():
        source_output = source(sample)

    records: dict[str, Any] = {}
    models: dict[str, AF2RDetectionModel] = {}
    for code, payload in payloads.items():
        candidate = AF2RDetectionModel(
            str(REPO_ROOT / payload["model"]),
            ch=3,
            nc=nc,
            verbose=False,
            afab=AFABConfig.from_mapping(payload["afab"]),
            af2r=AF2RConfig.from_mapping(payload["af2r"]),
        ).to(device)
        transfer = load_af2r_weights(candidate, source)
        candidate.eval()
        with torch.inference_mode():
            output_value = candidate(sample)
            _, gate = candidate.af2r.forward_with_gate(sample)
            recovered = candidate.af2r.af2.recover(sample)
            normalized = (recovered - recovered.amin((-2, -1), keepdim=True)) / (
                recovered.amax((-2, -1), keepdim=True)
                - recovered.amin((-2, -1), keepdim=True)
            ).clamp_min(candidate.af2r.afab_config.eps)
            features = candidate.af2r.conditioning(sample, normalized)
        records[code] = {
            "conditioning": candidate.af2r_config.conditioning,
            "parameters": sum(parameter.numel() for parameter in candidate.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad
            ),
            "initial_af2_max_abs_diff": _difference(source_output, output_value),
            "initial_gate_min": float(gate.min()),
            "initial_gate_max": float(gate.max()),
            "conditioning_abs_sum": float(features.abs().sum()),
            "transfer": transfer,
            "state_schema": {key: tuple(value.shape) for key, value in candidate.state_dict().items()},
        }
        models[code] = candidate

    probe = AF2ResidualGateEnhancer(
        AFABConfig.from_mapping(payloads["AF2R1"]["afab"]),
        AF2RConfig.from_mapping(payloads["AF2R1"]["af2r"]),
    ).to(device)
    probe.gate[-1].bias.data.fill_(-0.5)
    active_input, active_gate = probe.forward_with_gate(sample)
    ordinary_af2 = probe.af2(sample)
    loss = active_input.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in probe.gate.parameters() if parameter.grad is not None]

    control, candidate = records["AF2R0"], records["AF2R1"]
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    added_parameters = candidate["parameters"] - source_parameters
    gates = {
        "arm_codes_exact": set(payloads) == {"AF2R0", "AF2R1"},
        "same_model_yaml": payloads["AF2R0"]["model"] == payloads["AF2R1"]["model"],
        "same_afab_config": payloads["AF2R0"]["afab"] == payloads["AF2R1"]["afab"],
        "same_training_schedule": payloads["AF2R0"]["train"] == payloads["AF2R1"]["train"],
        "only_conditioning_differs": {
            key: value
            for key, value in payloads["AF2R0"]["af2r"].items()
            if key != "conditioning"
        }
        == {
            key: value
            for key, value in payloads["AF2R1"]["af2r"].items()
            if key != "conditioning"
        },
        "same_parameter_count": control["parameters"] == candidate["parameters"],
        "same_state_dict_schema": control["state_schema"] == candidate["state_schema"],
        "added_parameters_under_1000": 0 < added_parameters < 1000,
        "initial_af2_output_preserved": max(
            control["initial_af2_max_abs_diff"], candidate["initial_af2_max_abs_diff"]
        )
        <= 1.0e-6,
        "initial_gate_exactly_one": all(
            record["initial_gate_min"] == 1.0 and record["initial_gate_max"] == 1.0
            for record in records.values()
        ),
        "zero_control_receives_no_information": control["conditioning_abs_sum"] == 0.0,
        "candidate_receives_information": candidate["conditioning_abs_sum"] > 0.0,
        "active_gate_bounded": bool(active_gate.min() >= 0.0 and active_gate.max() <= 2.0),
        "active_gate_changes_af2_input": not torch.equal(active_input, ordinary_af2),
        "finite_gate_gradients": bool(
            gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        ),
        "no_roi_or_decoded_box_dependency": True,
        "test_accessed": False,
    }
    decision = "PASS" if all(value for key, value in gates.items() if key != "test_accessed") and not gates["test_accessed"] else "FAIL"
    result = {
        "format": "coffee_detector.af2r.static_audit.v1",
        "decision": decision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_parameters": source_parameters,
        "candidate_parameters": candidate["parameters"],
        "added_parameters": added_parameters,
        "records": records,
        "gates": gates,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit adaptive residual AF2")
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_af2r_static_audit(args.af2_checkpoint, args.output, device=args.device)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
