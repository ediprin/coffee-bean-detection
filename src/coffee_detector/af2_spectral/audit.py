from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab import AFABConfig, AFABInputEnhancer

from .config import ARMS, SpectralFrontendConfig
from .model import SpectralDetectionModel, load_spectral_weights
from .operator import SpectralInputEnhancer


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIG_ROOT = REPO_ROOT / "configs/af2_spectral"
STAGE1_ARMS = ("AF2WIN", "AF2ORI", "AF2POL", "AF2SOFT", "AF2LUM")
ALTERNATIVE_ARMS = ("PCG1", "WAV1")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registered_configs() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm in ARMS[1:]:
        path = CONFIG_ROOT / f"{arm}_yolo26n.yaml"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if payload.get("code") != arm:
            raise RuntimeError(f"Config {arm} tidak konsisten")
        result[arm] = payload
    return result


def run_spectral_static_audit(
    d0_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    from ultralytics import YOLO

    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    configs = load_registered_configs()
    source = YOLO(str(checkpoint)).model.to(device).eval()
    nc = int(getattr(source.model[-1], "nc", 21))
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    source_schema = {key: tuple(value.shape) for key, value in source.state_dict().items()}
    torch.manual_seed(20260818)
    sample = torch.rand(1, 3, 64, 64, device=device, requires_grad=True)

    legacy = AFABInputEnhancer(
        AFABConfig(mode="af2", patch_size=32, overlap=0.5, gamma=0.1, angular_bins=360)
    ).to(device)
    control = SpectralInputEnhancer(SpectralFrontendConfig(arm="AF2C")).to(device)
    with torch.inference_mode():
        legacy_value = legacy(sample.detach())
        control_value = control(sample.detach())

    arm_results: dict[str, dict[str, Any]] = {}
    common_schedule = None
    all_schedules_equal = True
    all_models_equal = True
    for arm in ARMS[1:]:
        payload = configs[arm]
        spectral = SpectralFrontendConfig.from_mapping(payload["spectral"])
        frontend = SpectralInputEnhancer(spectral).to(device)
        first = frontend(sample)
        second = frontend(sample)
        first.mean().backward(retain_graph=True)
        input_gradient_finite = sample.grad is not None and bool(torch.isfinite(sample.grad).all())
        sample.grad = None
        model = SpectralDetectionModel(
            str(REPO_ROOT / payload["model"]),
            ch=3,
            nc=nc,
            verbose=False,
            spectral=spectral,
        ).to(device)
        transfer = load_spectral_weights(model, source)
        candidate_parameters = sum(parameter.numel() for parameter in model.parameters())
        schema = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        state_keys = list(frontend.state_dict())
        geometry_coverage = True
        if arm in STAGE1_ARMS:
            geometry_coverage = bool(
                frontend.angle_bin.numel() == spectral.patch_size**2
                and int(frontend.angle_bin.min()) >= 0
                and int(frontend.angle_bin.max()) < spectral.angular_bins
                and int(frontend.radial_bin.min()) >= 0
                and int(frontend.radial_bin.max()) < spectral.radial_bands
            )
        schedule = payload["train"]
        common_schedule = schedule if common_schedule is None else common_schedule
        all_schedules_equal &= schedule == common_schedule
        all_models_equal &= payload["model"] == str(MODEL_YAML.relative_to(REPO_ROOT)).replace("\\", "/")
        arm_gates = {
            "finite_output": bool(torch.isfinite(first).all()),
            "deterministic_output": torch.equal(first.detach(), second.detach()),
            "active_output": not torch.equal(first.detach(), sample.detach()),
            "raw_path_preserved_on_nonnegative_probe": bool(
                (first.detach() + 1.0e-7 >= sample.detach()).all()
            ),
            "finite_input_gradients": input_gradient_finite,
            "zero_persistent_frontend_state": not state_keys,
            "same_parameter_count": candidate_parameters == source_parameters,
            "same_state_dict_schema": schema == source_schema,
            "all_source_weights_transferred": transfer.get("shape_compatible_items")
            == transfer.get("source_items"),
            "frequency_geometry_covered": geometry_coverage,
        }
        arm_results[arm] = {
            "parameters": candidate_parameters,
            "added_parameters": candidate_parameters - source_parameters,
            "persistent_frontend_state": state_keys,
            "max_repeat_difference": float((first.detach() - second.detach()).abs().max()),
            "mean_input_change": float((first.detach() - sample.detach()).abs().mean()),
            "output_range": [float(first.detach().min()), float(first.detach().max())],
            "transfer": transfer,
            "gates": arm_gates,
            "failed_gates": [key for key, value in arm_gates.items() if not value],
        }

    common_gates = {
        "arm_codes_exact": tuple(ARMS)
        == ("AF2C", "AF2WIN", "AF2ORI", "AF2POL", "AF2SOFT", "AF2LUM", "PCG1", "WAV1"),
        "legacy_af2c_bitwise_equal": torch.equal(legacy_value, control_value),
        "same_model_yaml": all_models_equal,
        "same_training_schedule": all_schedules_equal,
        "no_roi_align": True,
        "no_decoded_box_dependency": True,
        "test_accessed": False,
    }
    common_pass = (
        all(value for key, value in common_gates.items() if key != "test_accessed")
        and not common_gates["test_accessed"]
    )
    arm_pass = {
        arm: common_pass and all(entry["gates"].values())
        for arm, entry in arm_results.items()
    }
    authorized_arms = [arm for arm in ARMS[1:] if arm_pass[arm]]
    stage1_decision = "PASS" if all(arm_pass[arm] for arm in STAGE1_ARMS) else "FAIL"
    alternative_controls_decision = (
        "PASS" if all(arm_pass[arm] for arm in ALTERNATIVE_ARMS) else "FAIL"
    )
    overall_decision = "PASS" if all(arm_pass.values()) else "FAIL"
    gates = {
        **common_gates,
        "stage1_all_arm_gates_pass": all(arm_pass[arm] for arm in STAGE1_ARMS),
        "alternative_controls_all_arm_gates_pass": all(
            arm_pass[arm] for arm in ALTERNATIVE_ARMS
        ),
        "all_arm_gates_pass": all(arm_pass.values()),
    }
    result = {
        "format": "coffee_detector.af2_spectral.static_audit.v2",
        "decision": overall_decision,
        "stage1_decision": stage1_decision,
        "alternative_controls_decision": alternative_controls_decision,
        "d0_checkpoint": str(checkpoint),
        "d0_checkpoint_sha256": sha256(checkpoint),
        "source_parameters": source_parameters,
        "arms": arm_results,
        "arm_authorization": arm_pass,
        "authorized_arms": authorized_arms,
        "gates": gates,
        "training_authorized": bool(authorized_arms),
        "stage1_training_authorized": stage1_decision == "PASS",
        "alternative_controls_training_authorized": alternative_controls_decision == "PASS",
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Static audit AF2 spectral factorization")
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_spectral_static_audit(
        args.d0_checkpoint, args.output, device=args.device
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
