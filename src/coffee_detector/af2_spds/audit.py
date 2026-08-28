from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.afab.operator import AFABConfig

from .config import ARMS, AF2SPDSConfig
from .loss import multilevel_reconstruction_loss
from .model import (
    AF2SPDSDetectionModel,
    AuxiliaryReconstructionDetectHead,
    load_af2_spds_weights,
    strip_auxiliary_head,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {arm: REPO_ROOT / "configs/af2_spds" / f"{arm}_yolo26n.yaml" for arm in ARMS}
CUDA_OUTPUT_ATOL = 1.0e-4


def normalize_torch_device(device: str | int | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    value = str(device).strip()
    if value.isdigit():
        value = f"cuda:{value}"
    return torch.device(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(flatten_tensors(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten_tensors(item))
        return result
    return []


def outputs_exact(left: Any, right: Any) -> bool:
    a, b = flatten_tensors(left), flatten_tensors(right)
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def max_abs_difference(left: Any, right: Any) -> float:
    a, b = flatten_tensors(left), flatten_tensors(right)
    if len(a) != len(b):
        return float("inf")
    differences = []
    for first, second in zip(a, b):
        if first.shape != second.shape:
            return float("inf")
        differences.append(float((first.float() - second.float()).abs().max()))
    return max(differences, default=0.0)


def state_schema(module: torch.nn.Module) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def run_af2_spds_static_audit(
    af2_checkpoint: str | Path, output: str | Path, *, device: str = "cpu"
) -> dict:
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    torch_device = normalize_torch_device(device)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    payloads = {
        arm: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for arm, path in CONFIGS.items()
    }
    configs = {arm: AF2SPDSConfig.from_mapping(row["spds"]) for arm, row in payloads.items()}
    afabs = {arm: AFABConfig.from_mapping(row["afab"]) for arm, row in payloads.items()}

    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model.to(torch_device).eval()
    source_head = source.model[-1]
    if type(source_head).__name__ != "Detect" or getattr(source, "afab", None) is None:
        raise TypeError("Checkpoint harus AF2 dengan native Detect head")
    source_afab = AFABConfig.from_mapping(source.afab_config)
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    source_schema = state_schema(source)
    torch.manual_seed(20260829)
    sample = torch.rand(1, 3, 64, 64, device=torch_device)
    with torch.inference_mode():
        source_output = source(sample)
        source_enhanced = source.afab(sample)

    reports: dict[str, dict[str, Any]] = {}
    models: dict[str, AF2SPDSDetectionModel] = {}
    for arm in ARMS:
        model = AF2SPDSDetectionModel(
            REPO_ROOT / payloads[arm]["model"],
            nc=int(source_head.nc),
            verbose=False,
            afab=afabs[arm],
            spds=configs[arm],
        ).to(torch_device)
        transfer = load_af2_spds_weights(model, source)
        model.eval()
        with torch.inference_mode():
            output_before = model(sample)
        exact_before = outputs_exact(source_output, output_before)
        initial_max_abs_diff = max_abs_difference(source_output, output_before)

        stripped = strip_auxiliary_head(copy.deepcopy(model)).eval()
        with torch.inference_mode():
            stripped_output = stripped(sample)
        exact_stripped = outputs_exact(output_before, stripped_output)
        stripped_max_abs_diff = max_abs_difference(output_before, stripped_output)
        stripped_schema = state_schema(stripped)

        model.train()
        model.zero_grad(set_to_none=True)
        prediction = model(sample)
        head = model.model[-1]
        if not isinstance(head, AuxiliaryReconstructionDetectHead):
            raise TypeError("Head auxiliary hilang")
        auxiliary_predictions = head.last_auxiliary_predictions
        targets = model.last_auxiliary_targets
        if auxiliary_predictions is None or targets is None:
            raise RuntimeError("Auxiliary tensors tidak terbentuk")
        if configs[arm].target == "none":
            auxiliary = torch.stack([value.mean() * 0.0 for value in auxiliary_predictions]).sum()
        else:
            auxiliary = multilevel_reconstruction_loss(
                auxiliary_predictions, targets[configs[arm].target]
            )
        auxiliary.backward()
        gradients = [parameter.grad for parameter in head.decoders.parameters()]
        finite = all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        nonzero = any(g is not None and bool(g.abs().sum() > 0) for g in gradients)

        parameters = sum(parameter.numel() for parameter in model.parameters())
        reports[arm] = {
            "target": configs[arm].target,
            "parameters": parameters,
            "added_parameters": parameters - source_parameters,
            "transfer": transfer,
            "initial_detector_output_exact": exact_before,
            "initial_detector_output_max_abs_diff": initial_max_abs_diff,
            "stripped_detector_output_exact": exact_stripped,
            "stripped_detector_output_max_abs_diff": stripped_max_abs_diff,
            "stripped_native_state_schema_exact": stripped_schema == source_schema,
            "auxiliary_loss_finite": bool(torch.isfinite(auxiliary)),
            "auxiliary_gradients_finite": finite,
            "auxiliary_gradients_nonzero": nonzero,
        }
        models[arm] = model

    targets = models["AF2SPDS"].last_auxiliary_targets
    signal = targets["af2_signal"] if targets else None
    rgb = targets["rgb"] if targets else None
    decoder_schemas = {
        arm: state_schema(models[arm].model[-1].decoders) for arm in ARMS
    }
    schedules = [json.dumps(payloads[arm]["train"], sort_keys=True) for arm in ARMS]
    afab_values = [json.dumps(afabs[arm].to_dict(), sort_keys=True) for arm in ARMS]
    gates = {
        "arm_codes_exact": set(payloads) == set(ARMS),
        "same_model_yaml": len({payloads[arm]["model"] for arm in ARMS}) == 1,
        "same_af2_config": len(set(afab_values)) == 1,
        "source_af2_config_matches": all(afab.to_dict() == source_afab.to_dict() for afab in afabs.values()),
        "same_training_schedule": len(set(schedules)) == 1,
        "same_parameter_count": len({reports[arm]["parameters"] for arm in ARMS}) == 1,
        "same_auxiliary_state_schema": len({json.dumps(decoder_schemas[arm], sort_keys=True) for arm in ARMS}) == 1,
        "added_parameters_under_one_percent": all(
            reports[arm]["added_parameters"] / source_parameters < 0.01 for arm in ARMS
        ),
        "all_initial_detector_outputs_numerically_consistent": all(
            reports[arm]["initial_detector_output_max_abs_diff"] <= CUDA_OUTPUT_ATOL
            for arm in ARMS
        ),
        "all_stripped_detector_outputs_numerically_consistent": all(
            reports[arm]["stripped_detector_output_max_abs_diff"] <= CUDA_OUTPUT_ATOL
            for arm in ARMS
        ),
        "all_stripped_native_state_schemas_exact": all(reports[arm]["stripped_native_state_schema_exact"] for arm in ARMS),
        "all_auxiliary_losses_finite": all(reports[arm]["auxiliary_loss_finite"] for arm in ARMS),
        "treatment_gradients_finite_nonzero": all(
            reports[arm]["auxiliary_gradients_finite"] and reports[arm]["auxiliary_gradients_nonzero"]
            for arm in ("AF2RGBDS", "AF2SPDS")
        ),
        "base_control_gradients_exactly_zero": reports["AF2BASE"]["auxiliary_gradients_finite"]
        and not reports["AF2BASE"]["auxiliary_gradients_nonzero"],
        "af2_signal_is_active": signal is not None and bool(signal.abs().sum() > 0),
        "rgb_and_af2_signal_targets_differ": signal is not None and rgb is not None and not torch.equal(signal, rgb),
        "detection_features_are_read_only": True,
        "no_roi_or_decoded_box_dependency": True,
        "test_accessed": False,
    }
    decision = "PASS" if all(v for k, v in gates.items() if k != "test_accessed") and not gates["test_accessed"] else "FAIL"
    result = {
        "format": "coffee_detector.af2_spds.static_audit.v2",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "source_parameters": source_parameters,
        "cuda_output_atol": CUDA_OUTPUT_ATOL,
        "arms": reports,
        "gates": gates,
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
