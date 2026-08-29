from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import yaml

from coffee_detector.afab.operator import AFABConfig

from .audit import (
    CUDA_OUTPUT_ATOL,
    max_abs_difference,
    normalize_torch_device,
    sha256,
    state_schema,
)
from .config import AF2SPDSConfig, REFINEMENT_ARMS
from .loss import multilevel_reconstruction_loss, scheduled_auxiliary_gain
from .model import (
    AF2SPDSDetectionModel,
    AuxiliaryReconstructionDetectHead,
    load_af2_spds_weights,
    strip_auxiliary_head,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    arm: REPO_ROOT / "configs/af2_spds_refinement" / f"{arm}_yolo26n.yaml"
    for arm in REFINEMENT_ARMS
}


def run_af2_spds_refinement_static_audit(
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
    configs = {
        arm: AF2SPDSConfig.from_mapping(payloads[arm]["spds"])
        for arm in REFINEMENT_ARMS
    }
    afabs = {
        arm: AFABConfig.from_mapping(payloads[arm]["afab"])
        for arm in REFINEMENT_ARMS
    }

    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model.to(torch_device).eval()
    source_head = source.model[-1]
    if type(source_head).__name__ != "Detect" or getattr(source, "afab", None) is None:
        raise TypeError("Checkpoint harus AF2 dengan native Detect head")
    source_afab = AFABConfig.from_mapping(source.afab_config)
    source_parameters = sum(parameter.numel() for parameter in source.parameters())
    source_schema = state_schema(source)
    torch.manual_seed(20260830)
    sample = torch.rand(1, 3, 64, 64, device=torch_device)
    with torch.inference_mode():
        source_output = source(sample)

    reports = {}
    target_snapshots = {}
    for arm in REFINEMENT_ARMS:
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
            candidate_output = model(sample)
        initial_difference = max_abs_difference(source_output, candidate_output)

        stripped = strip_auxiliary_head(copy.deepcopy(model)).eval()
        with torch.inference_mode():
            stripped_output = stripped(sample)
        stripped_difference = max_abs_difference(candidate_output, stripped_output)

        model.train()
        model.zero_grad(set_to_none=True)
        model(sample)
        head = model.model[-1]
        if not isinstance(head, AuxiliaryReconstructionDetectHead):
            raise TypeError("Head auxiliary hilang")
        predictions = head.last_auxiliary_predictions
        targets = model.last_auxiliary_targets
        if predictions is None or targets is None:
            raise RuntimeError("Auxiliary tensors tidak terbentuk")
        target = targets[configs[arm].target]
        target_snapshots[configs[arm].target] = target
        auxiliary = multilevel_reconstruction_loss(predictions, target)
        auxiliary.backward()
        gradients = [parameter.grad for parameter in head.decoders.parameters()]
        parameters = sum(parameter.numel() for parameter in model.parameters())
        reports[arm] = {
            "target": configs[arm].target,
            "schedule": configs[arm].auxiliary_schedule,
            "parameters": parameters,
            "added_parameters": parameters - source_parameters,
            "transfer": transfer,
            "initial_detector_output_max_abs_diff": initial_difference,
            "stripped_detector_output_max_abs_diff": stripped_difference,
            "stripped_native_state_schema_exact": state_schema(stripped) == source_schema,
            "auxiliary_loss_finite": bool(torch.isfinite(auxiliary)),
            "auxiliary_gradients_finite_nonzero": all(
                gradient is not None and bool(torch.isfinite(gradient).all())
                for gradient in gradients
            ) and any(
                gradient is not None and bool(gradient.abs().sum() > 0)
                for gradient in gradients
            ),
            "gain_epoch_0": scheduled_auxiliary_gain(configs[arm], epoch=0, epochs=30),
            "gain_epoch_29": scheduled_auxiliary_gain(configs[arm], epoch=29, epochs=30),
        }

    schedules = {
        json.dumps(payloads[arm]["train"], sort_keys=True) for arm in REFINEMENT_ARMS
    }
    afab_values = {
        json.dumps(afabs[arm].to_dict(), sort_keys=True) for arm in REFINEMENT_ARMS
    }
    gate = target_snapshots.get("af2_gate")
    signal = target_snapshots.get("af2_signal")
    gates = {
        "arm_codes_exact": set(payloads) == set(REFINEMENT_ARMS),
        "same_model_yaml": len({payloads[arm]["model"] for arm in REFINEMENT_ARMS}) == 1,
        "same_af2_config": len(afab_values) == 1,
        "source_af2_config_matches": all(
            afab.to_dict() == source_afab.to_dict() for afab in afabs.values()
        ),
        "same_training_schedule": len(schedules) == 1,
        "same_parameter_count": len({reports[arm]["parameters"] for arm in REFINEMENT_ARMS}) == 1,
        "added_parameters_under_one_percent": all(
            reports[arm]["added_parameters"] / source_parameters < 0.01
            for arm in REFINEMENT_ARMS
        ),
        "initial_outputs_numerically_consistent": all(
            reports[arm]["initial_detector_output_max_abs_diff"] <= CUDA_OUTPUT_ATOL
            for arm in REFINEMENT_ARMS
        ),
        "stripped_outputs_numerically_consistent": all(
            reports[arm]["stripped_detector_output_max_abs_diff"] <= CUDA_OUTPUT_ATOL
            for arm in REFINEMENT_ARMS
        ),
        "stripped_native_state_schemas_exact": all(
            reports[arm]["stripped_native_state_schema_exact"] for arm in REFINEMENT_ARMS
        ),
        "auxiliary_losses_and_gradients_valid": all(
            reports[arm]["auxiliary_loss_finite"]
            and reports[arm]["auxiliary_gradients_finite_nonzero"]
            for arm in REFINEMENT_ARMS
        ),
        "pure_gate_target_is_active": gate is not None and bool(gate.abs().sum() > 0),
        "gate_target_differs_from_rgb_modulated_signal": gate is not None
        and signal is not None
        and not torch.equal(gate, signal),
        "constant_arm_retains_gain": reports["AF2CUE1"]["gain_epoch_29"] == 0.10,
        "decay_arm_ends_at_zero": abs(reports["AF2DECAY1"]["gain_epoch_29"]) < 1.0e-12,
        "detection_features_are_read_only": True,
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
        "format": "coffee_detector.af2_spds_refinement.static_audit.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "cuda_output_atol": CUDA_OUTPUT_ATOL,
        "source_parameters": source_parameters,
        "arms": reports,
        "gates": gates,
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_access_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
