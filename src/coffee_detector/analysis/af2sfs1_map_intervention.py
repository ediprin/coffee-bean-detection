from __future__ import annotations

import argparse
import json
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coffee_detector.af2_complement.modules import SpaceFrequencySelectionResidual
from coffee_detector.analysis.af2sfs1_root_cause import _adapter_intervention, _read, _validate_arm_report
from coffee_detector.dataset import discover_layout
from coffee_detector.evaluate import _classwise_summary


STATES = ("AF2CTRL", "NORMAL", "BYPASS", "SPATIAL_ONLY", "FREQUENCY_ONLY")
HEADLINE = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _metric_payload(metrics, names: dict[int, str]) -> dict[str, Any]:
    result = {key: float(value) for key, value in metrics.results_dict.items()}
    box = getattr(metrics, "box", None)
    if box is None or getattr(box, "ap", None) is None:
        raise RuntimeError("Validator tidak mengembalikan box AP")
    result.update(_classwise_summary(box, names))
    class_indices = [int(value) for value in np.asarray(box.ap_class_index).reshape(-1)]
    all_ap = getattr(box, "all_ap", None)
    if all_ap is None:
        raise RuntimeError("Validator tidak mengekspos matriks AP per IoU (all_ap)")
    matrix = np.asarray(all_ap, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    by_class = {}
    for row, class_id in enumerate(class_indices):
        if class_id not in names or row >= len(matrix):
            continue
        values = matrix[row]
        by_class[names[class_id]] = {
            "map50_95": float(values.mean()),
            "ap50": float(values[0]),
            "ap75": float(values[min(5, len(values) - 1)]),
        }
    result["ap_by_class_and_iou"] = by_class
    result["macro_ap50"] = float(np.mean([row["ap50"] for row in by_class.values()]))
    result["macro_ap75"] = float(np.mean([row["ap75"] for row in by_class.values()]))
    return result


def _evaluate_state(
    checkpoint: Path,
    data_yaml: Path,
    names: dict[int, str],
    state: str,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    wrapper = YOLO(str(checkpoint))
    adapter = getattr(wrapper.model.model[-1], "adapter", None)
    intervention = {
        "AF2CTRL": None,
        "NORMAL": None,
        "BYPASS": "bypass",
        "SPATIAL_ONLY": "spatial",
        "FREQUENCY_ONLY": "frequency",
    }[state]
    if state != "AF2CTRL" and not isinstance(adapter, SpaceFrequencySelectionResidual):
        raise TypeError("Checkpoint bukan AF2SFS1 SpaceFrequencySelectionResidual")
    context = nullcontext() if intervention is None else _adapter_intervention(adapter, intervention)
    with context:
        metrics = wrapper.val(
            data=str(data_yaml),
            split="val",
            imgsz=640,
            batch=16,
            workers=2,
            device=device,
            plots=False,
            verbose=False,
            project=str(output_root / "validator_runs"),
            name=state.lower(),
            exist_ok=True,
        )
    payload = _metric_payload(metrics, names)
    if payload.get("classes_without_ground_truth"):
        raise RuntimeError(f"State {state} kehilangan kelas validation")
    return payload


def _headline(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in HEADLINE}


def _subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: float(left[key] - right[key]) for key in left}


def decompose_metrics(
    control: dict[str, float], normal: dict[str, float], bypass: dict[str, float]
) -> dict[str, dict[str, float]]:
    total = _subtract(normal, control)
    direct = _subtract(normal, bypass)
    mediated = _subtract(bypass, control)
    error = {key: float(total[key] - direct[key] - mediated[key]) for key in total}
    return {
        "total_normal_minus_control": total,
        "direct_selector_normal_minus_bypass": direct,
        "optimization_mediated_bypass_minus_control": mediated,
        "additivity_error": error,
    }


def _group_name(name: str, head_index: int) -> str:
    if ".adapter." in name:
        return "sfs_adapter"
    if ".base_head.cv2." in name or ".base_head.one2one_cv2." in name:
        return "regression_head"
    if ".base_head.cv3." in name or ".base_head.one2one_cv3." in name:
        return "classification_head"
    match = re.search(r"(?:^|\.)model\.(\d+)\.", name)
    if match and int(match.group(1)) < head_index:
        return "feature_extractor"
    return "other_detector_state"


def _drift_summary(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float | int]:
    parameter_count = sum(left.numel() for left, _right in pairs)
    if not parameter_count:
        return {
            "tensors": 0,
            "parameters": 0,
            "l2_delta": 0.0,
            "relative_l2_delta": 0.0,
            "cosine_similarity": 1.0,
            "mean_absolute_delta": 0.0,
            "maximum_absolute_delta": 0.0,
        }
    delta_sq = source_sq = dot = candidate_sq = absolute_sum = 0.0
    maximum = 0.0
    for left, right in pairs:
        left = left.detach().double().cpu().reshape(-1)
        right = right.detach().double().cpu().reshape(-1)
        delta = right - left
        delta_sq += float(delta.square().sum())
        source_sq += float(left.square().sum())
        candidate_sq += float(right.square().sum())
        dot += float((left * right).sum())
        absolute_sum += float(delta.abs().sum())
        maximum = max(maximum, float(delta.abs().max()))
    denominator = math.sqrt(source_sq * candidate_sq)
    return {
        "tensors": len(pairs),
        "parameters": parameter_count,
        "l2_delta": math.sqrt(delta_sq),
        "relative_l2_delta": math.sqrt(delta_sq) / max(math.sqrt(source_sq), 1e-12),
        "cosine_similarity": dot / max(denominator, 1e-12),
        "mean_absolute_delta": absolute_sum / parameter_count,
        "maximum_absolute_delta": maximum,
    }


def checkpoint_drift(control_model: torch.nn.Module, candidate_model: torch.nn.Module) -> dict[str, Any]:
    control_state = control_model.state_dict()
    candidate_state = candidate_model.state_dict()
    head_index = len(control_model.model) - 1
    groups: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "feature_extractor": [],
        "regression_head": [],
        "classification_head": [],
        "other_detector_state": [],
        "sfs_adapter": [],
    }
    missing = []
    for name, candidate_value in candidate_state.items():
        group = _group_name(name, head_index)
        if group == "sfs_adapter":
            groups[group].append((torch.zeros_like(candidate_value), candidate_value))
            continue
        control_value = control_state.get(name)
        if control_value is None or control_value.shape != candidate_value.shape:
            missing.append(name)
            continue
        if torch.is_floating_point(candidate_value):
            groups[group].append((control_value, candidate_value))
    return {
        "groups": {name: _drift_summary(pairs) for name, pairs in groups.items()},
        "unmatched_candidate_tensors": missing,
    }


def _class_decomposition(
    names: dict[int, str],
    control_metrics: dict[str, Any],
    state_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for name in names.values():
        control = float(control_metrics["map50_95_by_class"][name])
        normal = float(state_metrics["NORMAL"]["map50_95_by_class"][name])
        bypass = float(state_metrics["BYPASS"]["map50_95_by_class"][name])
        spatial = float(state_metrics["SPATIAL_ONLY"]["map50_95_by_class"][name])
        frequency = float(state_metrics["FREQUENCY_ONLY"]["map50_95_by_class"][name])
        rows.append(
            {
                "class_name": name,
                "AF2CTRL": control,
                "NORMAL": normal,
                "BYPASS": bypass,
                "SPATIAL_ONLY": spatial,
                "FREQUENCY_ONLY": frequency,
                "total_gain": normal - control,
                "direct_selector_effect": normal - bypass,
                "optimization_mediated_effect": bypass - control,
            }
        )
    return sorted(rows, key=lambda row: row["total_gain"], reverse=True)


def run_af2sfs1_map_intervention(
    control_checkpoint: str | Path,
    candidate_checkpoint: str | Path,
    control_report: str | Path,
    candidate_report: str | Path,
    root_cause_report: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    from ultralytics import YOLO

    control_checkpoint = Path(control_checkpoint).expanduser().resolve()
    candidate_checkpoint = Path(candidate_checkpoint).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    control_payload = _read(control_report, "AF2CTRL report")
    candidate_payload = _read(candidate_report, "AF2SFS1 report")
    root_payload = _read(root_cause_report, "Root-cause report")
    _validate_arm_report(control_payload, "AF2CTRL")
    _validate_arm_report(candidate_payload, "AF2SFS1")
    if (
        root_payload.get("decision") != "INTERPRETABLE"
        or root_payload.get("training_executed") is not False
        or root_payload.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Root-cause prerequisite tidak valid")
    layout = discover_layout(data_root)
    if "val" not in layout.splits or len(layout.names) != 21:
        raise RuntimeError("Kontrak validation Faruq-v3 tidak lengkap")

    state_metrics = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for state in STATES:
        print(f"FULL mAP INTERVENTION: {state}", flush=True)
        checkpoint = control_checkpoint if state == "AF2CTRL" else candidate_checkpoint
        state_metrics[state] = _evaluate_state(
            checkpoint, layout.yaml_path, layout.names, state, output_root, device
        )

    control_completed = control_payload["metrics"]
    candidate_completed = candidate_payload["metrics"]
    calibration = {
        "AF2CTRL": {
            key: float(state_metrics["AF2CTRL"][key] - control_completed[key])
            for key in HEADLINE
        },
        "AF2SFS1": {
            key: float(state_metrics["NORMAL"][key] - candidate_completed[key])
            for key in HEADLINE
        },
    }
    metric_keys = (*HEADLINE, "macro_ap50", "macro_ap75")
    state_values = {
        state: {key: float(metrics[key]) for key in metric_keys}
        for state, metrics in state_metrics.items()
    }
    control_values = state_values["AF2CTRL"]
    decomposition = decompose_metrics(
        control_values, state_values["NORMAL"], state_values["BYPASS"]
    )
    spatial_vs_normal = _subtract(state_values["SPATIAL_ONLY"], state_values["NORMAL"])
    frequency_vs_normal = _subtract(state_values["FREQUENCY_ONLY"], state_values["NORMAL"])

    control_model = YOLO(str(control_checkpoint)).model.eval()
    candidate_model = YOLO(str(candidate_checkpoint)).model.eval()
    drift = checkpoint_drift(control_model, candidate_model)
    class_rows = _class_decomposition(layout.names, state_metrics["AF2CTRL"], state_metrics)
    maximum_additivity_error = max(
        abs(value) for value in decomposition["additivity_error"].values()
    )
    direct_macro = decomposition["direct_selector_normal_minus_bypass"]["macro_map50_95"]
    mediated_macro = decomposition["optimization_mediated_bypass_minus_control"]["macro_map50_95"]
    if direct_macro > 0 and direct_macro >= mediated_macro:
        attribution = "ACTIVE_SELECTOR_DOMINANT"
    elif mediated_macro > 0 and mediated_macro > direct_macro:
        attribution = "OPTIMIZATION_MEDIATED_DOMINANT"
    elif direct_macro <= 0 < mediated_macro:
        attribution = "OPTIMIZATION_MEDIATED_SELECTOR_NOT_BENEFICIAL"
    else:
        attribution = "NO_POSITIVE_COMPONENT"

    gates = {
        "root_cause_prerequisite_interpretable": True,
        "control_reproduces_completed_headline": all(abs(value) <= 1e-6 for value in calibration["AF2CTRL"].values()),
        "normal_reproduces_completed_headline": all(abs(value) <= 1e-6 for value in calibration["AF2SFS1"].values()),
        "all_states_have_21_classes": all(
            len(metrics["map50_95_by_class"]) == 21 and not metrics["classes_without_ground_truth"]
            for metrics in state_metrics.values()
        ),
        "decomposition_is_additive": maximum_additivity_error <= 1e-10,
        "training_not_executed": True,
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.af2sfs1.map_intervention.v1",
        "protocol": "faruq-v3-af2sfs1-map-intervention-2026-08-28",
        "states": state_values,
        "normal_calibration_vs_completed_report": calibration,
        "decomposition": decomposition,
        "forced_path_minus_normal": {
            "SPATIAL_ONLY": spatial_vs_normal,
            "FREQUENCY_ONLY": frequency_vs_normal,
        },
        "per_class_decomposition": class_rows,
        "top5_total_improvements": class_rows[:5],
        "top5_total_regressions": list(reversed(class_rows[-5:])),
        "checkpoint_drift": drift,
        "attribution": {
            "result": attribution,
            "direct_selector_macro_effect": direct_macro,
            "optimization_mediated_macro_effect": mediated_macro,
            "boundary": "validation-only post-hoc mechanism attribution; no model selection or test claim",
        },
        "gates": gates,
        "decision": "INTERPRETABLE" if all(gates.values()) else "INVALID",
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    destination = output_root / "af2sfs1_map_intervention.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2SFS1 full-mAP inference intervention")
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--root-cause-report", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_af2sfs1_map_intervention(
        args.control_checkpoint,
        args.candidate_checkpoint,
        args.control_report,
        args.candidate_report,
        args.root_cause_report,
        args.data_root,
        args.output_root,
        device=args.device,
    )
    print("STATES:", json.dumps(result["states"], indent=2))
    print("DECOMPOSITION:", json.dumps(result["decomposition"], indent=2))
    print("ATTRIBUTION:", result["attribution"])
    print("GATES:", result["gates"])
    print("DECISION:", result["decision"])
    print("SUMMARY:", result["summary"])


if __name__ == "__main__":
    main()
