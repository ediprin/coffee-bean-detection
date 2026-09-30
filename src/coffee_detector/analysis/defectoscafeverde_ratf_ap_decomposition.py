"""Validation-only AP decomposition for RATF1's ``partido`` regression.

The audit evaluates three endpoints in one dataloader pass: the matched native
detector, RATF1 as trained, and the same RATF1 checkpoint with only its
inference-time texture class residual zeroed.  It never trains and never reads
the test split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

import numpy as np
import torch
from torchvision.ops import box_iou

from coffee_detector.analysis.af2_box_score_factorial import (
    _match_predictions,
    _network_raw,
    _validation_loader,
    summarize_detection_stats,
)
from coffee_detector.analysis.coffee_fg_diagnostics import _unwrap_head
from coffee_detector.analysis.defectoscafeverde_ratf_partido_audit import (
    MODEL_ORDER,
    TARGET_CLASS,
    TARGET_CLASS_ID,
    _load_json,
    _sha256,
    validate_contracts,
)
from coffee_detector.data.prepare_defectoscafeverde_grouped import NAMES
from coffee_detector.dataset import discover_layout
from coffee_detector.defectos_ratf import RATFDetectHead
from coffee_detector.experiments.run_defectoscafeverde_af2_direct import (
    validate_development_dataset,
)


FORMAT = "coffee_detector.defectoscafeverde.ratf1_partido_ap_decomposition.v1"
PROTOCOL = "defectoscafeverde-ratf1-partido-ap-decomposition-validation-v1"
HEADLINE_METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _device(value: str) -> torch.device:
    device = torch.device(f"cuda:{value}" if str(value).isdigit() else value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {device}")
    return device


def _empty_stats() -> dict[str, list[np.ndarray]]:
    return {
        key: [] for key in ("tp", "confidence", "predicted_class", "target_class")
    }


def _empty_target_diagnostic() -> dict[str, Any]:
    return {
        "targets": 0,
        "predictions": 0,
        "tp50": 0,
        "tp75": 0,
        "matched_iou50": [],
        "matched_iou75": [],
        "tp_confidence50": [],
        "fp_confidence50": [],
        "fp_categories50": Counter(),
    }


def _greedy_iou_pairs(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Return the same IoU-prioritized one-to-one pairs used by validation AP."""

    if not len(predicted_boxes) or not len(target_boxes):
        return []
    matrix = box_iou(target_boxes, predicted_boxes).detach().cpu().numpy()
    matches = np.array(np.nonzero(matrix >= float(threshold))).T
    if not len(matches):
        return []
    if len(matches) > 1:
        matches = matches[matrix[matches[:, 0], matches[:, 1]].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return [
        (int(prediction), int(target), float(matrix[target, prediction]))
        for target, prediction in matches
    ]


def _update_target_diagnostic(
    totals: dict[str, Any],
    prediction: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
) -> None:
    target_mask = target_classes == TARGET_CLASS_ID
    predicted_mask = prediction[:, 5].long() == TARGET_CLASS_ID
    targets = target_boxes[target_mask]
    predicted = prediction[predicted_mask]
    predicted_boxes = predicted[:, :4]
    confidences = predicted[:, 4]

    totals["targets"] += int(len(targets))
    totals["predictions"] += int(len(predicted))
    pairs50 = _greedy_iou_pairs(predicted_boxes, targets, 0.50)
    pairs75 = _greedy_iou_pairs(predicted_boxes, targets, 0.75)
    totals["tp50"] += len(pairs50)
    totals["tp75"] += len(pairs75)
    totals["matched_iou50"].extend(iou for _, _, iou in pairs50)
    totals["matched_iou75"].extend(iou for _, _, iou in pairs75)
    matched_predictions = {prediction_index for prediction_index, _, _ in pairs50}
    totals["tp_confidence50"].extend(
        float(confidences[index]) for index in sorted(matched_predictions)
    )

    other_boxes = target_boxes[~target_mask]
    target_overlap = (
        box_iou(predicted_boxes, targets).max(dim=1).values
        if len(predicted_boxes) and len(targets)
        else predicted_boxes.new_zeros(len(predicted_boxes))
    )
    other_overlap = (
        box_iou(predicted_boxes, other_boxes).max(dim=1).values
        if len(predicted_boxes) and len(other_boxes)
        else predicted_boxes.new_zeros(len(predicted_boxes))
    )
    for index in range(len(predicted)):
        if index in matched_predictions:
            continue
        confidence = float(confidences[index])
        totals["fp_confidence50"].append(confidence)
        if float(target_overlap[index]) >= 0.50:
            category = "duplicate_target"
        elif float(other_overlap[index]) >= 0.50:
            category = "overlaps_other_class"
        else:
            category = "background_or_localization"
        totals["fp_categories50"][category] += 1


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return median(values) if values else None


def summarize_target_diagnostic(totals: Mapping[str, Any]) -> dict[str, Any]:
    targets = int(totals["targets"])
    predictions = int(totals["predictions"])
    tp50 = int(totals["tp50"])
    tp75 = int(totals["tp75"])
    fp_confidences = list(map(float, totals["fp_confidence50"]))
    return {
        "targets": targets,
        "predictions": predictions,
        "tp50": tp50,
        "fp50": predictions - tp50,
        "fn50": targets - tp50,
        "tp75": tp75,
        "fp75": predictions - tp75,
        "fn75": targets - tp75,
        "recall50": tp50 / max(targets, 1),
        "recall75": tp75 / max(targets, 1),
        "mean_matched_iou50": _safe_mean(list(totals["matched_iou50"])),
        "median_matched_iou50": _safe_median(list(totals["matched_iou50"])),
        "mean_matched_iou75": _safe_mean(list(totals["matched_iou75"])),
        "mean_tp_confidence50": _safe_mean(list(totals["tp_confidence50"])),
        "mean_fp_confidence50": _safe_mean(fp_confidences),
        "median_fp_confidence50": _safe_median(fp_confidences),
        "fp_confidence_at_least_0_25": sum(value >= 0.25 for value in fp_confidences),
        "fp_confidence_at_least_0_50": sum(value >= 0.50 for value in fp_confidences),
        "fp_categories50": dict(totals["fp_categories50"]),
    }


def target_ap_summary(
    true_positives: np.ndarray,
    confidences: np.ndarray,
    predicted_classes: np.ndarray,
    target_classes: np.ndarray,
) -> dict[str, Any]:
    from ultralytics.utils.metrics import ap_per_class

    result = ap_per_class(
        true_positives,
        confidences,
        predicted_classes,
        target_classes,
        plot=False,
        names=dict(enumerate(NAMES)),
    )
    _, _, precision, recall, f1, ap, class_indices, _, _, f1_curve, x, _ = result
    class_indices = np.asarray(class_indices, dtype=int).reshape(-1)
    rows = np.flatnonzero(class_indices == TARGET_CLASS_ID)
    if len(rows) != 1:
        raise RuntimeError(f"Evaluator tidak menghasilkan tepat satu baris {TARGET_CLASS}")
    row = int(rows[0])
    ap_row = np.asarray(ap, dtype=np.float64)[row]
    if ap_row.shape != (10,):
        raise RuntimeError(f"Kurva AP {TARGET_CLASS} bukan 10 threshold: {ap_row.shape}")
    f1_row = np.asarray(f1_curve, dtype=np.float64)[row]
    confidence_axis = np.asarray(x, dtype=np.float64)
    best_index = int(f1_row.argmax())
    thresholds = [f"{value:.2f}" for value in np.linspace(0.50, 0.95, 10)]
    return {
        "precision": float(np.asarray(precision)[row]),
        "recall": float(np.asarray(recall)[row]),
        "f1": float(np.asarray(f1)[row]),
        "ap50": float(ap_row[0]),
        "ap75": float(ap_row[5]),
        "ap50_95": float(ap_row.mean()),
        "ap_by_iou": {key: float(value) for key, value in zip(thresholds, ap_row)},
        "best_f1": float(f1_row[best_index]),
        "best_f1_confidence": float(confidence_axis[best_index]),
    }


def attribute_ap_regression(endpoints: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    d0 = endpoints["D0DIRECT"]["target_ap"]
    active = endpoints["RATF1_ACTIVE"]["target_ap"]
    zero = endpoints["RATF1_ZERO_RESIDUAL"]["target_ap"]
    d0_ap = float(d0["ap50_95"])
    active_ap = float(active["ap50_95"])
    zero_ap = float(zero["ap50_95"])
    high_keys = ("0.75", "0.80", "0.85", "0.90", "0.95")
    high_d0 = mean(float(d0["ap_by_iou"][key]) for key in high_keys)
    high_active = mean(float(active["ap_by_iou"][key]) for key in high_keys)
    deltas = {
        "active_minus_d0_ap50": float(active["ap50"]) - float(d0["ap50"]),
        "active_minus_d0_ap75": float(active["ap75"]) - float(d0["ap75"]),
        "active_minus_d0_ap50_95": active_ap - d0_ap,
        "active_minus_d0_high_iou_mean": high_active - high_d0,
        "zero_minus_active_ap50_95": zero_ap - active_ap,
        "zero_minus_d0_ap50_95": zero_ap - d0_ap,
    }
    tolerance = 0.002
    components = {
        "active_target_ap_regresses": active_ap < d0_ap - tolerance,
        "ap50_is_preserved": float(active["ap50"]) >= float(d0["ap50"]) - tolerance,
        "high_iou_ap_regresses": high_active < high_d0 - tolerance,
        "zero_residual_recovers_material_ap": zero_ap > active_ap + tolerance,
        "zero_residual_recovers_d0": zero_ap >= d0_ap - tolerance,
    }
    if not components["active_target_ap_regresses"]:
        attribution = "NO_TARGET_AP_REGRESSION_REPRODUCED"
    elif components["zero_residual_recovers_material_ap"]:
        attribution = (
            "INFERENCE_RESIDUAL_DOMINANT"
            if components["zero_residual_recovers_d0"]
            else "MIXED_INFERENCE_RESIDUAL_AND_TRAINING_PATH"
        )
    elif components["ap50_is_preserved"] and components["high_iou_ap_regresses"]:
        attribution = "HIGH_IOU_LOCALIZATION_OR_RANKING_DOMINANT"
    else:
        attribution = "TRAINING_PATH_SCORE_RANKING_OR_DUPLICATE_FP_DOMINANT"
    return {
        "deltas": deltas,
        "components": components,
        "attribution": attribution,
        "interpretation_boundary": (
            "Diagnostic attribution from validation AP, zero-residual intervention, "
            "and fixed endpoint predictions; not a new training claim."
        ),
    }


def _arrays(stats: Mapping[str, list[np.ndarray]]) -> tuple[np.ndarray, ...]:
    return tuple(
        np.concatenate(stats[key], axis=0)
        for key in ("tp", "confidence", "predicted_class", "target_class")
    )


def run_ratf_partido_ap_decomposition(
    data_root: str | Path,
    grouped_audit: str | Path,
    d0_result: str | Path,
    ratf_result: str | Path,
    d0_checkpoint: str | Path,
    ratf_checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 640,
    batch_size: int = 16,
    workers: int = 2,
    max_det: int = 500,
    confidence: float = 0.001,
    endpoint_tolerance: float = 0.002,
) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.utils import ops

    data_root = Path(data_root).expanduser().resolve()
    grouped_audit = Path(grouped_audit).expanduser().resolve()
    dataset_contract = validate_development_dataset(data_root, grouped_audit)
    contract = validate_contracts(
        d0_result, ratf_result, d0_checkpoint, ratf_checkpoint
    )
    d0_payload = _load_json(d0_result, "D0 result")
    ratf_payload = _load_json(ratf_result, "RATF1 result")
    expected_audit_sha = d0_payload["run_contract"]["dataset_audit_sha256"]
    if _sha256(grouped_audit) != expected_audit_sha:
        raise RuntimeError("Dataset audit yang diberikan tidak cocok dengan endpoint")
    if (data_root / "test").exists():
        raise RuntimeError("Test tidak boleh tersedia pada audit ini")

    torch_device = _device(device)
    yolo_models = {
        "D0DIRECT": YOLO(str(Path(d0_checkpoint).expanduser().resolve())),
        "RATF1_ACTIVE": YOLO(str(Path(ratf_checkpoint).expanduser().resolve())),
        "RATF1_ZERO_RESIDUAL": YOLO(str(Path(ratf_checkpoint).expanduser().resolve())),
    }
    networks = {
        name: model.model.to(torch_device).eval() for name, model in yolo_models.items()
    }
    for name in ("RATF1_ACTIVE", "RATF1_ZERO_RESIDUAL"):
        if not isinstance(networks[name].model[-1], RATFDetectHead):
            raise TypeError(f"{name} tidak memiliki RATFDetectHead")
    with torch.no_grad():
        for adapter in networks["RATF1_ZERO_RESIDUAL"].model[-1].adapters:
            adapter.class_residual.weight.zero_()
            adapter.class_residual.bias.zero_()

    dataset, loader = _validation_loader(
        yolo_models["D0DIRECT"],
        data_root / "data.yaml",
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
    )
    names = getattr(networks["D0DIRECT"], "names", None)
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(key): str(value) for key, value in dict(names).items()}
    layout = discover_layout(data_root)
    endpoint_names = {}
    for name, network in networks.items():
        values = getattr(network, "names", None)
        if isinstance(values, list):
            values = dict(enumerate(values))
        endpoint_names[name] = {int(key): str(value) for key, value in dict(values).items()}
    gates = {
        "dataset_contract_pass": all(dataset_contract.get("gates", {}).values()),
        "paired_endpoint_contract_pass": all(contract["paired"].values()),
        "same_12_class_ontology": (
            layout.names == names == dict(enumerate(NAMES))
            and all(values == names for values in endpoint_names.values())
        ),
        "target_class_is_partido_id9": TARGET_CLASS == "partido" and TARGET_CLASS_ID == 9,
        "all_heads_have_12_classes": all(
            int(_unwrap_head(network).nc) == len(NAMES) for network in networks.values()
        ),
        "zero_residual_exact": all(
            int(torch.count_nonzero(parameter)) == 0
            for adapter in networks["RATF1_ZERO_RESIDUAL"].model[-1].adapters
            for parameter in (
                adapter.class_residual.weight,
                adapter.class_residual.bias,
            )
        ),
        "validation_only": True,
        "training_not_executed": True,
        "test_not_accessed": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Static audit gate gagal: {gates}")

    thresholds = torch.linspace(0.50, 0.95, 10, device=torch_device)
    stats = {name: _empty_stats() for name in MODEL_ORDER}
    diagnostics = {name: _empty_target_diagnostic() for name in MODEL_ORDER}
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(torch_device, non_blocking=True).float().div_(255.0)
            batch_indices = batch["batch_idx"].to(torch_device)
            batch_boxes = batch["bboxes"].to(torch_device)
            batch_classes = batch["cls"].to(torch_device).long().reshape(-1)
            predictions = {
                name: _network_raw(network, images, max_det)[0]
                for name, network in networks.items()
            }
            height, width = images.shape[-2:]
            scale = torch.tensor([width, height, width, height], device=torch_device)
            for sample_index in range(len(images)):
                target_mask = batch_indices == sample_index
                target_classes = batch_classes[target_mask]
                target_boxes = batch_boxes[target_mask]
                if len(target_boxes):
                    target_boxes = ops.xywh2xyxy(target_boxes) * scale
                for name in MODEL_ORDER:
                    prediction = predictions[name][sample_index]
                    prediction = prediction[prediction[:, 4] > confidence]
                    predicted_classes = prediction[:, 5].long()
                    correct = _match_predictions(
                        predicted_classes,
                        target_classes,
                        prediction[:, :4],
                        target_boxes,
                        thresholds,
                    )
                    stats[name]["tp"].append(correct.numpy())
                    stats[name]["confidence"].append(
                        prediction[:, 4].float().cpu().numpy()
                    )
                    stats[name]["predicted_class"].append(
                        predicted_classes.cpu().numpy()
                    )
                    stats[name]["target_class"].append(target_classes.cpu().numpy())
                    _update_target_diagnostic(
                        diagnostics[name], prediction, target_boxes, target_classes
                    )
            completed += len(images)
            if completed % 100 < len(images) or completed == len(dataset):
                print(f"AP DECOMPOSITION {completed}/{len(dataset)}", flush=True)

    endpoints = {}
    for name in MODEL_ORDER:
        arrays = _arrays(stats[name])
        endpoints[name] = {
            "global_metrics": summarize_detection_stats(*arrays, names),
            "target_ap": target_ap_summary(*arrays),
            "target_prediction_diagnostic": summarize_target_diagnostic(
                diagnostics[name]
            ),
        }

    historical = {
        "D0DIRECT": d0_payload["metrics"],
        "RATF1_ACTIVE": ratf_payload["metrics"],
    }
    calibration = {
        name: {
            metric: float(endpoints[name]["global_metrics"][metric])
            - float(historical[name][metric])
            for metric in HEADLINE_METRICS
        }
        for name in historical
    }
    gates.update(
        {
            "d0_endpoint_calibrated": all(
                abs(value) <= endpoint_tolerance
                for value in calibration["D0DIRECT"].values()
            ),
            "ratf_active_endpoint_calibrated": all(
                abs(value) <= endpoint_tolerance
                for value in calibration["RATF1_ACTIVE"].values()
            ),
            "all_12_validation_classes_present": all(
                not row["global_metrics"]["classes_without_ground_truth"]
                for row in endpoints.values()
            ),
        }
    )
    if not all(gates.values()):
        raise RuntimeError(f"Evaluator calibration gate gagal: {gates}")
    decomposition = attribute_ap_regression(endpoints)
    payload = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "seed": 42,
        "target_class": TARGET_CLASS,
        "target_class_id": TARGET_CLASS_ID,
        "endpoints": endpoints,
        "historical_calibration": calibration,
        "decomposition": decomposition,
        "gates": gates,
        "contract": contract,
        "settings": {
            "image_size": image_size,
            "batch_size": batch_size,
            "workers": workers,
            "max_det": max_det,
            "confidence": confidence,
            "endpoint_tolerance": endpoint_tolerance,
        },
        "images": len(dataset),
        "evaluation_split": "val",
        "training_executed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="RATF1 partido AP decomposition")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-audit", required=True)
    parser.add_argument("--d0-result", required=True)
    parser.add_argument("--ratf-result", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--ratf-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = run_ratf_partido_ap_decomposition(
        args.data_root,
        args.grouped_audit,
        args.d0_result,
        args.ratf_result,
        args.d0_checkpoint,
        args.ratf_checkpoint,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print("TARGET AP:", json.dumps({
        name: row["target_ap"] for name, row in result["endpoints"].items()
    }, indent=2))
    print("TARGET DIAGNOSTIC:", json.dumps({
        name: row["target_prediction_diagnostic"]
        for name, row in result["endpoints"].items()
    }, indent=2))
    print("CALIBRATION:", result["historical_calibration"])
    print("DECOMPOSITION:", result["decomposition"])
    print("GATES:", result["gates"])
    print("TRAINING: False | TEST: False")
    print("SUMMARY:", result["summary"])


if __name__ == "__main__":
    main()
