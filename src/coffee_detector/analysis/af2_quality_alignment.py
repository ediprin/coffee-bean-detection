"""Validation-only confidence/IoU alignment audit for D0FT and AF2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torchvision.ops import box_iou

from coffee_detector.analysis.af2_box_score_factorial import (
    METRICS,
    _device,
    _match_predictions,
    _validation_loader,
    summarize_detection_stats,
)
from coffee_detector.analysis.coffee_fg_diagnostics import _unwrap_head
from coffee_detector.dataset import discover_layout
from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)


PROTOCOL = "faruq-v3-af2-quality-alignment-validation-audit-v1"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return values
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    if len(left_rank) < 2 or np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def continuous_ece(
    confidences: np.ndarray, qualities: np.ndarray, bins: int = 15
) -> float:
    """Expected absolute gap between confidence and same-class IoU quality."""

    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    qualities = np.asarray(qualities, dtype=np.float64).reshape(-1)
    if len(confidences) != len(qualities):
        raise ValueError("Confidence dan quality harus sama panjang")
    if not len(confidences):
        return 0.0
    indices = np.minimum((confidences.clip(0, 1) * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(confidences[mask].mean()) - float(qualities[mask].mean())
            )
    return float(error)


def same_class_iou_quality(
    predicted_boxes: torch.Tensor,
    predicted_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
) -> torch.Tensor:
    """Maximum IoU to a GT of the predicted class; zero for unmatched classes."""

    if not len(predicted_boxes) or not len(target_boxes):
        return predicted_boxes.new_zeros(len(predicted_boxes))
    iou = box_iou(predicted_boxes, target_boxes)
    same_class = predicted_classes[:, None] == target_classes[None, :]
    return (iou * same_class).amax(dim=1)


def _new_stats() -> dict[str, list[np.ndarray]]:
    return {
        key: []
        for key in (
            "tp",
            "confidence",
            "quality",
            "predicted_class",
            "target_class",
        )
    }


def _alignment_summary(
    confidence: np.ndarray,
    quality: np.ndarray,
    predicted_class: np.ndarray,
    names: Mapping[int, str],
) -> dict[str, Any]:
    rows = {}
    for class_id, name in names.items():
        mask = predicted_class == class_id
        if not mask.any():
            continue
        rows[name] = {
            "predictions": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()),
            "mean_same_class_iou_quality": float(quality[mask].mean()),
            "spearman_confidence_quality": spearman_correlation(
                confidence[mask], quality[mask]
            ),
            "continuous_ece": continuous_ece(confidence[mask], quality[mask]),
            "quality_brier": float(np.square(confidence[mask] - quality[mask]).mean()),
        }
    return {
        "predictions": int(len(confidence)),
        "mean_confidence": float(confidence.mean()) if len(confidence) else 0.0,
        "mean_same_class_iou_quality": float(quality.mean()) if len(quality) else 0.0,
        "spearman_confidence_quality": spearman_correlation(confidence, quality),
        "continuous_ece": continuous_ece(confidence, quality),
        "quality_brier": float(np.square(confidence - quality).mean()) if len(quality) else 0.0,
        "per_class": rows,
    }


def _validate_factorial(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Factorial summary tidak ditemukan: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("format") != "coffee_detector.af2_box_score_factorial.result.v1"
        or payload.get("decision") != "AF2_BOX_SCORE_INTERACTION_NECESSARY"
        or payload.get("training_executed") is not False
        or payload.get("test_images_accessed") is not False
        or not all(payload.get("gates", {}).values())
    ):
        raise RuntimeError("Factorial AF2 bukan hasil valid yang mengotorisasi audit")
    return payload


def _decision(results: Mapping[str, Mapping[str, Any]]) -> tuple[dict, str, str]:
    af2 = results["AF2"]
    d0ft = results["D0FT"]
    af2_gain = af2["oracle_minus_native"]
    d0ft_gain = d0ft["oracle_minus_native"]
    criteria = {
        "af2_fixed_candidate_oracle_macro_gain_at_least_0_5_point": af2_gain[
            "macro_map50_95"
        ]
        >= 0.005,
        "af2_fixed_candidate_oracle_bottom3_gain_at_least_0_5_point": af2_gain[
            "bottom3_class_map50_95"
        ]
        >= 0.005,
        "af2_fixed_candidate_oracle_worst_not_lower": af2_gain[
            "worst_class_map50_95"
        ]
        >= 0.0,
        "af2_has_nonperfect_confidence_quality_ranking": af2["alignment"][
            "spearman_confidence_quality"
        ]
        < 0.95,
    }
    comparison = {
        "AF2_oracle_minus_native": af2_gain,
        "D0FT_oracle_minus_native": d0ft_gain,
        "AF2_minus_D0FT_oracle_headroom": {
            metric: float(af2_gain[metric]) - float(d0ft_gain[metric])
            for metric in METRICS
        },
        "AF2_minus_D0FT_alignment": {
            metric: float(af2["alignment"][metric])
            - float(d0ft["alignment"][metric])
            for metric in (
                "spearman_confidence_quality",
                "continuous_ece",
                "quality_brier",
            )
        },
        "criteria": criteria,
    }
    if all(criteria.values()):
        return (
            comparison,
            "QUALITY_ALIGNMENT_HEADROOM_SUPPORTED",
            "AUTHORIZE_MATCHED_AF2_QUALITY_LOSS_SCREEN",
        )
    return comparison, "QUALITY_ALIGNMENT_HEADROOM_INSUFFICIENT", "RETAIN_AF2_WITHOUT_QUALITY_LOSS"


def run_af2_quality_alignment_audit(
    d0ft_checkpoint: str | Path,
    af2_checkpoint: str | Path,
    data_root: str | Path,
    grouped_summary: str | Path,
    factorial_summary: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 640,
    batch_size: int = 16,
    workers: int = 2,
    max_det: int = 500,
    confidence: float = 0.001,
) -> dict:
    from ultralytics import YOLO
    from ultralytics.utils import ops

    data_root = Path(data_root).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    factorial = _validate_factorial(factorial_summary)
    checkpoints = {
        "D0FT": Path(d0ft_checkpoint).expanduser().resolve(),
        "AF2": Path(af2_checkpoint).expanduser().resolve(),
    }
    if any(not path.is_file() for path in checkpoints.values()):
        raise FileNotFoundError(f"Checkpoint tidak lengkap: {checkpoints}")
    layout = discover_layout(data_root)
    torch_device = _device(device)
    yolo_models = {name: YOLO(str(path)) for name, path in checkpoints.items()}
    networks = {
        name: wrapper.model.to(torch_device).eval()
        for name, wrapper in yolo_models.items()
    }
    names = {
        name: {int(key): str(value) for key, value in network.names.items()}
        for name, network in networks.items()
    }
    if names["D0FT"] != names["AF2"] or names["AF2"] != layout.names:
        raise RuntimeError("Ontologi checkpoint dan validation tidak identik")
    dataset, loader = _validation_loader(
        yolo_models["D0FT"],
        layout.yaml_path,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
    )
    thresholds = torch.linspace(0.5, 0.95, 10, device=torch_device)
    stats = {name: _new_stats() for name in networks}
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(torch_device, non_blocking=True).float().div_(255.0)
            batch_indices = batch["batch_idx"].to(torch_device)
            batch_boxes = batch["bboxes"].to(torch_device)
            batch_classes = batch["cls"].to(torch_device).long().reshape(-1)
            predictions = {}
            for model_name, network in networks.items():
                head = _unwrap_head(network)
                head.max_det = int(max_det)
                network_output = network(images)
                predictions[model_name] = (
                    network_output[0]
                    if isinstance(network_output, tuple)
                    else network_output
                )
            height, width = images.shape[-2:]
            scale = torch.tensor([width, height, width, height], device=torch_device)
            for sample_index in range(len(images)):
                target_mask = batch_indices == sample_index
                target_classes = batch_classes[target_mask]
                target_boxes = batch_boxes[target_mask]
                if len(target_boxes):
                    target_boxes = ops.xywh2xyxy(target_boxes) * scale
                for model_name in networks:
                    prediction = predictions[model_name][sample_index]
                    prediction = prediction[prediction[:, 4] > confidence]
                    predicted_boxes = prediction[:, :4]
                    predicted_classes = prediction[:, 5].long()
                    quality = same_class_iou_quality(
                        predicted_boxes,
                        predicted_classes,
                        target_boxes,
                        target_classes,
                    )
                    correct = _match_predictions(
                        predicted_classes,
                        target_classes,
                        predicted_boxes,
                        target_boxes,
                        thresholds,
                    )
                    stats[model_name]["tp"].append(correct.numpy())
                    stats[model_name]["confidence"].append(
                        prediction[:, 4].float().cpu().numpy()
                    )
                    stats[model_name]["quality"].append(quality.float().cpu().numpy())
                    stats[model_name]["predicted_class"].append(
                        predicted_classes.cpu().numpy()
                    )
                    stats[model_name]["target_class"].append(target_classes.cpu().numpy())
            completed += len(images)
            if completed % 50 < len(images) or completed == len(dataset):
                print(f"QUALITY AUDIT {completed}/{len(dataset)}", flush=True)

    results = {}
    for model_name, values in stats.items():
        merged = {key: np.concatenate(rows, axis=0) for key, rows in values.items()}
        native = summarize_detection_stats(
            merged["tp"],
            merged["confidence"],
            merged["predicted_class"],
            merged["target_class"],
            layout.names,
        )
        # Fixed candidate set and boxes/classes: only confidence ordering becomes
        # the unattainable GT-IoU target, measuring ranking headroom rather than
        # proposal or localization headroom.
        oracle = summarize_detection_stats(
            merged["tp"],
            merged["quality"],
            merged["predicted_class"],
            merged["target_class"],
            layout.names,
        )
        results[model_name] = {
            "native": native,
            "fixed_candidate_iou_oracle": oracle,
            "oracle_minus_native": {
                metric: float(oracle[metric]) - float(native[metric]) for metric in METRICS
            },
            "alignment": _alignment_summary(
                merged["confidence"],
                merged["quality"],
                merged["predicted_class"],
                layout.names,
            ),
        }
    endpoint_deltas = {
        "D0FT_vs_factorial_DD": {
            metric: float(results["D0FT"]["native"][metric])
            - float(factorial["results"]["DD"][metric])
            for metric in METRICS
        },
        "AF2_vs_factorial_AA": {
            metric: float(results["AF2"]["native"][metric])
            - float(factorial["results"]["AA"][metric])
            for metric in METRICS
        },
    }
    gates = {
        "same_21_class_ontology": len(layout.names) == 21,
        "fixed_candidate_set_for_oracle": True,
        "d0ft_endpoint_exact": all(
            abs(value) <= 1e-12 for value in endpoint_deltas["D0FT_vs_factorial_DD"].values()
        ),
        "af2_endpoint_exact": all(
            abs(value) <= 1e-12 for value in endpoint_deltas["AF2_vs_factorial_AA"].values()
        ),
        "all_21_classes_present": all(
            not row["native"]["classes_without_ground_truth"] for row in results.values()
        ),
        "training_not_executed": True,
        "test_not_accessed": True,
    }
    if all(gates.values()):
        comparison, decision, next_action = _decision(results)
    else:
        comparison, decision, next_action = (
            {"criteria": {}},
            "INVALID_ALIGNMENT_AUDIT",
            "STOP_WITHOUT_TRAINING",
        )
    payload = {
        "format": "coffee_detector.af2_quality_alignment.result.v1",
        "protocol": PROTOCOL,
        "seed": 42,
        "results": results,
        "endpoint_deltas": endpoint_deltas,
        "gates": gates,
        "comparison": comparison,
        "decision": decision,
        "next": next_action,
        "factorial_summary": str(Path(factorial_summary).expanduser().resolve()),
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "evaluation_split": "val",
        "images": len(dataset),
        "confidence_threshold": float(confidence),
        "max_det": int(max_det),
        "oracle_scope": "fixed_native_final_candidate_set_gt_iou_score_only",
        "training_executed": False,
        "test_images_accessed": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2 score/IoU alignment audit, validation only")
    parser.add_argument("--d0ft-checkpoint", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--factorial-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--confidence", type=float, default=0.001)
    args = parser.parse_args()
    result = run_af2_quality_alignment_audit(
        args.d0ft_checkpoint,
        args.af2_checkpoint,
        args.data_root,
        args.grouped_summary,
        args.factorial_summary,
        args.output,
        device=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        max_det=args.max_det,
        confidence=args.confidence,
    )
    table = {
        model: {
            "native": {metric: row["native"][metric] for metric in METRICS},
            "oracle": {
                metric: row["fixed_candidate_iou_oracle"][metric] for metric in METRICS
            },
            "gain": row["oracle_minus_native"],
            "alignment": {
                key: row["alignment"][key]
                for key in (
                    "spearman_confidence_quality",
                    "continuous_ece",
                    "quality_brier",
                )
            },
        }
        for model, row in result["results"].items()
    }
    print(json.dumps(table, indent=2, ensure_ascii=False))
    print("GATES:", result["gates"])
    print("COMPARISON:", json.dumps(result["comparison"], indent=2))
    print("DECISION:", result["decision"])
    print("NEXT:", result["next"])
    print("TRAINING: False | TEST: False")
    print("SUMMARY:", result["summary"])


if __name__ == "__main__":
    main()
