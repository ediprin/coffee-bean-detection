"""Validation-only residual error attribution for selected ACMC1 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from coffee_detector.analysis.coffee_fg_diagnostics import diagnose_checkpoint
from coffee_detector.dataset import discover_layout

IOU_THRESHOLDS = tuple(round(float(x), 2) for x in np.linspace(0.50, 0.95, 10))
DEFAULT_SEEDS = (42, 123, 2026)

LOW_AP50_THRESHOLD = 0.90
HIGH_IOU_DROP_THRESHOLD = 0.10
CLASS_HEADROOM_THRESHOLD = 0.10
PROPOSAL_GAP_THRESHOLD = 0.10


def _profile_from_ap_matrix(ap_matrix, class_indices, names):
    matrix = np.asarray(ap_matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] != len(IOU_THRESHOLDS):
        raise ValueError(
            f"AP matrix harus punya {len(IOU_THRESHOLDS)} threshold IoU, dapat {matrix.shape}"
        )
    indices = [int(x) for x in np.asarray(class_indices).reshape(-1)]
    if len(indices) != matrix.shape[0]:
        raise ValueError("Jumlah ap_class_index tidak cocok dengan baris AP matrix")

    out = {}
    for row, class_id in enumerate(indices):
        if class_id not in names:
            continue
        values = matrix[row]
        out[names[class_id]] = {
            "ap50": float(values[0]),
            "ap75": float(values[5]),
            "ap95": float(values[9]),
            "map50_95": float(values.mean()),
            "ap50_to_ap75_drop": float(values[0] - values[5]),
            "ap75_to_ap95_drop": float(values[5] - values[9]),
        }
    return out


def _evidence_flags(row):
    return {
        "low_ap50": row["ap50"] < LOW_AP50_THRESHOLD,
        "high_iou_localization_drop": row["ap50_to_ap75_drop"] >= HIGH_IOU_DROP_THRESHOLD,
        "classification_headroom_material": (
            1.0 - row["class_accuracy_given_iou50_match"]
        ) >= CLASS_HEADROOM_THRESHOLD,
        "proposal_gap_material": (
            1.0 - row["proposal_accessibility_iou50"]
        ) >= PROPOSAL_GAP_THRESHOLD,
    }


def _attribution_label(row):
    flags = _evidence_flags(row)
    cls = flags["classification_headroom_material"]
    loc = flags["high_iou_localization_drop"]
    proposal = flags["proposal_gap_material"]

    if proposal and flags["low_ap50"] and not cls:
        return "proposal_or_detection_limited"
    if cls and loc:
        return "mixed_classification_localization"
    if cls:
        return "classification_or_ranking_limited"
    if loc:
        return "high_iou_localization_limited"
    if proposal:
        return "proposal_or_detection_limited"
    return "no_single_dominant_signal"


def _validate_only_layout(data_root):
    layout = discover_layout(data_root)
    if "val" not in layout.splits:
        raise RuntimeError("Validation split tidak tersedia")
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit ini menolak dataset yang mengekspos test split")
    return layout


def _top_confusions(confusion, limit=20):
    rows = []
    for expected, predicted_counts in confusion.items():
        for predicted, count in predicted_counts.items():
            if expected == predicted:
                continue
            rows.append(
                {"expected": expected, "predicted": predicted, "count": int(count)}
            )
    return sorted(
        rows, key=lambda x: (-x["count"], x["expected"], x["predicted"])
    )[:limit]


def _run_seed(checkpoint, data_root, *, seed, device):
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint seed {seed} tidak ditemukan: {checkpoint}")
    layout = _validate_only_layout(data_root)

    metrics = YOLO(str(checkpoint)).val(
        data=str(layout.yaml_path),
        split="val",
        plots=False,
        verbose=False,
        device=device,
    )
    metric_object = getattr(metrics, "box", None)
    if metric_object is None or getattr(metric_object, "ap", None) is None:
        raise RuntimeError("Ultralytics tidak mengembalikan box AP matrix")

    classwise = _profile_from_ap_matrix(
        metric_object.ap,
        metric_object.ap_class_index,
        layout.names,
    )
    if set(classwise) != set(layout.names.values()):
        missing = sorted(set(layout.names.values()) - set(classwise))
        raise RuntimeError(f"Validation kehilangan kelas: {missing}")

    diagnostic = diagnose_checkpoint(
        checkpoint,
        data_root,
        split="val",
        image_size=640,
        candidate_counts=(500,),
        iou_threshold=0.50,
        confidence_threshold=0.25,
        nms_iou=0.70,
        max_det=500,
        device=device,
    )
    final = diagnostic["final_detections"]

    rows = {}
    for class_name, profile in classwise.items():
        diag = final["per_class"][class_name]
        row = {
            **profile,
            "targets": int(diag["targets"]),
            "proposal_accessibility_iou50": float(diag["proposal_accessibility"]),
            "matched_recall_iou50": float(diag["matched_recall"]),
            "class_accuracy_given_iou50_match": float(
                diag["localization_conditioned_class_accuracy"]
            ),
        }
        row["classification_headroom_iou50"] = (
            1.0 - row["class_accuracy_given_iou50_match"]
        )
        row["flags"] = _evidence_flags(row)
        row["attribution"] = _attribution_label(row)
        rows[class_name] = row

    return {
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "global": {
            "precision": float(metrics.results_dict["metrics/precision(B)"]),
            "recall": float(metrics.results_dict["metrics/recall(B)"]),
            "map50": float(metrics.results_dict["metrics/mAP50(B)"]),
            "map50_95": float(metrics.results_dict["metrics/mAP50-95(B)"]),
            "proposal_accessibility_iou50": float(final["proposal_accessibility"]),
            "matched_recall_iou50": float(final["matched_recall"]),
            "class_accuracy_given_iou50_match": float(
                final["localization_conditioned_class_accuracy"]
            ),
            "classification_headroom_iou50": float(
                final["oracle_class_accuracy_headroom"]
            ),
        },
        "per_class": rows,
        "top_directional_confusions": _top_confusions(final["confusion"]),
    }


def _aggregate(per_seed):
    class_names = sorted(
        set.intersection(
            *(set(payload["per_class"]) for payload in per_seed.values())
        )
    )
    rows = []
    metric_keys = (
        "ap50",
        "ap75",
        "ap95",
        "map50_95",
        "ap50_to_ap75_drop",
        "ap75_to_ap95_drop",
        "proposal_accessibility_iou50",
        "matched_recall_iou50",
        "class_accuracy_given_iou50_match",
        "classification_headroom_iou50",
    )
    for class_name in class_names:
        row = {"class_name": class_name}
        for metric in metric_keys:
            values = [
                float(payload["per_class"][class_name][metric])
                for payload in per_seed.values()
            ]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values)
        evidence_row = {
            "ap50": row["ap50_mean"],
            "ap50_to_ap75_drop": row["ap50_to_ap75_drop_mean"],
            "class_accuracy_given_iou50_match": row[
                "class_accuracy_given_iou50_match_mean"
            ],
            "proposal_accessibility_iou50": row[
                "proposal_accessibility_iou50_mean"
            ],
        }
        row["flags"] = _evidence_flags(evidence_row)
        row["attribution"] = _attribution_label(evidence_row)
        row["attribution_seed_agreement"] = sum(
            payload["per_class"][class_name]["attribution"] == row["attribution"]
            for payload in per_seed.values()
        )
        rows.append(row)
    return sorted(
        rows,
        key=lambda r: (
            float(r["map50_95_mean"]),
            float(r["class_accuracy_given_iou50_match_mean"]),
            r["class_name"],
        ),
    )


def _write_csv(rows, path):
    scalar_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k != "flags"}
        for key, value in row["flags"].items():
            flat[f"flag_{key}"] = value
        scalar_rows.append(flat)
    if not scalar_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def run_acmc1_residual_error_audit(
    checkpoints, data_root, output_root, *, device="0"
):
    _validate_only_layout(data_root)
    seeds = tuple(sorted(int(seed) for seed in checkpoints))
    if seeds != DEFAULT_SEEDS:
        raise RuntimeError(f"Audit dikunci ke seeds {DEFAULT_SEEDS}, dapat {seeds}")

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    per_seed = {
        str(seed): _run_seed(
            checkpoints[seed], data_root, seed=seed, device=device
        )
        for seed in DEFAULT_SEEDS
    }
    aggregate = _aggregate(per_seed)
    global_keys = (
        "precision",
        "recall",
        "map50",
        "map50_95",
        "proposal_accessibility_iou50",
        "matched_recall_iou50",
        "class_accuracy_given_iou50_match",
        "classification_headroom_iou50",
    )
    global_aggregate = {}
    for key in global_keys:
        values = [float(per_seed[str(seed)]["global"][key]) for seed in DEFAULT_SEEDS]
        global_aggregate[key] = {
            "mean": mean(values),
            "std": pstdev(values),
        }

    category_counts = {}
    for row in aggregate:
        category_counts[row["attribution"]] = category_counts.get(row["attribution"], 0) + 1

    payload = {
        "protocol": "faruq-v3-acmc1-residual-error-attribution-v1",
        "selected_model": "ACMC1",
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "seeds": list(DEFAULT_SEEDS),
        "iou_thresholds": list(IOU_THRESHOLDS),
        "frozen_thresholds": {
            "low_ap50": LOW_AP50_THRESHOLD,
            "high_iou_localization_drop": HIGH_IOU_DROP_THRESHOLD,
            "classification_headroom_material": CLASS_HEADROOM_THRESHOLD,
            "proposal_gap_material": PROPOSAL_GAP_THRESHOLD,
        },
        "global_aggregate": global_aggregate,
        "attribution_counts": category_counts,
        "per_class_aggregate": aggregate,
        "per_seed": per_seed,
    }
    summary = output_root / "acmc1_residual_error_attribution.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(aggregate, output_root / "acmc1_residual_error_attribution.csv")
    payload["summary"] = str(summary)
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="3-seed ACMC1 validation-only residual error attribution."
    )
    parser.add_argument("--seed42-checkpoint", required=True)
    parser.add_argument("--seed123-checkpoint", required=True)
    parser.add_argument("--seed2026-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    result = run_acmc1_residual_error_audit(
        {
            42: args.seed42_checkpoint,
            123: args.seed123_checkpoint,
            2026: args.seed2026_checkpoint,
        },
        args.data_root,
        args.output_root,
        device=args.device,
    )
    print(json.dumps(result["global_aggregate"], indent=2, ensure_ascii=False))
    print("ATTRIBUTION COUNTS:", result["attribution_counts"])
    print("BOTTOM CLASSES:")
    for row in result["per_class_aggregate"][:10]:
        print(
            row["class_name"],
            f"mAP={row['map50_95_mean']:.2%}",
            f"AP50={row['ap50_mean']:.2%}",
            f"AP75={row['ap75_mean']:.2%}",
            f"class@IoU50={row['class_accuracy_given_iou50_match_mean']:.2%}",
            row["attribution"],
        )
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
