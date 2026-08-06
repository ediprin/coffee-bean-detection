"""Robust validation-only residual error attribution for selected ACMC1 checkpoints.

Version 2 intentionally avoids private/raw YOLO26 head decoding. AP-by-IoU is
read from the public Ultralytics validation metrics, while residual detection
and class attribution is computed from final predictions matched to validation
GT boxes class-agnostically at IoU 0.50.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout, parse_label

IOU_THRESHOLDS = tuple(round(float(x), 2) for x in np.linspace(0.50, 0.95, 10))
DEFAULT_SEEDS = (42, 123, 2026)

LOW_AP50_THRESHOLD = 0.90
HIGH_IOU_DROP_THRESHOLD = 0.10
CLASS_HEADROOM_THRESHOLD = 0.10
DETECTION_GAP_THRESHOLD = 0.10
CONFIDENCE_THRESHOLD = 0.25
MATCH_IOU = 0.50


def _validate_only_layout(data_root):
    layout = discover_layout(data_root)
    if "val" not in layout.splits:
        raise RuntimeError("Validation split tidak tersedia")
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit ini menolak dataset yang mengekspos test split")
    return layout


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
        "classification_headroom_material": row["classification_headroom_iou50"] >= CLASS_HEADROOM_THRESHOLD,
        "final_detection_gap_material": (1.0 - row["detection_accessibility_iou50"]) >= DETECTION_GAP_THRESHOLD,
    }


def _attribution_label(row):
    flags = _evidence_flags(row)
    cls = flags["classification_headroom_material"]
    loc = flags["high_iou_localization_drop"]
    det = flags["final_detection_gap_material"]
    if det and flags["low_ap50"] and not cls and not loc:
        return "detection_or_confidence_limited"
    if cls and loc:
        return "mixed_classification_localization"
    if cls:
        return "classification_or_ranking_limited"
    if loc:
        return "high_iou_localization_limited"
    if det:
        return "detection_or_confidence_limited"
    return "no_single_dominant_signal"


def _val_samples(layout):
    image_root, label_root = layout.splits["val"]
    samples = []
    for image_path in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        samples.append((image_path, parse_label(label_path, set(layout.names))))
    if not samples:
        raise RuntimeError("Validation split kosong")
    return samples


def _target_tensors(image_path, annotations, device):
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Gambar tidak dapat dibaca: {image_path}")
    height, width = image.shape[:2]
    boxes, labels = [], []
    for item in annotations:
        boxes.append(
            [
                (item.x_center - item.width * 0.5) * width,
                (item.y_center - item.height * 0.5) * height,
                (item.x_center + item.width * 0.5) * width,
                (item.y_center + item.height * 0.5) * height,
            ]
        )
        labels.append(item.class_id)
    return (
        torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def _confidence_ordered_match(pred_boxes, pred_conf, target_boxes, threshold=MATCH_IOU):
    if not len(pred_boxes) or not len(target_boxes):
        return []
    matrix = box_iou(pred_boxes, target_boxes)
    available = torch.ones(len(target_boxes), dtype=torch.bool, device=target_boxes.device)
    matches = []
    for pred_index in pred_conf.argsort(descending=True).tolist():
        candidate = matrix[pred_index].clone()
        candidate[~available] = -1
        value, target_index = candidate.max(dim=0)
        if float(value) < threshold:
            continue
        target_index = int(target_index)
        available[target_index] = False
        matches.append((int(pred_index), target_index, float(value)))
    return matches


def _new_counts(num_classes):
    return {
        "targets": 0,
        "accessible": 0,
        "matched": 0,
        "correct": 0,
        "wrong": 0,
        "class_targets": np.zeros(num_classes, dtype=np.int64),
        "class_accessible": np.zeros(num_classes, dtype=np.int64),
        "class_matched": np.zeros(num_classes, dtype=np.int64),
        "class_correct": np.zeros(num_classes, dtype=np.int64),
        "confusion": np.zeros((num_classes, num_classes), dtype=np.int64),
    }


def _update_counts(counts, pred_boxes, pred_labels, pred_conf, target_boxes, target_labels):
    num_classes = len(counts["class_targets"])
    target_count = len(target_boxes)
    counts["targets"] += target_count
    if target_count:
        counts["class_targets"] += np.bincount(
            target_labels.detach().cpu().numpy(), minlength=num_classes
        )
    if not target_count:
        return

    matrix = box_iou(pred_boxes, target_boxes) if len(pred_boxes) else target_boxes.new_zeros((0, target_count))
    accessible_mask = (
        matrix.max(dim=0).values >= MATCH_IOU
        if len(pred_boxes)
        else torch.zeros(target_count, dtype=torch.bool, device=target_boxes.device)
    )
    counts["accessible"] += int(accessible_mask.sum())
    if int(accessible_mask.sum()):
        counts["class_accessible"] += np.bincount(
            target_labels[accessible_mask].detach().cpu().numpy(), minlength=num_classes
        )

    matches = _confidence_ordered_match(pred_boxes, pred_conf, target_boxes)
    counts["matched"] += len(matches)
    for pred_index, target_index, _ in matches:
        expected = int(target_labels[target_index])
        actual = int(pred_labels[pred_index])
        counts["class_matched"][expected] += 1
        counts["confusion"][expected, actual] += 1
        if expected == actual:
            counts["correct"] += 1
            counts["class_correct"][expected] += 1
        else:
            counts["wrong"] += 1


def _finalize_counts(counts, names):
    targets = max(int(counts["targets"]), 1)
    matched = max(int(counts["matched"]), 1)
    per_class = {}
    for class_id, class_name in names.items():
        class_targets = int(counts["class_targets"][class_id])
        class_matched = int(counts["class_matched"][class_id])
        class_correct = int(counts["class_correct"][class_id])
        per_class[class_name] = {
            "targets": class_targets,
            "detection_accessibility_iou50": float(
                counts["class_accessible"][class_id] / max(class_targets, 1)
            ),
            "matched_recall_iou50": float(class_matched / max(class_targets, 1)),
            "class_accuracy_given_iou50_match": float(class_correct / max(class_matched, 1)),
            "classification_headroom_iou50": float(1.0 - class_correct / max(class_matched, 1)),
        }

    confusion = {
        names[row]: {
            names[column]: int(counts["confusion"][row, column])
            for column in range(len(names))
            if counts["confusion"][row, column]
        }
        for row in range(len(names))
        if counts["confusion"][row].sum()
    }
    return {
        "targets": int(counts["targets"]),
        "accessible": int(counts["accessible"]),
        "matched": int(counts["matched"]),
        "correct_class": int(counts["correct"]),
        "wrong_class": int(counts["wrong"]),
        "detection_accessibility_iou50": float(counts["accessible"] / targets),
        "matched_recall_iou50": float(counts["matched"] / targets),
        "class_accuracy_given_iou50_match": float(counts["correct"] / matched),
        "classification_headroom_iou50": float(counts["wrong"] / matched),
        "per_class": per_class,
        "confusion": confusion,
    }


def _top_confusions(confusion, limit=20):
    rows = []
    for expected, predicted_counts in confusion.items():
        for predicted, count in predicted_counts.items():
            if expected != predicted:
                rows.append({"expected": expected, "predicted": predicted, "count": int(count)})
    return sorted(rows, key=lambda x: (-x["count"], x["expected"], x["predicted"]))[:limit]


def _run_seed(checkpoint, data_root, *, seed, device):
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint seed {seed} tidak ditemukan: {checkpoint}")
    layout = _validate_only_layout(data_root)

    print(f"SEED {seed}: validation AP matrix", flush=True)
    metrics = YOLO(str(checkpoint)).val(
        data=str(layout.yaml_path), split="val", plots=False, verbose=False, device=device
    )
    metric_object = getattr(metrics, "box", None)
    if metric_object is None or getattr(metric_object, "ap", None) is None:
        raise RuntimeError("Ultralytics tidak mengembalikan box AP matrix")
    classwise = _profile_from_ap_matrix(metric_object.ap, metric_object.ap_class_index, layout.names)
    if set(classwise) != set(layout.names.values()):
        missing = sorted(set(layout.names.values()) - set(classwise))
        raise RuntimeError(f"Validation kehilangan kelas: {missing}")

    print(f"SEED {seed}: final-detection matching", flush=True)
    model = YOLO(str(checkpoint))
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    counts = _new_counts(len(layout.names))
    samples = _val_samples(layout)
    for index, (image_path, annotations) in enumerate(samples, 1):
        target_boxes, target_labels = _target_tensors(image_path, annotations, torch_device)
        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            iou=0.70,
            max_det=500,
            device=device,
            verbose=False,
        )[0]
        if result.boxes is None:
            pred_boxes = target_boxes.new_zeros((0, 4))
            pred_conf = target_boxes.new_zeros((0,))
            pred_labels = target_labels.new_zeros((0,))
        else:
            pred_boxes = result.boxes.xyxy.to(torch_device)
            pred_conf = result.boxes.conf.to(torch_device)
            pred_labels = result.boxes.cls.long().to(torch_device)
        _update_counts(counts, pred_boxes, pred_labels, pred_conf, target_boxes, target_labels)
        if index % 100 == 0 or index == len(samples):
            print(f"SEED {seed}: {index}/{len(samples)} images", flush=True)

    final = _finalize_counts(counts, layout.names)
    rows = {}
    for class_name, profile in classwise.items():
        row = {**profile, **final["per_class"][class_name]}
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
            "detection_accessibility_iou50": final["detection_accessibility_iou50"],
            "matched_recall_iou50": final["matched_recall_iou50"],
            "class_accuracy_given_iou50_match": final["class_accuracy_given_iou50_match"],
            "classification_headroom_iou50": final["classification_headroom_iou50"],
        },
        "per_class": rows,
        "top_directional_confusions": _top_confusions(final["confusion"]),
        "confusion": final["confusion"],
    }


def _aggregate(per_seed):
    class_names = sorted(set.intersection(*(set(payload["per_class"]) for payload in per_seed.values())))
    metric_keys = (
        "ap50", "ap75", "ap95", "map50_95", "ap50_to_ap75_drop", "ap75_to_ap95_drop",
        "detection_accessibility_iou50", "matched_recall_iou50",
        "class_accuracy_given_iou50_match", "classification_headroom_iou50",
    )
    rows = []
    for class_name in class_names:
        row = {"class_name": class_name}
        for metric in metric_keys:
            values = [float(payload["per_class"][class_name][metric]) for payload in per_seed.values()]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values)
        evidence = {
            "ap50": row["ap50_mean"],
            "ap50_to_ap75_drop": row["ap50_to_ap75_drop_mean"],
            "detection_accessibility_iou50": row["detection_accessibility_iou50_mean"],
            "classification_headroom_iou50": row["classification_headroom_iou50_mean"],
        }
        row["flags"] = _evidence_flags(evidence)
        row["attribution"] = _attribution_label(evidence)
        row["attribution_seed_agreement"] = sum(
            payload["per_class"][class_name]["attribution"] == row["attribution"]
            for payload in per_seed.values()
        )
        rows.append(row)
    return sorted(rows, key=lambda r: (r["map50_95_mean"], r["class_accuracy_given_iou50_match_mean"], r["class_name"]))


def _aggregate_confusions(per_seed, limit=20):
    counts = Counter()
    for payload in per_seed.values():
        for expected, predicted_counts in payload["confusion"].items():
            for predicted, count in predicted_counts.items():
                if expected != predicted:
                    counts[(expected, predicted)] += int(count)
    return [
        {"expected": expected, "predicted": predicted, "count": count}
        for (expected, predicted), count in counts.most_common(limit)
    ]


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


def run_acmc1_residual_error_audit(checkpoints, data_root, output_root, *, device="0"):
    _validate_only_layout(data_root)
    seeds = tuple(sorted(int(seed) for seed in checkpoints))
    if seeds != DEFAULT_SEEDS:
        raise RuntimeError(f"Audit dikunci ke seeds {DEFAULT_SEEDS}, dapat {seeds}")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    per_seed = {
        str(seed): _run_seed(checkpoints[seed], data_root, seed=seed, device=device)
        for seed in DEFAULT_SEEDS
    }
    aggregate = _aggregate(per_seed)
    global_keys = (
        "precision", "recall", "map50", "map50_95", "detection_accessibility_iou50",
        "matched_recall_iou50", "class_accuracy_given_iou50_match", "classification_headroom_iou50",
    )
    global_aggregate = {}
    for key in global_keys:
        values = [float(per_seed[str(seed)]["global"][key]) for seed in DEFAULT_SEEDS]
        global_aggregate[key] = {"mean": mean(values), "std": pstdev(values)}

    category_counts = dict(Counter(row["attribution"] for row in aggregate))
    payload = {
        "protocol": "faruq-v3-acmc1-residual-error-attribution-v2",
        "selected_model": "ACMC1",
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "seeds": list(DEFAULT_SEEDS),
        "iou_thresholds": list(IOU_THRESHOLDS),
        "matching": {
            "source": "final_predictions",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": MATCH_IOU,
            "class_agnostic_matching": True,
        },
        "frozen_thresholds": {
            "low_ap50": LOW_AP50_THRESHOLD,
            "high_iou_localization_drop": HIGH_IOU_DROP_THRESHOLD,
            "classification_headroom_material": CLASS_HEADROOM_THRESHOLD,
            "final_detection_gap_material": DETECTION_GAP_THRESHOLD,
        },
        "global_aggregate": global_aggregate,
        "attribution_counts": category_counts,
        "top_directional_confusions_3seed": _aggregate_confusions(per_seed),
        "per_class_aggregate": aggregate,
        "per_seed": per_seed,
    }
    summary = output_root / "acmc1_residual_error_attribution_v2.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(aggregate, output_root / "acmc1_residual_error_attribution_v2.csv")
    payload["summary"] = str(summary)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Robust 3-seed ACMC1 validation-only residual error attribution.")
    parser.add_argument("--seed42-checkpoint", required=True)
    parser.add_argument("--seed123-checkpoint", required=True)
    parser.add_argument("--seed2026-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run_acmc1_residual_error_audit(
        {42: args.seed42_checkpoint, 123: args.seed123_checkpoint, 2026: args.seed2026_checkpoint},
        args.data_root, args.output_root, device=args.device,
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
    print("TOP CONFUSIONS:", result["top_directional_confusions_3seed"][:10])
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
