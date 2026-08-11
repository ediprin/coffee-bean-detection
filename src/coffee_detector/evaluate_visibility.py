from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> np.ndarray:
    x, y, width, height = (float(value) for value in box)
    return np.asarray([x, y, x + width, y + height], dtype=np.float64)


def _box_iou(box: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
    if not boxes:
        return np.empty(0, dtype=np.float64)
    matrix = np.stack(boxes)
    top_left = np.maximum(box[:2], matrix[:, :2])
    bottom_right = np.minimum(box[2:], matrix[:, 2:])
    intersection = np.maximum(0.0, bottom_right - top_left)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    matrix_area = np.maximum(0.0, matrix[:, 2] - matrix[:, 0]) * np.maximum(
        0.0, matrix[:, 3] - matrix[:, 1]
    )
    union = box_area + matrix_area - intersection_area
    return np.divide(
        intersection_area,
        union,
        out=np.zeros_like(intersection_area),
        where=union > 0,
    )


def _interpolated_ap(true_positive: list[int], false_positive: list[int], positives: int) -> tuple[float, float]:
    if positives <= 0:
        return float("nan"), float("nan")
    if not true_positive:
        return 0.0, 0.0
    tp = np.cumsum(np.asarray(true_positive, dtype=np.float64))
    fp = np.cumsum(np.asarray(false_positive, dtype=np.float64))
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1e-12)
    samples = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision[recall >= level])) if np.any(recall >= level) else 0.0
        for level in samples
    ]
    return float(np.mean(interpolated)), float(recall[-1])


def _evaluate_class_bin(
    predictions: list[dict],
    annotations: list[dict],
    category_id: int,
    bin_name: str,
    iou_threshold: float,
) -> dict | None:
    target: dict[int, list[np.ndarray]] = defaultdict(list)
    ignored: dict[int, list[np.ndarray]] = defaultdict(list)
    for row in annotations:
        if int(row["category_id"]) != category_id or row.get("bbox") is None:
            continue
        image_id = int(row["image_id"])
        is_ignored = bool(int(row.get("ignore", 0)))
        is_target = not is_ignored and (
            bin_name == "all" or str(row.get("visibility_bin")) == bin_name
        )
        if is_target:
            target[image_id].append(_xywh_to_xyxy(row["bbox"]))
        else:
            # Objects from other visibility bins remain ignore regions.  This
            # prevents a correct detection of a non-target bin becoming an FP.
            ignored[image_id].append(_xywh_to_xyxy(row["bbox"]))
    positives = sum(len(items) for items in target.values())
    if positives == 0:
        return None
    detections = sorted(
        (
            row
            for row in predictions
            if int(row["category_id"]) == category_id
        ),
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    matched = {image_id: [False] * len(items) for image_id, items in target.items()}
    tp: list[int] = []
    fp: list[int] = []
    duplicate_predictions = 0
    ignored_predictions = 0
    for detection in detections:
        image_id = int(detection["image_id"])
        box = _xywh_to_xyxy(detection["bbox"])
        target_boxes = target.get(image_id, [])
        target_ious = _box_iou(box, target_boxes)
        if target_ious.size and float(target_ious.max()) >= iou_threshold:
            order = np.argsort(-target_ious)
            match_index = next(
                (
                    int(index)
                    for index in order
                    if float(target_ious[index]) >= iou_threshold
                    and not matched[image_id][int(index)]
                ),
                None,
            )
            if match_index is not None:
                matched[image_id][match_index] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
                duplicate_predictions += 1
            continue
        ignore_ious = _box_iou(box, ignored.get(image_id, []))
        if ignore_ious.size and float(ignore_ious.max()) >= iou_threshold:
            ignored_predictions += 1
            continue
        tp.append(0)
        fp.append(1)
    ap, recall = _interpolated_ap(tp, fp, positives)
    return {
        "positives": positives,
        "detections": len(detections),
        "evaluated_detections": len(tp),
        "ignored_detections": ignored_predictions,
        "true_positive": int(sum(tp)),
        "false_positive": int(sum(fp)),
        "duplicate_predictions": duplicate_predictions,
        "ap": ap,
        "recall": recall,
    }


def evaluate_predictions_by_visibility(
    predictions: list[dict],
    annotations: list[dict],
    categories: dict[int, str],
    *,
    iou_thresholds: tuple[float, ...] = tuple(np.arange(0.50, 0.96, 0.05)),
) -> dict:
    bins = sorted(
        {
            str(row["visibility_bin"])
            for row in annotations
            if not int(row.get("ignore", 0)) and row.get("visibility_bin")
        }
    )
    result = {}
    for bin_name in ["all", *bins]:
        per_class = {}
        for category_id, name in categories.items():
            threshold_rows = []
            for threshold in iou_thresholds:
                row = _evaluate_class_bin(
                    predictions,
                    annotations,
                    category_id,
                    bin_name,
                    float(threshold),
                )
                if row is not None:
                    threshold_rows.append((float(threshold), row))
            if not threshold_rows:
                continue
            ap50_row = min(threshold_rows, key=lambda item: abs(item[0] - 0.50))[1]
            per_class[name] = {
                "positives": ap50_row["positives"],
                "ap50": ap50_row["ap"],
                "map50_95": float(
                    np.mean([item[1]["ap"] for item in threshold_rows])
                ),
                "recall50": ap50_row["recall"],
                "false_positive50": ap50_row["false_positive"],
                "duplicate_predictions50": ap50_row["duplicate_predictions"],
                "ignored_detections50": ap50_row["ignored_detections"],
            }
        result[bin_name] = {
            "classes_evaluated": len(per_class),
            "ap50": float(np.mean([row["ap50"] for row in per_class.values()])) if per_class else None,
            "map50_95": float(np.mean([row["map50_95"] for row in per_class.values()])) if per_class else None,
            "recall50": float(np.mean([row["recall50"] for row in per_class.values()])) if per_class else None,
            "per_class": per_class,
        }
    return result


def count_metrics(
    predictions: list[dict],
    annotations: list[dict],
    image_ids: list[int],
    *,
    confidence: float = 0.25,
    duplicate_iou: float = 0.50,
) -> dict:
    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    pred_by_image: dict[int, list[dict]] = defaultdict(list)
    for row in annotations:
        if not int(row.get("ignore", 0)) and row.get("bbox") is not None:
            gt_by_image[int(row["image_id"])].append(row)
    for row in predictions:
        if float(row["score"]) >= confidence:
            pred_by_image[int(row["image_id"])].append(row)
    errors = []
    over = under = exact = 0
    duplicate_predictions = 0
    total_predictions = 0
    for image_id in image_ids:
        ground_truth = gt_by_image.get(image_id, [])
        detections = sorted(
            pred_by_image.get(image_id, []),
            key=lambda row: float(row["score"]),
            reverse=True,
        )
        error = len(detections) - len(ground_truth)
        errors.append(error)
        over += error > 0
        under += error < 0
        exact += error == 0
        total_predictions += len(detections)
        matched = [False] * len(ground_truth)
        for detection in detections:
            candidates = [
                index
                for index, target in enumerate(ground_truth)
                if int(target["category_id"]) == int(detection["category_id"])
            ]
            ious = _box_iou(
                _xywh_to_xyxy(detection["bbox"]),
                [_xywh_to_xyxy(ground_truth[index]["bbox"]) for index in candidates],
            )
            if not ious.size or float(ious.max()) < duplicate_iou:
                continue
            ordered = np.argsort(-ious)
            unmatched = next(
                (
                    candidates[int(index)]
                    for index in ordered
                    if float(ious[index]) >= duplicate_iou
                    and not matched[candidates[int(index)]]
                ),
                None,
            )
            if unmatched is not None:
                matched[unmatched] = True
            else:
                duplicate_predictions += 1
    count = len(image_ids)
    absolute = [abs(value) for value in errors]
    return {
        "images": count,
        "confidence": confidence,
        "mean_absolute_count_error": float(np.mean(absolute)) if count else 0.0,
        "mean_count_bias": float(np.mean(errors)) if count else 0.0,
        "median_count_error": float(np.median(errors)) if count else 0.0,
        "exact_count_match_rate": exact / count if count else 0.0,
        "over_count_rate": over / count if count else 0.0,
        "under_count_rate": under / count if count else 0.0,
        "predicted_instances": total_predictions,
        "duplicate_predictions": duplicate_predictions,
        "duplicate_prediction_rate": (
            duplicate_predictions / total_predictions if total_predictions else 0.0
        ),
    }


def evaluate_visibility(
    checkpoint: str | Path,
    data_root: str | Path,
    metadata_path: str | Path,
    output: str | Path,
    *,
    device: str | None = None,
    batch_size: int = 16,
    prediction_confidence: float = 0.001,
    count_confidence: float = 0.25,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang. Jalankan `pip install -e .`.") from error
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    data_root = Path(data_root).expanduser().resolve()
    metadata_path = Path(metadata_path).expanduser().resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]) for row in metadata["categories"]}
    image_rows = sorted(metadata["images"], key=lambda row: int(row["id"]))
    image_paths = [data_root / row["file_name"] for row in image_rows]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Image metadata tidak ditemukan:\n- " + "\n- ".join(missing[:20]))
    model = YOLO(str(checkpoint))
    predictions = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        kwargs = {
            "source": [str(path) for path in batch_paths],
            "conf": prediction_confidence,
            "stream": False,
            "verbose": False,
            "batch": len(batch_paths),
            "max_det": 1000,
        }
        if device is not None:
            kwargs["device"] = device
        results = model.predict(**kwargs)
        for image_row, result in zip(image_rows[start : start + len(batch_paths)], results):
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            scores = result.boxes.conf.detach().cpu().numpy()
            for box, class_id, score in zip(boxes, classes, scores):
                x1, y1, x2, y2 = (float(value) for value in box)
                predictions.append(
                    {
                        "image_id": int(image_row["id"]),
                        "category_id": int(class_id),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
        print(
            f"Visibility evaluation: {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}",
            flush=True,
        )
    payload = {
        "format": "coffee_detector.visibility_evaluation.v1",
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "metadata": str(metadata_path),
        "prediction_confidence": prediction_confidence,
        "visibility_metrics": evaluate_predictions_by_visibility(
            predictions, metadata["annotations"], categories
        ),
        "count_metrics": count_metrics(
            predictions,
            metadata["annotations"],
            [int(row["id"]) for row in image_rows],
            confidence=count_confidence,
        ),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluasi AP per visibility bin dengan GT bin lain sebagai ignore."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prediction-confidence", type=float, default=0.001)
    parser.add_argument("--count-confidence", type=float, default=0.25)
    args = parser.parse_args()
    result = evaluate_visibility(
        args.checkpoint,
        args.data_root,
        args.metadata,
        args.output,
        device=args.device,
        batch_size=args.batch_size,
        prediction_confidence=args.prediction_confidence,
        count_confidence=args.count_confidence,
    )
    print("=== VISIBILITY EVALUATION ===")
    for name, row in result["visibility_metrics"].items():
        print(
            f"{name:10s} AP50={row['ap50']} mAP50-95={row['map50_95']} "
            f"Recall50={row['recall50']}"
        )
    print("COUNT", json.dumps(result["count_metrics"], ensure_ascii=False))
    print(f"SAVED: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
