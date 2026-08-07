from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .integrated import GroundTruthRecord, IntegratedPredictionRecord, IOU_THRESHOLDS


def _box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in left)
    bx1, by1, bx2, by2 = (float(value) for value in right)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def compute_ap101(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101 recall-threshold mean of the precision envelope.

    This deliberately averages the sampled precision values rather than
    integrating them with a trapezoid. A perfect precision/recall curve must
    therefore evaluate to exactly 1.0.
    """

    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    if recall.ndim != 1 or precision.ndim != 1 or recall.shape != precision.shape:
        raise ValueError("recall dan precision harus 1-D dan berukuran sama")
    if not len(recall):
        return 0.0

    # Precision envelope, followed by the maximum precision available at or
    # above each of the 101 recall thresholds.
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_grid = np.linspace(0.0, 1.0, 101)
    sampled = np.zeros_like(recall_grid)
    for index, threshold in enumerate(recall_grid):
        valid = np.flatnonzero(recall >= threshold)
        if len(valid):
            sampled[index] = precision_envelope[valid[0]]
    return float(sampled.mean())


def detection_map_summary(
    predictions: list[IntegratedPredictionRecord],
    ground_truth: list[GroundTruthRecord],
    num_classes: int,
    *,
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS,
) -> dict:
    """Class-aware AP50:95 for a fixed set of boxes and detector scores."""

    gt_by_class_image: dict[int, dict[Path, list[GroundTruthRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in ground_truth:
        gt_by_class_image[int(record.class_id)][record.image_path].append(record)

    predictions_by_class: dict[int, list[IntegratedPredictionRecord]] = defaultdict(list)
    for record in predictions:
        predictions_by_class[int(record.predicted_class_id)].append(record)

    per_class_ap: list[float] = []
    per_class_ap50: list[float] = []
    gt_counts: list[int] = []

    for class_id in range(int(num_classes)):
        class_gt = gt_by_class_image.get(class_id, {})
        n_gt = sum(len(value) for value in class_gt.values())
        gt_counts.append(n_gt)
        class_predictions = sorted(
            predictions_by_class.get(class_id, []),
            key=lambda record: -float(record.predicted_confidence),
        )
        class_aps: list[float] = []

        for threshold in iou_thresholds:
            matched = {
                image_path: [False] * len(items)
                for image_path, items in class_gt.items()
            }
            tp = np.zeros(len(class_predictions), dtype=np.float64)
            fp = np.zeros(len(class_predictions), dtype=np.float64)

            for prediction_index, prediction in enumerate(class_predictions):
                candidates = class_gt.get(prediction.image_path, [])
                best_iou = -1.0
                best_index = -1
                for gt_index, target in enumerate(candidates):
                    if matched[prediction.image_path][gt_index]:
                        continue
                    overlap = _box_iou(prediction.predicted_xyxy, target.xyxy)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_index = gt_index
                if best_index >= 0 and best_iou >= float(threshold):
                    matched[prediction.image_path][best_index] = True
                    tp[prediction_index] = 1.0
                else:
                    fp[prediction_index] = 1.0

            if n_gt <= 0:
                class_aps.append(float("nan"))
                continue
            cumulative_tp = np.cumsum(tp)
            cumulative_fp = np.cumsum(fp)
            recall = cumulative_tp / float(n_gt)
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
            class_aps.append(compute_ap101(recall, precision))

        finite = [value for value in class_aps if math.isfinite(value)]
        per_class_ap.append(float(np.mean(finite)) if finite else float("nan"))
        per_class_ap50.append(float(class_aps[0]))

    active = [index for index, count in enumerate(gt_counts) if count > 0]
    if not active:
        raise RuntimeError("Tidak ada GT class aktif")
    active_ap = [per_class_ap[index] for index in active]
    active_ap50 = [per_class_ap50[index] for index in active]
    ordered = sorted(active_ap)
    return {
        "map50_95": float(np.mean(active_ap)),
        "map50": float(np.mean(active_ap50)),
        "bottom3_ap50_95": float(np.mean(ordered[: min(3, len(ordered))])),
        "worst_ap50_95": float(ordered[0]),
        "per_class_ap50_95": {str(index): float(per_class_ap[index]) for index in active},
        "per_class_ap50": {str(index): float(per_class_ap50[index]) for index in active},
        "gt_counts": {str(index): int(gt_counts[index]) for index in active},
        "iou_thresholds": [float(value) for value in iou_thresholds],
        "prediction_count": len(predictions),
        "ground_truth_count": len(ground_truth),
        "ap_interpolation": "101-recall-threshold mean of precision envelope",
    }
