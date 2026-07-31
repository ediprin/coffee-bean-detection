from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import IMAGE_SUFFIXES, discover_layout, parse_label
from .prepare_sni_fullscene import SNI21_CLASSES


CONDITION_ORDER = (
    "R0_real_val",
    "B0_empirical_mild",
    "B1_empirical_mild",
    "B2_empirical_mild",
    "B3_empirical_mild",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(-1, 4)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 4)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    intersection_left = np.maximum(left[:, None, :2], right[None, :, :2])
    intersection_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection_size = np.maximum(intersection_right - intersection_left, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    left_area = np.maximum(left[:, 2] - left[:, 0], 0.0) * np.maximum(
        left[:, 3] - left[:, 1], 0.0
    )
    right_area = np.maximum(right[:, 2] - right[:, 0], 0.0) * np.maximum(
        right[:, 3] - right[:, 1], 0.0
    )
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def diagnose_image(
    ground_truth_classes: np.ndarray,
    ground_truth_xyxy: np.ndarray,
    prediction_classes: np.ndarray,
    prediction_xyxy: np.ndarray,
    *,
    iou_threshold: float,
    max_det: int,
) -> dict:
    """Separate proposal/localization failures from localized class errors.

    This diagnostic is deliberately simpler than AP matching. For each ground
    truth instance it asks whether any post-NMS candidate reaches the IoU
    threshold, and if so whether the best-overlap candidate has the right
    class. Official performance remains the Ultralytics validation metric.
    """

    gt_classes = np.asarray(ground_truth_classes, dtype=np.int64).reshape(-1)
    gt_boxes = np.asarray(ground_truth_xyxy, dtype=np.float64).reshape(-1, 4)
    pred_classes = np.asarray(prediction_classes, dtype=np.int64).reshape(-1)
    pred_boxes = np.asarray(prediction_xyxy, dtype=np.float64).reshape(-1, 4)
    if len(gt_classes) != len(gt_boxes):
        raise ValueError("Jumlah kelas dan box ground truth berbeda")
    if len(pred_classes) != len(pred_boxes):
        raise ValueError("Jumlah kelas dan box prediksi berbeda")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold harus pada (0, 1]")
    if max_det <= 0:
        raise ValueError("max_det harus positif")

    ious = _pairwise_iou(pred_boxes, gt_boxes)
    if len(pred_boxes):
        best_prediction = ious.argmax(axis=0) if len(gt_boxes) else np.empty(0, dtype=int)
        best_iou = ious.max(axis=0) if len(gt_boxes) else np.empty(0)
    else:
        best_prediction = np.zeros(len(gt_boxes), dtype=int)
        best_iou = np.zeros(len(gt_boxes), dtype=np.float64)

    categories = []
    by_class: dict[int, Counter] = defaultdict(Counter)
    for index, class_id in enumerate(gt_classes.tolist()):
        if best_iou[index] < iou_threshold:
            category = "proposal_miss"
            predicted_class = None
        else:
            predicted_class = int(pred_classes[best_prediction[index]])
            category = (
                "localized_correct"
                if predicted_class == class_id
                else "localized_wrong_class"
            )
        categories.append(
            {
                "ground_truth_index": index,
                "ground_truth_class": class_id,
                "best_iou": float(best_iou[index]),
                "best_prediction_class": predicted_class,
                "category": category,
            }
        )
        by_class[class_id]["ground_truth"] += 1
        by_class[class_id][category] += 1

    if len(pred_boxes) and len(gt_boxes):
        prediction_best_iou = ious.max(axis=1)
        localized_predictions = int(
            np.count_nonzero(prediction_best_iou >= iou_threshold)
        )
        covered_ground_truth = int(
            np.count_nonzero(best_iou >= iou_threshold)
        )
    else:
        localized_predictions = 0
        covered_ground_truth = 0
    duplicate_candidates = max(
        0, localized_predictions - covered_ground_truth
    )
    proposal_miss = sum(
        item["category"] == "proposal_miss" for item in categories
    )
    localized_wrong = sum(
        item["category"] == "localized_wrong_class" for item in categories
    )
    localized_correct = sum(
        item["category"] == "localized_correct" for item in categories
    )
    return {
        "ground_truth_count": len(gt_boxes),
        "prediction_count": len(pred_boxes),
        "proposal_accessible": localized_correct + localized_wrong,
        "localized_correct": localized_correct,
        "localized_wrong_class": localized_wrong,
        "proposal_miss": proposal_miss,
        "localized_prediction_count": localized_predictions,
        "duplicate_candidate_count": duplicate_candidates,
        "unlocalized_prediction_count": len(pred_boxes)
        - localized_predictions,
        "saturated_max_det": len(pred_boxes) >= max_det,
        "ground_truth_diagnosis": categories,
        "by_class": {
            str(class_id): dict(counts)
            for class_id, counts in sorted(by_class.items())
        },
    }


def _ground_truth_arrays(
    label_path: Path,
    names: dict[int, str],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = parse_label(label_path, set(names))
    classes = np.asarray([box.class_id for box in boxes], dtype=np.int64)
    xyxy = np.asarray(
        [
            (
                (box.x_center - box.width / 2.0) * width,
                (box.y_center - box.height / 2.0) * height,
                (box.x_center + box.width / 2.0) * width,
                (box.y_center + box.height / 2.0) * height,
            )
            for box in boxes
        ],
        dtype=np.float64,
    ).reshape(-1, 4)
    return classes, xyxy


def _classwise_ap(metric_object, names: dict[int, str]) -> dict:
    class_indices = [
        int(value)
        for value in np.asarray(metric_object.ap_class_index).reshape(-1)
    ]
    average_precision = np.asarray(metric_object.ap, dtype=np.float64)
    if average_precision.ndim == 1:
        average_precision = average_precision[:, None]
    values = {
        names[class_id]: float(average_precision[row].mean())
        for row, class_id in enumerate(class_indices)
        if class_id in names and row < len(average_precision)
    }
    present = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "map50_95_by_class": values,
        "macro_map50_95": float(present.mean()) if len(present) else None,
        "bottom3_map50_95": (
            float(np.sort(present)[: min(3, len(present))].mean())
            if len(present)
            else None
        ),
        "worst_map50_95": float(present.min()) if len(present) else None,
        "classes_without_ground_truth": [
            names[class_id]
            for class_id in sorted(set(names) - set(class_indices))
        ],
    }


def _metric_summary(metrics, names: dict[int, str]) -> dict:
    result = {
        key: float(value) for key, value in metrics.results_dict.items()
    }
    box = getattr(metrics, "box", None)
    if box is not None and getattr(box, "ap", None) is not None:
        result.update(_classwise_ap(box, names))
    return result


def _aggregate_diagnosis(
    rows: list[dict],
    per_class: dict[int, Counter],
    names: dict[int, str],
) -> dict:
    totals = Counter()
    for row in rows:
        for key in (
            "ground_truth_count",
            "prediction_count",
            "proposal_accessible",
            "localized_correct",
            "localized_wrong_class",
            "proposal_miss",
            "duplicate_candidate_count",
            "unlocalized_prediction_count",
        ):
            totals[key] += int(row[key])
    images = len(rows)
    saturated = sum(bool(row["saturated_max_det"]) for row in rows)
    ground_truth = totals["ground_truth_count"]
    accessible = totals["proposal_accessible"]
    class_rows = {}
    for class_id, counts in sorted(per_class.items()):
        class_ground_truth = counts["ground_truth"]
        class_accessible = (
            counts["localized_correct"] + counts["localized_wrong_class"]
        )
        class_rows[names[class_id]] = {
            **dict(counts),
            "proposal_recall_at_50": (
                class_accessible / class_ground_truth
                if class_ground_truth
                else None
            ),
            "conditional_class_accuracy": (
                counts["localized_correct"] / class_accessible
                if class_accessible
                else None
            ),
        }
    return {
        **dict(totals),
        "images": images,
        "saturated_images": saturated,
        "saturation_rate": saturated / images if images else 0.0,
        "proposal_recall_at_50": (
            accessible / ground_truth if ground_truth else 0.0
        ),
        "proposal_miss_rate": (
            totals["proposal_miss"] / ground_truth
            if ground_truth
            else 0.0
        ),
        "localized_wrong_class_rate": (
            totals["localized_wrong_class"] / ground_truth
            if ground_truth
            else 0.0
        ),
        "conditional_class_accuracy": (
            totals["localized_correct"] / accessible
            if accessible
            else 0.0
        ),
        "count_mae": (
            sum(
                abs(row["prediction_count"] - row["ground_truth_count"])
                for row in rows
            )
            / images
            if images
            else 0.0
        ),
        "count_bias": (
            sum(
                row["prediction_count"] - row["ground_truth_count"]
                for row in rows
            )
            / images
            if images
            else 0.0
        ),
        "exact_count_rate": (
            sum(
                row["prediction_count"] == row["ground_truth_count"]
                for row in rows
            )
            / images
            if images
            else 0.0
        ),
        "per_class": class_rows,
    }


def _diagnose_dataset(
    model,
    data_root: Path,
    output_root: Path,
    *,
    device: str | None,
    imgsz: int,
    confidence: float,
    nms_iou: float,
    diagnostic_iou: float,
    max_det: int,
    batch_size: int,
) -> dict:
    layout = discover_layout(data_root)
    if layout.names != {index: name for index, name in enumerate(SNI21_CLASSES)}:
        raise RuntimeError(f"Mapping kelas bukan canonical SNI-21: {data_root}")
    if "val" not in layout.splits:
        raise FileNotFoundError(f"Split val tidak tersedia: {data_root}")
    image_root, label_root = layout.splits["val"]
    images = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"Split val kosong: {data_root}")

    rows: list[dict] = []
    records_path = output_root / "prediction_records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    per_class: dict[int, Counter] = defaultdict(Counter)
    with records_path.open("w", encoding="utf-8") as records:
        for start in range(0, len(images), batch_size):
            batch_paths = images[start : start + batch_size]
            kwargs = {
                "source": [str(path) for path in batch_paths],
                "imgsz": imgsz,
                "conf": confidence,
                "iou": nms_iou,
                "max_det": max_det,
                "batch": len(batch_paths),
                "stream": False,
                "verbose": False,
            }
            if device is not None:
                kwargs["device"] = device
            predictions = model.predict(**kwargs)
            for image_path, result in zip(batch_paths, predictions):
                relative = image_path.relative_to(image_root)
                label_path = (label_root / relative).with_suffix(".txt")
                with Image.open(image_path) as image:
                    width, height = image.size
                gt_classes, gt_xyxy = _ground_truth_arrays(
                    label_path, layout.names, width, height
                )
                if result.boxes is None:
                    pred_classes = np.empty(0, dtype=np.int64)
                    pred_xyxy = np.empty((0, 4), dtype=np.float64)
                    pred_confidence = np.empty(0, dtype=np.float64)
                else:
                    pred_classes = (
                        result.boxes.cls.detach().cpu().numpy().astype(np.int64)
                    )
                    pred_xyxy = (
                        result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
                    )
                    pred_confidence = (
                        result.boxes.conf.detach().cpu().numpy().astype(np.float64)
                    )
                diagnosis = diagnose_image(
                    gt_classes,
                    gt_xyxy,
                    pred_classes,
                    pred_xyxy,
                    iou_threshold=diagnostic_iou,
                    max_det=max_det,
                )
                for class_id, counts in diagnosis.pop("by_class").items():
                    per_class[int(class_id)].update(counts)
                row = {
                    "image": str(relative).replace("\\", "/"),
                    **{
                        key: value
                        for key, value in diagnosis.items()
                        if key != "ground_truth_diagnosis"
                    },
                }
                rows.append(row)
                record = {
                    **row,
                    "width": width,
                    "height": height,
                    "ground_truth": [
                        {
                            "class_id": int(class_id),
                            "xyxy": [float(value) for value in box],
                        }
                        for class_id, box in zip(gt_classes, gt_xyxy)
                    ],
                    "predictions": [
                        {
                            "class_id": int(class_id),
                            "confidence": float(score),
                            "xyxy": [float(value) for value in box],
                        }
                        for class_id, score, box in zip(
                            pred_classes, pred_confidence, pred_xyxy
                        )
                    ],
                    "ground_truth_diagnosis": diagnosis[
                        "ground_truth_diagnosis"
                    ],
                }
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"  diagnosis {min(start + len(batch_paths), len(images))}/"
                f"{len(images)}",
                flush=True,
            )

    csv_path = output_root / "image_diagnosis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = _aggregate_diagnosis(rows, per_class, layout.names)
    aggregate["records"] = str(records_path)
    aggregate["image_table"] = str(csv_path)
    return aggregate


def run_sni21_density_evaluation(
    checkpoint: str | Path,
    benchmark_root: str | Path,
    output_root: str | Path,
    *,
    real_root: str | Path | None = None,
    device: str | None = "0",
    imgsz: int = 640,
    batch_size: int = 8,
    confidence: float = 0.001,
    nms_iou: float = 0.7,
    diagnostic_iou: float = 0.5,
    max_det: int = 300,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "Ultralytics belum terpasang. Jalankan `pip install -e .`."
        ) from error

    checkpoint = Path(checkpoint).expanduser().resolve()
    benchmark_root = Path(benchmark_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    setup_path = benchmark_root / "setup_core_summary.json"
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    if setup.get("ready_for_evaluation") is not True:
        raise RuntimeError(f"Benchmark belum siap: {setup_path}")
    if setup.get("training_executed") is not False:
        raise RuntimeError("Setup benchmark mencatat training tidak semestinya")
    if setup.get("test_images_accessed") is not False:
        raise RuntimeError("Setup benchmark telah mengakses test")

    conditions: dict[str, Path] = {}
    if real_root is not None:
        conditions["R0_real_val"] = Path(real_root).expanduser().resolve()
    for name in CONDITION_ORDER:
        if name == "R0_real_val":
            continue
        if name not in setup["arms"]:
            raise RuntimeError(f"Arm benchmark hilang: {name}")
        conditions[name] = benchmark_root / name

    expected_names = {
        index: name for index, name in enumerate(SNI21_CLASSES)
    }
    for name, root in conditions.items():
        layout = discover_layout(root)
        if "val" not in layout.splits:
            raise FileNotFoundError(f"{name}: split val tidak tersedia")
        if layout.names != expected_names:
            raise RuntimeError(f"{name}: mapping kelas berbeda dari SNI-21")

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256(checkpoint)
    run_config = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "imgsz": imgsz,
        "batch_size": batch_size,
        "confidence": confidence,
        "nms_iou": nms_iou,
        "diagnostic_iou": diagnostic_iou,
        "max_det": max_det,
        "split": "val",
        "training_executed": False,
        "test_images_accessed": False,
    }
    model = YOLO(str(checkpoint))
    reports = {}
    ordered_conditions = [
        name for name in CONDITION_ORDER if name in conditions
    ]
    for index, name in enumerate(ordered_conditions, 1):
        data_root = conditions[name]
        condition_root = output_root / name
        report_path = condition_root / "evaluation.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("run_config") != run_config:
                raise RuntimeError(
                    f"Konfigurasi evaluasi lama konflik: {report_path}"
                )
            if report.get("complete") is not True:
                raise RuntimeError(f"Report parsial ditemukan: {report_path}")
            print(
                f"[{index}/{len(ordered_conditions)}] Reuse {name}",
                flush=True,
            )
            reports[name] = report
            continue

        print(
            f"\n[{index}/{len(ordered_conditions)}] VALIDATE {name}",
            flush=True,
        )
        layout = discover_layout(data_root)
        validation_kwargs = {
            "data": str(layout.yaml_path),
            "split": "val",
            "imgsz": imgsz,
            "batch": batch_size,
            "conf": confidence,
            "iou": nms_iou,
            "max_det": max_det,
            "plots": False,
            "verbose": True,
            "project": str(condition_root / "ultralytics"),
            "name": "validation",
            "exist_ok": True,
        }
        if device is not None:
            validation_kwargs["device"] = device
        metrics = model.val(**validation_kwargs)
        print(
            f"[{index}/{len(ordered_conditions)}] DIAGNOSE {name}",
            flush=True,
        )
        diagnosis = _diagnose_dataset(
            model,
            data_root,
            condition_root,
            device=device,
            imgsz=imgsz,
            confidence=confidence,
            nms_iou=nms_iou,
            diagnostic_iou=diagnostic_iou,
            max_det=max_det,
            batch_size=batch_size,
        )
        report = {
            "format": "coffee_detector.sni21_density_evaluation.v1",
            "condition": name,
            "data_root": str(data_root),
            "run_config": run_config,
            "official_metrics": _metric_summary(metrics, layout.names),
            "diagnosis": diagnosis,
            "resampling_units": (
                setup["arms"].get(name, {}).get("resampling_units")
            ),
            "development_only": True,
            "complete": True,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        reports[name] = report

    rows = []
    for name in CONDITION_ORDER:
        if name not in reports:
            continue
        metrics = reports[name]["official_metrics"]
        diagnosis = reports[name]["diagnosis"]
        rows.append(
            {
                "condition": name,
                "map50_95": metrics.get("metrics/mAP50-95(B)"),
                "map50": metrics.get("metrics/mAP50(B)"),
                "precision": metrics.get("metrics/precision(B)"),
                "recall": metrics.get("metrics/recall(B)"),
                "macro_map50_95": metrics.get("macro_map50_95"),
                "bottom3_map50_95": metrics.get("bottom3_map50_95"),
                "worst_map50_95": metrics.get("worst_map50_95"),
                "proposal_recall_at_50": diagnosis["proposal_recall_at_50"],
                "conditional_class_accuracy": diagnosis[
                    "conditional_class_accuracy"
                ],
                "proposal_miss_rate": diagnosis["proposal_miss_rate"],
                "localized_wrong_class_rate": diagnosis[
                    "localized_wrong_class_rate"
                ],
                "saturation_rate": diagnosis["saturation_rate"],
                "count_mae": diagnosis["count_mae"],
                "count_bias": diagnosis["count_bias"],
            }
        )
    table_path = output_root / "density_evaluation_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format": "coffee_detector.sni21_density_evaluation_summary.v1",
        "run_config": run_config,
        "benchmark_setup": str(setup_path),
        "conditions": list(reports),
        "reports": {
            name: str(output_root / name / "evaluation.json")
            for name in reports
        },
        "table": str(table_path),
        "rows": rows,
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "performance_uncertainty_computed": False,
        "next_action": (
            "Interpret proposal miss, localized wrong class, and max_det "
            "saturation before modifying or training a detector."
        ),
    }
    summary_path = output_root / "density_evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n=== DENSITY EVALUATION COMPLETE ===", flush=True)
    print("TRAINING   : False", flush=True)
    print("TEST ACCESS: False", flush=True)
    print("SUMMARY    :", summary_path, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one frozen SNI-21 checkpoint on R0 and held-out B0-B3 "
            "without training or test access."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--real-root")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--diagnostic-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    run_sni21_density_evaluation(
        args.checkpoint,
        args.benchmark_root,
        args.output_root,
        real_root=args.real_root,
        device=args.device,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        diagnostic_iou=args.diagnostic_iou,
        max_det=args.max_det,
    )


if __name__ == "__main__":
    main()
