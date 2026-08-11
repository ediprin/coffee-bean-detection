from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from .dataset import IMAGE_SUFFIXES, discover_layout, parse_label
from .prepare_sni_fullscene import SNI21_CLASSES, canonical_source_identity
from .run_sni21_density_evaluation import (
    _diagnose_dataset,
    _metric_summary,
    _sha256,
    _write_runtime_dataset_yaml,
)


CONDITION = "R0_real_val_repaste"
PROTOCOL = "docs/SNI21_R0_REPASTE_CONTROL_PROTOCOL.md"


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} bukan JSON object: {path}")
    return payload


def _dataset_and_parent(image_path: Path) -> tuple[str, str]:
    name = image_path.name
    if "__" not in name:
        raise ValueError(f"Nama A0 tidak menyimpan dataset source: {name}")
    dataset, source_name = name.split("__", 1)
    return dataset, canonical_source_identity(source_name)


def _pixel_boxes(label_path: Path, width: int, height: int) -> list[dict]:
    boxes = parse_label(label_path, set(range(len(SNI21_CLASSES))))
    rows = []
    for index, box in enumerate(boxes):
        x1 = max(0.0, (box.x_center - box.width / 2.0) * width)
        y1 = max(0.0, (box.y_center - box.height / 2.0) * height)
        x2 = min(float(width), (box.x_center + box.width / 2.0) * width)
        y2 = min(float(height), (box.y_center + box.height / 2.0) * height)
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"BBox tidak valid: {label_path}:{index}")
        rows.append(
            {
                "index": index,
                "class_id": int(box.class_id),
                "xyxy": [x1, y1, x2, y2],
                "aspect_ratio": box_width / box_height,
            }
        )
    return rows


def match_assets_to_boxes(boxes: list[dict], assets: list[dict]) -> list[tuple[dict, dict]]:
    """Pair same-class assets and boxes by ordered log aspect ratio.

    The crop archive does not retain COCO annotation IDs. In one dimension,
    sorting both aspect-ratio sequences minimizes total absolute rank-wise
    mismatch and is deterministic. Coverage and mismatch are reported; this
    function never invents a label or crosses classes.
    """

    boxes_by_class: dict[int, list[dict]] = defaultdict(list)
    assets_by_class: dict[int, list[dict]] = defaultdict(list)
    for box in boxes:
        boxes_by_class[int(box["class_id"])].append(box)
    for asset in assets:
        assets_by_class[int(asset["class_id"])].append(asset)

    pairs = []
    for class_id in sorted(set(boxes_by_class) | set(assets_by_class)):
        class_boxes = sorted(
            boxes_by_class[class_id],
            key=lambda row: (math.log(float(row["aspect_ratio"])), int(row["index"])),
        )
        class_assets = sorted(
            assets_by_class[class_id],
            key=lambda row: (
                math.log(float(row["intrinsic_aspect_ratio"])),
                str(row["asset_id"]),
            ),
        )
        for box, asset in zip(class_boxes, class_assets):
            pairs.append((box, asset))
    return sorted(pairs, key=lambda pair: int(pair[0]["index"]))


def classify_repaste_retention(
    map_retention: float, conditional_accuracy_retention: float
) -> str:
    lower = min(map_retention, conditional_accuracy_retention)
    if lower >= 0.80:
        return "cutout_repaste_not_primary_cause"
    if lower >= 0.50:
        return "cutout_repaste_material_partial_cause"
    return "cutout_repaste_dominant_cause"


def prepare_r0_repaste_dataset(
    real_root: str | Path,
    object_library_root: str | Path,
    output_root: str | Path,
    *,
    minimum_coverage: float = 0.85,
) -> dict:
    real_root = Path(real_root).expanduser().resolve()
    library_root = Path(object_library_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage harus pada (0, 1]")
    layout = discover_layout(real_root)
    expected_names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    if layout.names != expected_names:
        raise RuntimeError("R0 bukan canonical SNI-21")
    if "val" not in layout.splits:
        raise FileNotFoundError("R0 tidak memiliki split validation")

    library = _read_json(library_root / "object_library.json", "Object library")
    library_names = {int(key): str(value) for key, value in library["classes"].items()}
    if library_names != expected_names:
        raise RuntimeError("Object library bukan canonical SNI-21")
    assets_by_parent: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for asset in library["assets"]:
        if str(asset.get("source_split")) != "val":
            raise RuntimeError("Object library memuat aset non-validation")
        key = (str(asset["source_dataset"]), str(asset["source_parent_id"]))
        assets_by_parent[key].append(asset)

    image_root, label_root = layout.splits["val"]
    images = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError("R0 validation kosong")
    setup_path = output_root / "repaste_setup.json"
    if setup_path.is_file():
        existing = _read_json(setup_path, "Setup R0-repaste")
        if existing.get("complete") is not True:
            raise RuntimeError("Setup R0-repaste lama tidak lengkap")
        return existing
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Output R0-repaste parsial/tidak dikenal: {output_root}")

    output_images = output_root / "val" / "images"
    output_labels = output_root / "val" / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    for kind in ("images", "labels"):
        (output_root / "train" / kind).mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    matched_boxes = 0
    images_with_matches = 0
    mismatch_logs = []
    by_class = defaultdict(Counter)
    print(f"R0-REPASTE: mulai 0/{len(images)} gambar", flush=True)
    for image_index, image_path in enumerate(images, 1):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        boxes = _pixel_boxes(label_path, image.width, image.height)
        dataset, parent_id = _dataset_and_parent(image_path)
        parent_assets = assets_by_parent.get((dataset, parent_id), [])
        pairs = match_assets_to_boxes(boxes, parent_assets)
        total_boxes += len(boxes)
        matched_boxes += len(pairs)
        images_with_matches += bool(pairs)
        for box in boxes:
            by_class[int(box["class_id"])]["ground_truth"] += 1
        for box, asset in pairs:
            class_id = int(box["class_id"])
            by_class[class_id]["matched"] += 1
            x1, y1, x2, y2 = box["xyxy"]
            left = max(0, int(round(x1)))
            top = max(0, int(round(y1)))
            right = min(image.width, max(left + 1, int(round(x2))))
            bottom = min(image.height, max(top + 1, int(round(y2))))
            asset_path = library_root / str(asset["image"])
            with Image.open(asset_path) as cutout_source:
                cutout = cutout_source.convert("RGBA").resize(
                    (right - left, bottom - top), Image.Resampling.LANCZOS
                )
            image.paste(cutout, (left, top), cutout)
            ratio_error = abs(
                math.log(float(box["aspect_ratio"]))
                - math.log(float(asset["intrinsic_aspect_ratio"]))
            )
            if ratio_error > math.log(1.5) and len(mismatch_logs) < 100:
                mismatch_logs.append(
                    {
                        "image": relative.as_posix(),
                        "class": SNI21_CLASSES[class_id],
                        "asset_id": str(asset["asset_id"]),
                        "aspect_ratio_log_error": ratio_error,
                    }
                )

        target_image = (output_images / relative).with_suffix(".png")
        target_label = (output_labels / relative).with_suffix(".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        image.save(target_image, format="PNG", compress_level=3)
        shutil.copy2(label_path, target_label)
        if image_index % 50 == 0 or image_index == len(images):
            print(f"  repaste {image_index}/{len(images)}", flush=True)

    coverage = matched_boxes / total_boxes if total_boxes else 0.0
    class_coverage = {
        SNI21_CLASSES[class_id]: {
            **dict(counts),
            "coverage": counts["matched"] / counts["ground_truth"],
        }
        for class_id, counts in sorted(by_class.items())
    }
    payload = {
        "format": "coffee_detector.sni21_r0_repaste_setup.v1",
        "protocol": PROTOCOL,
        "real_root": str(real_root),
        "object_library": str(library_root),
        "output_root": str(output_root),
        "images": len(images),
        "images_with_matches": images_with_matches,
        "ground_truth_boxes": total_boxes,
        "matched_boxes": matched_boxes,
        "coverage": coverage,
        "minimum_coverage": minimum_coverage,
        "coverage_pass": coverage >= minimum_coverage,
        "class_coverage": class_coverage,
        "high_aspect_mismatch_examples": mismatch_logs,
        "matching_key": "source_dataset + source_parent_id + class_id",
        "within_class_matching": "sorted log aspect ratio",
        "output_codec": "PNG to preserve decoded pixels outside pasted boxes",
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "complete": True,
    }
    data_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "names": expected_names,
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    setup_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if coverage < minimum_coverage:
        raise RuntimeError(
            f"Coverage repaste {coverage:.2%} < {minimum_coverage:.2%}; "
            f"jangan evaluasi. Lihat {setup_path}"
        )
    return payload


def _row_from_report(condition: str, report: dict) -> dict:
    metrics = report["official_metrics"]
    diagnosis = report["diagnosis"]
    return {
        "condition": condition,
        "map50_95": metrics.get("metrics/mAP50-95(B)"),
        "map50": metrics.get("metrics/mAP50(B)"),
        "precision": metrics.get("metrics/precision(B)"),
        "recall": metrics.get("metrics/recall(B)"),
        "macro_map50_95": metrics.get("macro_map50_95"),
        "bottom3_map50_95": metrics.get("bottom3_map50_95"),
        "worst_map50_95": metrics.get("worst_map50_95"),
        "proposal_recall_at_50": diagnosis["proposal_recall_at_50"],
        "conditional_class_accuracy": diagnosis["conditional_class_accuracy"],
    }


def run_sni21_r0_repaste_control(
    checkpoint: str | Path,
    real_root: str | Path,
    source_benchmark_root: str | Path,
    density_evaluation_root: str | Path,
    benchmark_output_root: str | Path,
    evaluation_output_root: str | Path,
    *,
    minimum_coverage: float = 0.85,
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
        raise RuntimeError("Ultralytics belum terpasang. Jalankan pip install -e .") from error

    checkpoint = Path(checkpoint).expanduser().resolve()
    real_root = Path(real_root).expanduser().resolve()
    source_root = Path(source_benchmark_root).expanduser().resolve()
    density_root = Path(density_evaluation_root).expanduser().resolve()
    benchmark_root = Path(benchmark_output_root).expanduser().resolve()
    evaluation_root = Path(evaluation_output_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    density_summary = _read_json(
        density_root / "density_evaluation_summary.json", "Summary density"
    )
    if density_summary.get("training_executed") is not False:
        raise RuntimeError("Summary density tidak menjamin training=False")
    if density_summary.get("test_images_accessed") is not False:
        raise RuntimeError("Summary density telah mengakses test")
    if _sha256(checkpoint) != density_summary["run_config"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint berbeda dari R0 yang sudah dievaluasi")
    old_rows = {row["condition"]: row for row in density_summary["rows"]}
    if "R0_real_val" not in old_rows:
        raise RuntimeError("Summary density belum memiliki R0_real_val")

    print("[1/2] Materialisasi R0-repaste validation", flush=True)
    setup = prepare_r0_repaste_dataset(
        real_root,
        source_root / "val_object_library",
        benchmark_root / CONDITION,
        minimum_coverage=minimum_coverage,
    )
    if setup.get("coverage_pass") is not True:
        raise RuntimeError("Coverage R0-repaste belum PASS")

    evaluation_root.mkdir(parents=True, exist_ok=True)
    condition_root = evaluation_root / CONDITION
    report_path = condition_root / "evaluation.json"
    run_config = {
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
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
    if report_path.is_file():
        report = _read_json(report_path, "Evaluasi R0-repaste")
        if report.get("run_config") != run_config or report.get("complete") is not True:
            raise RuntimeError("Evaluasi R0-repaste lama konflik atau parsial")
        print("[2/2] Reuse evaluasi R0-repaste", flush=True)
    else:
        print("[2/2] Validate dan diagnose R0-repaste", flush=True)
        model = YOLO(str(checkpoint))
        repaste_root = benchmark_root / CONDITION
        layout = discover_layout(repaste_root)
        runtime_yaml = _write_runtime_dataset_yaml(
            layout, condition_root / "runtime_data.yaml"
        )
        kwargs = {
            "data": str(runtime_yaml),
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
            kwargs["device"] = device
        metrics = model.val(**kwargs)
        diagnosis = _diagnose_dataset(
            model,
            repaste_root,
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
            "format": "coffee_detector.sni21_r0_repaste_evaluation.v1",
            "condition": CONDITION,
            "data_root": str(repaste_root),
            "run_config": run_config,
            "official_metrics": _metric_summary(metrics, layout.names),
            "diagnosis": diagnosis,
            "development_only": True,
            "complete": True,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    real_row = old_rows["R0_real_val"]
    repaste_row = _row_from_report(CONDITION, report)
    map_retention = repaste_row["map50_95"] / real_row["map50_95"]
    class_retention = (
        repaste_row["conditional_class_accuracy"]
        / real_row["conditional_class_accuracy"]
    )
    attribution = {
        "map50_95_retention": map_retention,
        "conditional_class_accuracy_retention": class_retention,
        "interpretation": classify_repaste_retention(map_retention, class_retention),
    }
    rows = [real_row, repaste_row]
    table_path = evaluation_root / "r0_repaste_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format": "coffee_detector.sni21_r0_repaste_summary.v1",
        "protocol": PROTOCOL,
        "setup": str(benchmark_root / CONDITION / "repaste_setup.json"),
        "report": str(report_path),
        "table": str(table_path),
        "rows": rows,
        "attribution": attribution,
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
    }
    summary_path = evaluation_root / "r0_repaste_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== R0 REPASTE CONTROL ===", flush=True)
    print(f"COVERAGE     : {setup['coverage']:.2%}", flush=True)
    print(f"mAP RETENTION: {map_retention:.2%}", flush=True)
    print(f"CLS RETENTION: {class_retention:.2%}", flush=True)
    print(f"INTERPRETASI : {attribution['interpretation']}", flush=True)
    print("TRAINING     : False", flush=True)
    print("TEST ACCESS  : False", flush=True)
    print("SUMMARY      :", summary_path, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-training R0 original-position repaste control."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--source-benchmark-root", required=True)
    parser.add_argument("--density-evaluation-root", required=True)
    parser.add_argument("--benchmark-output-root", required=True)
    parser.add_argument("--evaluation-output-root", required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.85)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--diagnostic-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    run_sni21_r0_repaste_control(
        args.checkpoint,
        args.real_root,
        args.source_benchmark_root,
        args.density_evaluation_root,
        args.benchmark_output_root,
        args.evaluation_output_root,
        minimum_coverage=args.minimum_coverage,
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
