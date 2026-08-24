from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .analyze_sni21_fullframe_context import (
    _exact_mcnemar,
    _read_jsonl,
    corrected_record_diagnosis,
)
from .dataset import discover_layout
from .prepare_sni_fullscene import SNI21_CLASSES
from .run_sni21_density_evaluation import (
    _diagnose_dataset,
    _metric_summary,
    _pairwise_iou,
    _sha256,
    _write_runtime_dataset_yaml,
)
from .run_sni21_local_context_control import _paste_cutout


CONDITION = "FC3_repaste_empirical_fullframe"
PROTOCOL = "docs/SNI21_EMPIRICAL_BACKGROUND_CONTROL_PROTOCOL.md"


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} bukan JSON object: {path}")
    return payload


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _write_layout(root: Path) -> None:
    for kind in ("images", "labels"):
        (root / "train" / kind).mkdir(parents=True, exist_ok=True)
        (root / "val" / kind).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "names": {
                    index: name for index, name in enumerate(SNI21_CLASSES)
                },
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _expanded_box(box: list[int], size: tuple[int, int], ratio: float) -> list[int]:
    width, height = size
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = (x2 - x1) * ratio
    pad_y = (y2 - y1) * ratio
    return [
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    ]


def _inpaint_object(image: Image.Image, box: list[int]) -> Image.Image:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("OpenCV dari dependensi Ultralytics tidak tersedia") from error
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    left, top, right, bottom = _expanded_box(box, image.size, 0.12)
    roi_left, roi_top, roi_right, roi_bottom = _expanded_box(
        [left, top, right, bottom], image.size, 0.60
    )
    roi = rgb[roi_top:roi_bottom, roi_left:roi_right]
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    mask[
        top - roi_top : bottom - roi_top,
        left - roi_left : right - roi_left,
    ] = 255
    bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    radius = max(3.0, min(right - left, bottom - top) * 0.04)
    restored = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    output = rgb.copy()
    output[roi_top:roi_bottom, roi_left:roi_right] = cv2.cvtColor(
        restored, cv2.COLOR_BGR2RGB
    )
    return Image.fromarray(output)


def _donor_pairs(rows: list[dict], seed: int) -> list[tuple[dict, dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(int(value) for value in row["canvas_size"])].append(row)
    pairs = []
    for size, group in sorted(grouped.items()):
        if len(group) < 2:
            raise RuntimeError(f"Canvas group tidak dapat di-derange: {size}")
        ordered = sorted(
            group,
            key=lambda row: _stable_key(seed, str(row["source_asset_id"])),
        )
        donors = ordered[1:] + ordered[:1]
        pairs.extend(zip(ordered, donors))
    return sorted(pairs, key=lambda pair: str(pair[0]["sample_id"]))


def prepare_empirical_background_dataset(
    real_root: str | Path,
    object_library_root: str | Path,
    fullframe_benchmark_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
) -> dict:
    real_root = Path(real_root).expanduser().resolve()
    library_root = Path(object_library_root).expanduser().resolve()
    fullframe_root = Path(fullframe_benchmark_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    setup_path = output_root / "empirical_background_setup.json"
    if setup_path.is_file():
        setup = _read_json(setup_path, "Setup empirical background")
        if setup.get("complete") is not True:
            raise RuntimeError("Setup empirical background lama tidak lengkap")
        return setup
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Output empirical background parsial: {output_root}")

    manifest_path = fullframe_root / "local_context_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest full-frame tidak ditemukan: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("Manifest full-frame harus berupa list")
    layout = discover_layout(real_root)
    if layout.names != {index: name for index, name in enumerate(SNI21_CLASSES)}:
        raise RuntimeError("R0 bukan canonical SNI-21")
    image_root, _ = layout.splits["val"]
    library = _read_json(library_root / "object_library.json", "Object library")
    assets = {str(row["asset_id"]): row for row in library["assets"]}
    pairs = _donor_pairs(manifest, seed)
    _write_layout(output_root)

    rows = []
    bbox_overlaps = []
    print(f"EMPIRICAL BACKGROUND: mulai 0/{len(pairs)}", flush=True)
    for index, (target, donor) in enumerate(pairs, 1):
        target_asset = assets[str(target["source_asset_id"])]
        donor_path = image_root / str(donor["source_image"])
        if not donor_path.is_file():
            raise FileNotFoundError(donor_path)
        with Image.open(donor_path) as source:
            donor_image = source.convert("RGB")
        if list(donor_image.size) != list(target["canvas_size"]):
            raise RuntimeError("Donor dan target canvas berbeda")
        donor_clean = _inpaint_object(donor_image, donor["object_xyxy"])
        composed = _paste_cutout(
            donor_clean,
            library_root / str(target_asset["image"]),
            target["object_xyxy"],
        )
        sample_id = str(target["sample_id"])
        composed.save(
            output_root / "val" / "images" / f"{sample_id}.jpg",
            quality=95,
            subsampling=0,
        )
        left, top, right, bottom = target["object_xyxy"]
        width, height = composed.size
        label = (
            f"{int(target['class_id'])} "
            f"{((left + right) / 2) / width:.8f} "
            f"{((top + bottom) / 2) / height:.8f} "
            f"{(right - left) / width:.8f} "
            f"{(bottom - top) / height:.8f}\n"
        )
        (output_root / "val" / "labels" / f"{sample_id}.txt").write_text(
            label, encoding="utf-8"
        )
        overlap = float(
            _pairwise_iou(
                np.asarray([target["object_xyxy"]], dtype=np.float64),
                np.asarray([donor["object_xyxy"]], dtype=np.float64),
            )[0, 0]
        )
        bbox_overlaps.append(overlap)
        rows.append(
            {
                "sample_id": sample_id,
                "class_id": int(target["class_id"]),
                "target_asset_id": str(target["source_asset_id"]),
                "target_source_image": str(target["source_image"]),
                "donor_asset_id": str(donor["source_asset_id"]),
                "donor_source_image": str(donor["source_image"]),
                "target_donor_bbox_iou": overlap,
            }
        )
        if index % 25 == 0 or index == len(pairs):
            print(f"  empirical background {index}/{len(pairs)}", flush=True)

    manifest_path = output_root / "empirical_background_manifest.json"
    manifest_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    setup = {
        "format": "coffee_detector.sni21_empirical_background_setup.v1",
        "protocol": PROTOCOL,
        "condition": CONDITION,
        "samples": len(rows),
        "classes": len({row["class_id"] for row in rows}),
        "donor_policy": "within-canvas deterministic derangement; each donor once",
        "inpainting": "expanded_bbox_12_percent_opencv_telea",
        "median_target_donor_bbox_iou": float(np.median(bbox_overlaps)),
        "manifest": str(manifest_path),
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "complete": True,
    }
    setup_path.write_text(
        json.dumps(setup, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return setup


def _record_pairs(root: Path, left_arm: str, right_arm: str, iou: float) -> list[dict]:
    left = _read_jsonl(root / left_arm / "prediction_records.jsonl")
    right = _read_jsonl(root / right_arm / "prediction_records.jsonl")
    if set(left) != set(right):
        raise RuntimeError(f"Identity berbeda: {left_arm} vs {right_arm}")
    rows = []
    for image in sorted(left):
        before = corrected_record_diagnosis(left[image], iou)
        after = corrected_record_diagnosis(right[image], iou)
        if before["class_id"] != after["class_id"]:
            raise RuntimeError(f"Ground truth berbeda: {image}")
        rows.append(
            {
                "image": image,
                "class_id": before["class_id"],
                "before_correct": before["correct"],
                "after_correct": after["correct"],
                "before_proposal": before["proposal"],
                "after_proposal": after["proposal"],
                "before_state": before["state"],
                "after_state": after["state"],
            }
        )
    return rows


def _macro(rows: list[dict], field: str) -> float:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_id"])].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _paired_bootstrap(rows: list[dict], iterations: int, seed: int) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_id"])].append(row)
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    groups = list(grouped.values())
    for iteration in range(iterations):
        class_deltas = []
        for group in groups:
            indices = rng.integers(0, len(group), size=len(group))
            class_deltas.append(
                np.mean(
                    [
                        group[int(index)]["after_correct"]
                        - group[int(index)]["before_correct"]
                        for index in indices
                    ]
                )
            )
        deltas[iteration] = float(np.mean(class_deltas))
    point = _macro(rows, "after_correct") - _macro(rows, "before_correct")
    return {
        "point": point,
        "ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "probability_above_zero": float(np.mean(deltas > 0)),
        "iterations": iterations,
        "seed": seed,
    }


def run_empirical_background_control(
    checkpoint: str | Path,
    real_root: str | Path,
    object_library_root: str | Path,
    fullframe_benchmark_root: str | Path,
    fullframe_evaluation_root: str | Path,
    benchmark_output_root: str | Path,
    *,
    device: str | None = "0",
    imgsz: int = 640,
    batch_size: int = 8,
    confidence: float = 0.001,
    nms_iou: float = 0.7,
    diagnostic_iou: float = 0.5,
    max_det: int = 300,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang") from error
    checkpoint = Path(checkpoint).expanduser().resolve()
    evaluation_root = Path(fullframe_evaluation_root).expanduser().resolve()
    benchmark_root = Path(benchmark_output_root).expanduser().resolve()
    context_summary = _read_json(
        evaluation_root / "local_context_summary.json", "Summary full-frame"
    )
    if context_summary.get("training_executed") is not False or context_summary.get("test_images_accessed") is not False:
        raise RuntimeError("Summary full-frame tidak aman")
    source_report = _read_json(
        evaluation_root / "FC1_repaste_real_fullframe" / "evaluation.json",
        "FC1 evaluation",
    )
    procedural_report = _read_json(
        evaluation_root / "FC2_repaste_procedural_fullframe" / "evaluation.json",
        "FC2 evaluation",
    )
    expected_sha = source_report["run_config"]["checkpoint_sha256"]
    if _sha256(checkpoint) != expected_sha:
        raise RuntimeError("Checkpoint berbeda dari kontrol full-frame")
    requested_config = {
        "imgsz": imgsz,
        "batch_size": batch_size,
        "confidence": confidence,
        "nms_iou": nms_iou,
        "diagnostic_iou": diagnostic_iou,
        "max_det": max_det,
    }
    for key, value in requested_config.items():
        if source_report["run_config"].get(key) != value:
            raise RuntimeError(
                f"Konfigurasi {key} berbeda dari kontrol full-frame: "
                f"{value} != {source_report['run_config'].get(key)}"
            )

    setup = prepare_empirical_background_dataset(
        real_root,
        object_library_root,
        fullframe_benchmark_root,
        benchmark_root / CONDITION,
        seed=seed,
    )
    report_root = evaluation_root / CONDITION
    report_path = report_root / "evaluation.json"
    run_config = dict(source_report["run_config"])
    if report_path.is_file():
        empirical_report = _read_json(report_path, "FC3 evaluation")
        if empirical_report.get("run_config") != run_config or empirical_report.get("complete") is not True:
            raise RuntimeError("Evaluasi FC3 lama konflik/parsial")
        print("Reuse evaluasi FC3", flush=True)
    else:
        model = YOLO(str(checkpoint))
        layout = discover_layout(benchmark_root / CONDITION)
        runtime_yaml = _write_runtime_dataset_yaml(
            layout, report_root / "runtime_data.yaml"
        )
        kwargs = {
            "data": str(runtime_yaml), "split": "val", "imgsz": imgsz,
            "batch": batch_size, "conf": confidence, "iou": nms_iou,
            "max_det": max_det, "plots": False, "verbose": True,
            "project": str(report_root / "ultralytics"),
            "name": "validation", "exist_ok": True,
        }
        if device is not None:
            kwargs["device"] = device
        metrics = model.val(**kwargs)
        diagnosis = _diagnose_dataset(
            model, benchmark_root / CONDITION, report_root,
            device=device, imgsz=imgsz, confidence=confidence,
            nms_iou=nms_iou, diagnostic_iou=diagnostic_iou,
            max_det=max_det, batch_size=batch_size,
        )
        empirical_report = {
            "format": "coffee_detector.sni21_empirical_background_evaluation.v1",
            "condition": CONDITION,
            "run_config": run_config,
            "official_metrics": _metric_summary(metrics, layout.names),
            "diagnosis": diagnosis,
            "complete": True,
            "development_only": True,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(empirical_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    paired = _record_pairs(
        evaluation_root,
        "FC2_repaste_procedural_fullframe",
        CONDITION,
        diagnostic_iou,
    )
    bootstrap = _paired_bootstrap(paired, iterations, seed)
    harmed = sum(
        row["before_correct"] == 1 and row["after_correct"] == 0 for row in paired
    )
    rescued = sum(
        row["before_correct"] == 0 and row["after_correct"] == 1 for row in paired
    )
    mcnemar = _exact_mcnemar(harmed, rescued)
    fc1_map = source_report["official_metrics"]["metrics/mAP50-95(B)"]
    fc2_map = procedural_report["official_metrics"]["metrics/mAP50-95(B)"]
    fc3_map = empirical_report["official_metrics"]["metrics/mAP50-95(B)"]
    map_denominator = fc1_map - fc2_map
    map_recovery = (fc3_map - fc2_map) / map_denominator if map_denominator > 0 else None

    fc1_records = _record_pairs(
        evaluation_root,
        "FC2_repaste_procedural_fullframe",
        "FC1_repaste_real_fullframe",
        diagnostic_iou,
    )
    fc1_top1 = _macro(fc1_records, "after_correct")
    fc2_top1 = _macro(paired, "before_correct")
    fc3_top1 = _macro(paired, "after_correct")
    top1_denominator = fc1_top1 - fc2_top1
    top1_recovery = (
        (fc3_top1 - fc2_top1) / top1_denominator
        if top1_denominator > 0 else None
    )
    supported = (
        map_recovery is not None
        and map_recovery >= 0.50
        and bootstrap["ci95"][0] > 0
        and mcnemar["p_two_sided"] < 0.05
    )
    transitions = Counter(
        f"{row['before_state']}->{row['after_state']}" for row in paired
    )
    summary = {
        "format": "coffee_detector.sni21_empirical_background_summary.v1",
        "protocol": PROTOCOL,
        "analysis_status": "development_control",
        "setup": setup,
        "official_map50_95": {"fc1_real": fc1_map, "fc2_procedural": fc2_map, "fc3_empirical": fc3_map},
        "map50_95_recovery_fraction": map_recovery,
        "corrected_macro_top1": {"fc1_real": fc1_top1, "fc2_procedural": fc2_top1, "fc3_empirical": fc3_top1},
        "macro_top1_recovery_fraction": top1_recovery,
        "fc3_minus_fc2_bootstrap": bootstrap,
        "fc3_minus_fc2_mcnemar": mcnemar,
        "transitions": dict(sorted(transitions.items())),
        "empirical_background_priority_supported": supported,
        "interpretation": (
            "empirical_background_is_supported_simulator_priority"
            if supported else "empirical_background_recovery_not_confirmed"
        ),
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
    }
    summary_path = evaluation_root / "empirical_background_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== EMPIRICAL BACKGROUND CONTROL ===")
    print("mAP FC1/FC2/FC3:", summary["official_map50_95"])
    print("mAP recovery    :", map_recovery)
    print("Top1 FC1/FC2/FC3:", summary["corrected_macro_top1"])
    print("Top1 recovery   :", top1_recovery)
    print("Bootstrap       :", bootstrap)
    print("McNemar         :", mcnemar)
    print("SUPPORTED       :", supported)
    print("SAVED           :", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SNI-21 empirical donor-background control")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--object-library-root", required=True)
    parser.add_argument("--fullframe-benchmark-root", required=True)
    parser.add_argument("--fullframe-evaluation-root", required=True)
    parser.add_argument("--benchmark-output-root", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--diagnostic-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_empirical_background_control(
        args.checkpoint, args.real_root, args.object_library_root,
        args.fullframe_benchmark_root, args.fullframe_evaluation_root,
        args.benchmark_output_root, device=args.device, imgsz=args.imgsz,
        batch_size=args.batch_size, confidence=args.confidence,
        nms_iou=args.nms_iou, diagnostic_iou=args.diagnostic_iou,
        max_det=args.max_det, iterations=args.iterations, seed=args.seed,
    )


if __name__ == "__main__":
    main()
