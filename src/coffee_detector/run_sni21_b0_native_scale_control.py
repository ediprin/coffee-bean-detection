from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .audit_vadcp import audit_vadcp_dataset
from .dataset import discover_layout
from .generate_vadcp_dataset import generate_vadcp_dataset
from .prepare_sni_fullscene import SNI21_CLASSES
from .run_sni21_density_benchmark_setup import _write_resampling_units
from .run_sni21_density_evaluation import (
    _diagnose_dataset,
    _metric_summary,
    _sha256,
    _write_runtime_dataset_yaml,
)
from .vadcp.library import load_object_library
from .vadcp.profile import load_scene_calibration


ORIGINAL_CONDITION = "B0_empirical_mild"
CONTROL_CONDITION = "B0_empirical_mild_native_scale"
PROTOCOL = "docs/SNI21_B0_NATIVE_SCALE_CONTROL_PROTOCOL.md"


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} bukan JSON object: {path}")
    return payload


def derive_native_scale(
    records_path: str | Path,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> dict:
    """Derive real-validation object long-side fractions from GT records."""

    records_path = Path(records_path).expanduser().resolve()
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Quantile scale tidak valid")
    if not records_path.is_file():
        raise FileNotFoundError(f"R0 prediction records tidak ditemukan: {records_path}")
    values: list[float] = []
    images = 0
    with records_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            width = int(row["width"])
            height = int(row["height"])
            if width <= 0 or height <= 0:
                raise ValueError(f"Ukuran R0 tidak valid pada baris {line_number}")
            images += 1
            canvas_long = max(width, height)
            for item in row.get("ground_truth", []):
                x1, y1, x2, y2 = (float(value) for value in item["xyxy"])
                long_side = max(x2 - x1, y2 - y1)
                if long_side <= 0:
                    raise ValueError(
                        f"Ground-truth box R0 tidak valid pada baris {line_number}"
                    )
                values.append(long_side / canvas_long)
    if not values:
        raise RuntimeError("R0 prediction records tidak memiliki ground truth")
    array = np.asarray(values, dtype=np.float64)
    quantile_points = (0.01, lower_quantile, 0.25, 0.50, 0.75, upper_quantile, 0.99)
    quantiles = np.quantile(array, quantile_points)
    names = ("q01", "q05", "q25", "q50", "q75", "q95", "q99")
    return {
        "records": str(records_path),
        "images": images,
        "boxes": len(values),
        "normalization": "box_long_side / image_long_side",
        "quantiles": {
            name: float(value) for name, value in zip(names, quantiles)
        },
        "selected_interval": [
            float(np.quantile(array, lower_quantile)),
            float(np.quantile(array, upper_quantile)),
        ],
    }


def _scene_draws(metadata_path: Path) -> dict[str, list[tuple[int, str, str | None]]]:
    payload = _read_json(metadata_path, "Metadata scene")
    images = {int(row["id"]): str(row["generation_seed"]) for row in payload["images"]}
    rows: dict[str, list[tuple[int, str, str | None]]] = {
        scene_id: [] for scene_id in images.values()
    }
    for annotation in payload["annotations"]:
        scene_id = images[int(annotation["image_id"])]
        rows[scene_id].append(
            (
                int(annotation["category_id"]),
                str(annotation["source_asset_id"]),
                (
                    str(annotation["source_parent_id"])
                    if annotation.get("source_parent_id") is not None
                    else None
                ),
            )
        )
    return rows


def audit_paired_scene_draws(original_metadata: Path, control_metadata: Path) -> dict:
    original = _scene_draws(original_metadata)
    control = _scene_draws(control_metadata)
    scene_ids_equal = set(original) == set(control)
    mismatches = [
        scene_id
        for scene_id in sorted(set(original) | set(control))
        if original.get(scene_id) != control.get(scene_id)
    ]
    return {
        "original_metadata": str(original_metadata),
        "control_metadata": str(control_metadata),
        "original_scenes": len(original),
        "control_scenes": len(control),
        "scene_ids_equal": scene_ids_equal,
        "exact_draw_match": scene_ids_equal and not mismatches,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }


def classify_scale_recovery(recovery_fraction: float) -> str:
    if recovery_fraction >= 0.50:
        return "scale_explains_majority"
    if recovery_fraction >= 0.20:
        return "scale_is_material_partial_cause"
    return "scale_alone_does_not_explain_collapse"


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
        "proposal_miss_rate": diagnosis["proposal_miss_rate"],
        "localized_wrong_class_rate": diagnosis["localized_wrong_class_rate"],
        "saturation_rate": diagnosis["saturation_rate"],
        "count_mae": diagnosis["count_mae"],
        "count_bias": diagnosis["count_bias"],
    }


def _safe_remove_partial(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    resolved_parent = parent.resolve()
    if resolved.parent != resolved_parent:
        raise RuntimeError(f"Menolak menghapus path di luar output root: {resolved}")
    shutil.rmtree(resolved)


def run_sni21_b0_native_scale_control(
    checkpoint: str | Path,
    source_benchmark_root: str | Path,
    density_evaluation_root: str | Path,
    benchmark_output_root: str | Path,
    evaluation_output_root: str | Path,
    *,
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
        raise RuntimeError("Ultralytics belum terpasang. Jalankan `pip install -e .`.") from error

    checkpoint = Path(checkpoint).expanduser().resolve()
    source_root = Path(source_benchmark_root).expanduser().resolve()
    old_evaluation_root = Path(density_evaluation_root).expanduser().resolve()
    benchmark_output_root = Path(benchmark_output_root).expanduser().resolve()
    evaluation_output_root = Path(evaluation_output_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")

    setup = _read_json(source_root / "setup_core_summary.json", "Setup density")
    old_summary = _read_json(
        old_evaluation_root / "density_evaluation_summary.json",
        "Evaluasi density lama",
    )
    for payload, label in ((setup, "setup"), (old_summary, "evaluasi")):
        if payload.get("training_executed") is not False:
            raise RuntimeError(f"{label} tidak menjamin training=False")
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"{label} telah mengakses test")
    if _sha256(checkpoint) != old_summary["run_config"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint berbeda dari evaluasi density yang sudah selesai")

    old_rows = {row["condition"]: row for row in old_summary["rows"]}
    required_rows = {"R0_real_val", ORIGINAL_CONDITION}
    if not required_rows <= set(old_rows):
        raise RuntimeError("Evaluasi lama belum memiliki R0 dan B0")
    r0_records = old_evaluation_root / "R0_real_val" / "prediction_records.jsonl"
    scale_audit = derive_native_scale(r0_records)
    scale_min, scale_max = scale_audit["selected_interval"]

    original_manifest = _read_json(
        source_root / ORIGINAL_CONDITION / "metadata" / "generation_manifest.json",
        "Manifest B0 original",
    )
    original_spec = original_manifest["spec"]
    expected = {
        "artifact_role": "development_benchmark",
        "library_source_split": "val",
        "synthetic_split": "val",
        "mode": "visibility",
        "preset": "sni_spread",
    }
    for key, value in expected.items():
        if original_manifest.get(key) != value:
            raise RuntimeError(f"B0 original tidak sesuai protokol: {key}")
    if original_spec.get("object_range") != [1, 5]:
        raise RuntimeError("B0 original bukan density 1--5")
    if original_spec.get("target_bin_weights", {}).get("mild") != 1.0:
        raise RuntimeError("B0 original bukan target visibility mild")
    if bool(original_spec.get("class_balanced")):
        raise RuntimeError("B0 original bukan empirical source prior")

    library_root = source_root / "val_object_library"
    scene_profile = source_root / "val_scene_profile.json"
    names, _cutouts, _library_info = load_object_library(
        library_root, train_only=False
    )
    canonical = {index: name for index, name in enumerate(SNI21_CLASSES)}
    if names != canonical:
        raise RuntimeError("Object library bukan canonical SNI-21")
    calibration = load_scene_calibration(scene_profile)

    benchmark_output_root.mkdir(parents=True, exist_ok=True)
    arm_root = benchmark_output_root / CONTROL_CONDITION
    manifest_path = arm_root / "metadata" / "generation_manifest.json"
    generation_config = {
        "synthetic_images": int(original_manifest["synthetic_images"]),
        "seed": int(original_manifest["seed"]),
        "canvas_size": int(max(original_spec["canvas_size"])),
        "object_range": [1, 5],
        "object_scale": [scale_min, scale_max],
        "asset_reuse_limit": int(original_manifest["asset_reuse"]["limit"]),
        "parent_reuse_limit": int(original_manifest["parent_reuse"]["limit"]),
    }
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, "Manifest B0 native-scale")
        actual_config = {
            "synthetic_images": int(manifest["synthetic_images"]),
            "seed": int(manifest["seed"]),
            "canvas_size": int(max(manifest["spec"]["canvas_size"])),
            "object_range": manifest["spec"]["object_range"],
            "object_scale": manifest["spec"]["object_scale"],
            "asset_reuse_limit": int(manifest["asset_reuse"]["limit"]),
            "parent_reuse_limit": int(manifest["parent_reuse"]["limit"]),
        }
        if actual_config != generation_config:
            raise RuntimeError("Output B0 native-scale lama memiliki konfigurasi berbeda")
        print("[1/3] Reuse dataset B0 native-scale", flush=True)
    else:
        if arm_root.exists() and any(arm_root.iterdir()):
            print("[1/3] Hapus output B0 native-scale parsial", flush=True)
            _safe_remove_partial(arm_root, benchmark_output_root)
        print(
            f"[1/3] Generate B0 native-scale {scale_min:.4f}--{scale_max:.4f}",
            flush=True,
        )
        generate_vadcp_dataset(
            None,
            library_root,
            arm_root,
            synthetic_images=generation_config["synthetic_images"],
            seed=generation_config["seed"],
            mode="visibility",
            preset="sni_spread",
            canvas_size=generation_config["canvas_size"],
            object_range=(1, 5),
            object_scale=(scale_min, scale_max),
            include_real_train=False,
            materialize_real_splits=False,
            scene_profile=calibration,
            target_names=names,
            artifact_role="development_benchmark",
            library_source_split="val",
            synthetic_split="val",
            class_balanced=False,
            target_visibility_bin="mild",
            max_asset_reuse=generation_config["asset_reuse_limit"],
            max_parent_reuse=generation_config["parent_reuse_limit"],
        )

    print("[2/3] Audit geometry, provenance, dan paired draw", flush=True)
    geometry_audit = audit_vadcp_dataset(
        arm_root, arm_root / "metadata" / "vadcp_audit.json"
    )
    if not geometry_audit["safe_for_training"]:
        raise RuntimeError("B0 native-scale gagal audit VA-DCP")
    resampling = _write_resampling_units(arm_root)
    original_metadata = (
        source_root
        / ORIGINAL_CONDITION
        / "metadata"
        / "instances_synthetic_val.json"
    )
    control_metadata = arm_root / "metadata" / "instances_synthetic_val.json"
    pairing = audit_paired_scene_draws(original_metadata, control_metadata)
    if not pairing["exact_draw_match"]:
        raise RuntimeError("B0 original/native-scale tidak memiliki paired draw identik")

    setup_report = {
        "format": "coffee_detector.sni21_b0_native_scale_setup.v1",
        "protocol": PROTOCOL,
        "source_benchmark": str(source_root),
        "source_evaluation": str(old_evaluation_root),
        "control_root": str(arm_root),
        "r0_scale_audit": scale_audit,
        "generation_config": generation_config,
        "pairing_audit": pairing,
        "geometry_audit": str(arm_root / "metadata" / "vadcp_audit.json"),
        "resampling_units": resampling,
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "ready_for_evaluation": True,
    }
    setup_path = benchmark_output_root / "b0_native_scale_setup.json"
    setup_path.write_text(
        json.dumps(setup_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    evaluation_output_root.mkdir(parents=True, exist_ok=True)
    condition_root = evaluation_output_root / CONTROL_CONDITION
    report_path = condition_root / "evaluation.json"
    run_config = {
        # Use content identity rather than a Drive mount path so the report is
        # reusable after switching Colab accounts.
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
        report = _read_json(report_path, "Evaluasi B0 native-scale")
        if report.get("run_config") != run_config or report.get("complete") is not True:
            raise RuntimeError("Evaluasi B0 native-scale lama konflik atau parsial")
        print("[3/3] Reuse evaluasi B0 native-scale", flush=True)
    else:
        print("[3/3] Validate dan diagnose B0 native-scale", flush=True)
        model = YOLO(str(checkpoint))
        layout = discover_layout(arm_root)
        runtime_yaml = _write_runtime_dataset_yaml(
            layout, condition_root / "runtime_data.yaml"
        )
        validation_kwargs = {
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
            validation_kwargs["device"] = device
        metrics = model.val(**validation_kwargs)
        diagnosis = _diagnose_dataset(
            model,
            arm_root,
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
            "format": "coffee_detector.sni21_b0_native_scale_evaluation.v1",
            "condition": CONTROL_CONDITION,
            "data_root": str(arm_root),
            "run_config": run_config,
            "official_metrics": _metric_summary(metrics, layout.names),
            "diagnosis": diagnosis,
            "resampling_units": resampling,
            "development_only": True,
            "complete": True,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    rows = [
        old_rows["R0_real_val"],
        old_rows[ORIGINAL_CONDITION],
        _row_from_report(CONTROL_CONDITION, report),
    ]
    original_map = float(old_rows[ORIGINAL_CONDITION]["map50_95"])
    native_map = float(rows[-1]["map50_95"])
    real_map = float(old_rows["R0_real_val"]["map50_95"])
    denominator = real_map - original_map
    recovery = (native_map - original_map) / denominator if denominator > 0 else 0.0
    attribution = {
        "metric": "mAP50-95",
        "r0": real_map,
        "b0_original": original_map,
        "b0_native_scale": native_map,
        "native_minus_original": native_map - original_map,
        "remaining_gap_to_r0": real_map - native_map,
        "recovery_fraction": recovery,
        "interpretation": classify_scale_recovery(recovery),
    }
    table_path = evaluation_output_root / "b0_native_scale_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format": "coffee_detector.sni21_b0_native_scale_summary.v1",
        "protocol": PROTOCOL,
        "setup": str(setup_path),
        "report": str(report_path),
        "table": str(table_path),
        "rows": rows,
        "scale_attribution": attribution,
        "paired_draws": pairing,
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
    }
    summary_path = evaluation_output_root / "b0_native_scale_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== B0 NATIVE-SCALE CONTROL COMPLETE ===", flush=True)
    print(f"SCALE       : {scale_min:.4f}--{scale_max:.4f}", flush=True)
    print(f"PAIRED DRAW : {pairing['exact_draw_match']}", flush=True)
    print(f"RECOVERY    : {recovery:.1%}", flush=True)
    print(f"INTERPRETASI: {attribution['interpretation']}", flush=True)
    print("TRAINING    : False", flush=True)
    print("TEST ACCESS : False", flush=True)
    print("SUMMARY     :", summary_path, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate only the paired B0 native-scale control."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-benchmark-root", required=True)
    parser.add_argument("--density-evaluation-root", required=True)
    parser.add_argument("--benchmark-output-root", required=True)
    parser.add_argument("--evaluation-output-root", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--diagnostic-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    run_sni21_b0_native_scale_control(
        args.checkpoint,
        args.source_benchmark_root,
        args.density_evaluation_root,
        args.benchmark_output_root,
        args.evaluation_output_root,
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
