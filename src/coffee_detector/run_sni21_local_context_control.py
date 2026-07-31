from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

from .dataset import IMAGE_SUFFIXES, discover_layout
from .prepare_sni_fullscene import SNI21_CLASSES
from .run_sni21_density_evaluation import (
    _diagnose_dataset,
    _metric_summary,
    _sha256,
    _write_runtime_dataset_yaml,
)
from .run_sni21_r0_repaste_control import (
    _dataset_and_parent,
    _pixel_boxes,
    match_assets_to_boxes,
)
from .vadcp.compositor import load_background
from .vadcp.profile import load_scene_calibration


PROTOCOL = "docs/SNI21_LOCAL_CONTEXT_CONTROL_PROTOCOL.md"
ARMS = (
    "LC0_original_context",
    "LC1_repaste_real_context",
    "LC2_repaste_procedural_context",
)


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} bukan JSON object: {path}")
    return payload


def _intersection_area(left: list[float], right: list[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def isolated_context_box(
    target: list[float],
    all_boxes: list[list[float]],
    target_index: int,
    image_size: tuple[int, int],
    *,
    context_multiplier: float,
) -> list[int] | None:
    if context_multiplier <= 1.0:
        raise ValueError("context_multiplier harus > 1")
    width, height = image_size
    x1, y1, x2, y2 = (float(value) for value in target)
    side = context_multiplier * max(x2 - x1, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    right = int(round(center_x + side / 2.0))
    bottom = int(round(center_y + side / 2.0))
    # Do not pad or shift edge objects: that would introduce a second context
    # intervention and move the target away from the paired center.
    if left < 0 or top < 0 or right > width or bottom > height:
        return None
    patch = [float(left), float(top), float(right), float(bottom)]
    for index, other in enumerate(all_boxes):
        if index != target_index and _intersection_area(patch, other) > 0:
            return None
    return [left, top, right, bottom]


def _stable_score(seed: int, asset_id: str) -> str:
    return hashlib.sha256(f"{seed}:{asset_id}".encode("utf-8")).hexdigest()


def _paste_cutout(
    background: Image.Image,
    asset_path: Path,
    local_box: list[int],
) -> Image.Image:
    image = background.convert("RGB")
    left, top, right, bottom = local_box
    with Image.open(asset_path) as source:
        cutout = source.convert("RGBA").resize(
            (right - left, bottom - top), Image.Resampling.LANCZOS
        )
    image.paste(cutout, (left, top), cutout)
    return image


def _write_arm_layout(root: Path) -> None:
    names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    for kind in ("images", "labels"):
        (root / "train" / kind).mkdir(parents=True, exist_ok=True)
        (root / "val" / kind).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def prepare_local_context_dataset(
    real_root: str | Path,
    object_library_root: str | Path,
    scene_profile: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    max_per_class: int = 20,
    context_multiplier: float = 3.0,
    minimum_samples: int = 150,
    minimum_classes: int = 15,
) -> dict:
    if max_per_class <= 0:
        raise ValueError("max_per_class harus positif")
    real_root = Path(real_root).expanduser().resolve()
    library_root = Path(object_library_root).expanduser().resolve()
    scene_profile = Path(scene_profile).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    setup_path = output_root / "local_context_setup.json"
    if setup_path.is_file():
        existing = _read_json(setup_path, "Setup local-context")
        if existing.get("complete") is not True:
            raise RuntimeError("Setup local-context lama tidak lengkap")
        return existing
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Output local-context parsial: {output_root}")

    layout = discover_layout(real_root)
    expected_names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    if layout.names != expected_names or "val" not in layout.splits:
        raise RuntimeError("R0 validation bukan canonical SNI-21")
    library = _read_json(library_root / "object_library.json", "Object library")
    library_names = {int(key): str(value) for key, value in library["classes"].items()}
    if library_names != expected_names:
        raise RuntimeError("Object library bukan canonical SNI-21")
    assets_by_parent: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for asset in library["assets"]:
        if str(asset.get("source_split")) != "val":
            raise RuntimeError("Object library memuat aset non-validation")
        assets_by_parent[
            (str(asset["source_dataset"]), str(asset["source_parent_id"]))
        ].append(asset)
    calibration = load_scene_calibration(scene_profile)

    image_root, label_root = layout.splits["val"]
    images = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    candidates: dict[int, list[dict]] = defaultdict(list)
    for image_path in images:
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        with Image.open(image_path) as source:
            image_size = source.size
        boxes = _pixel_boxes(label_path, *image_size)
        dataset, parent_id = _dataset_and_parent(image_path)
        pairs = match_assets_to_boxes(
            boxes, assets_by_parent.get((dataset, parent_id), [])
        )
        all_xyxy = [row["xyxy"] for row in boxes]
        for box, asset in pairs:
            patch = isolated_context_box(
                box["xyxy"],
                all_xyxy,
                int(box["index"]),
                image_size,
                context_multiplier=context_multiplier,
            )
            if patch is None:
                continue
            candidates[int(box["class_id"])].append(
                {
                    "image": image_path,
                    "relative": relative,
                    "box": box,
                    "asset": asset,
                    "patch": patch,
                    "score": _stable_score(seed, str(asset["asset_id"])),
                }
            )

    selected = []
    for class_id, rows in sorted(candidates.items()):
        selected.extend(sorted(rows, key=lambda row: row["score"])[:max_per_class])
    selected.sort(key=lambda row: (int(row["box"]["class_id"]), row["score"]))
    selected_classes = sorted({int(row["box"]["class_id"]) for row in selected})
    if len(selected) < minimum_samples or len(selected_classes) < minimum_classes:
        raise RuntimeError(
            f"Kandidat local-context tidak cukup: samples={len(selected)}, "
            f"classes={len(selected_classes)}"
        )

    for arm in ARMS:
        _write_arm_layout(output_root / arm)
    manifest = []
    print(f"LOCAL-CONTEXT: mulai 0/{len(selected)} objek", flush=True)
    for sample_index, row in enumerate(selected, 1):
        image_path = Path(row["image"])
        left, top, right, bottom = row["patch"]
        with Image.open(image_path) as source:
            original = source.convert("RGB").crop((left, top, right, bottom))
        x1, y1, x2, y2 = row["box"]["xyxy"]
        local_box = [
            max(0, int(round(x1 - left))),
            max(0, int(round(y1 - top))),
            min(original.width, int(round(x2 - left))),
            min(original.height, int(round(y2 - top))),
        ]
        asset_path = library_root / str(row["asset"]["image"])
        real_repaste = _paste_cutout(original, asset_path, local_box)
        background_rng = random.Random(seed * 1_000_003 + sample_index)
        procedural = load_background(
            None, original.size, background_rng, calibration
        ).convert("RGB")
        procedural_repaste = _paste_cutout(procedural, asset_path, local_box)
        stem = f"lc_{sample_index:04d}_{row['asset']['asset_id']}"
        arms = {
            ARMS[0]: original,
            ARMS[1]: real_repaste,
            ARMS[2]: procedural_repaste,
        }
        for arm, image in arms.items():
            image.save(output_root / arm / "val" / "images" / f"{stem}.png")
            patch_width, patch_height = image.size
            box_width = local_box[2] - local_box[0]
            box_height = local_box[3] - local_box[1]
            x_center = (local_box[0] + box_width / 2.0) / patch_width
            y_center = (local_box[1] + box_height / 2.0) / patch_height
            label = (
                f"{int(row['box']['class_id'])} {x_center:.8f} {y_center:.8f} "
                f"{box_width / patch_width:.8f} {box_height / patch_height:.8f}\n"
            )
            (output_root / arm / "val" / "labels" / f"{stem}.txt").write_text(
                label, encoding="utf-8"
            )
        manifest.append(
            {
                "sample_id": stem,
                "class_id": int(row["box"]["class_id"]),
                "class_name": SNI21_CLASSES[int(row["box"]["class_id"])],
                "source_image": str(row["relative"]).replace("\\", "/"),
                "source_asset_id": str(row["asset"]["asset_id"]),
                "source_parent_id": row["asset"].get("source_parent_id"),
                "patch_xyxy": row["patch"],
                "object_xyxy_in_patch": local_box,
            }
        )
        if sample_index % 50 == 0 or sample_index == len(selected):
            print(f"  local-context {sample_index}/{len(selected)}", flush=True)

    class_counts = defaultdict(int)
    for row in manifest:
        class_counts[row["class_name"]] += 1
    payload = {
        "format": "coffee_detector.sni21_local_context_setup.v1",
        "protocol": PROTOCOL,
        "arms": {arm: str(output_root / arm) for arm in ARMS},
        "samples": len(manifest),
        "classes": len(class_counts),
        "samples_by_class": dict(sorted(class_counts.items())),
        "seed": seed,
        "max_per_class": max_per_class,
        "context_multiplier": context_multiplier,
        "selection": "isolated matched validation objects; stable class-balanced cap",
        "object_geometry": "identical cutout, bbox, and local position in LC1/LC2",
        "manifest": str(output_root / "local_context_manifest.json"),
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "complete": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "local_context_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    setup_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def classify_background_retention(map_retention: float, class_retention: float) -> str:
    lower = min(map_retention, class_retention)
    if lower >= 0.80:
        return "procedural_background_not_primary_cause"
    if lower >= 0.50:
        return "procedural_background_material_partial_cause"
    return "procedural_background_dominant_cause"


def _retention(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0:
        raise RuntimeError(f"Retensi {label} tidak terdefinisi: baseline <= 0")
    return numerator / denominator


def _report_row(condition: str, report: dict) -> dict:
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


def run_sni21_local_context_control(
    checkpoint: str | Path,
    real_root: str | Path,
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
    max_det: int = 100,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang") from error
    checkpoint = Path(checkpoint).expanduser().resolve()
    source_root = Path(source_benchmark_root).expanduser().resolve()
    density_root = Path(density_evaluation_root).expanduser().resolve()
    benchmark_root = Path(benchmark_output_root).expanduser().resolve()
    evaluation_root = Path(evaluation_output_root).expanduser().resolve()
    density_summary = _read_json(
        density_root / "density_evaluation_summary.json", "Summary density"
    )
    if density_summary.get("training_executed") is not False:
        raise RuntimeError("Summary density tidak menjamin training=False")
    if density_summary.get("test_images_accessed") is not False:
        raise RuntimeError("Summary density telah mengakses test")
    if _sha256(checkpoint) != density_summary["run_config"]["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint berbeda dari evaluasi A0 beku")

    print("[1/2] Membuat tiga arm local-context", flush=True)
    setup = prepare_local_context_dataset(
        real_root,
        source_root / "val_object_library",
        source_root / "val_scene_profile.json",
        benchmark_root,
    )
    model = YOLO(str(checkpoint))
    evaluation_root.mkdir(parents=True, exist_ok=True)
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
    reports = {}
    print("[2/2] Evaluasi tiga arm dengan checkpoint yang sama", flush=True)
    for arm in ARMS:
        arm_output = evaluation_root / arm
        report_path = arm_output / "evaluation.json"
        if report_path.is_file():
            report = _read_json(report_path, f"Evaluasi {arm}")
            if report.get("run_config") != run_config or report.get("complete") is not True:
                raise RuntimeError(f"Evaluasi lama konflik/parsial: {arm}")
            print(f"  reuse {arm}", flush=True)
        else:
            print(f"  validate {arm}", flush=True)
            layout = discover_layout(benchmark_root / arm)
            runtime_yaml = _write_runtime_dataset_yaml(
                layout, arm_output / "runtime_data.yaml"
            )
            kwargs = {
                "data": str(runtime_yaml), "split": "val", "imgsz": imgsz,
                "batch": batch_size, "conf": confidence, "iou": nms_iou,
                "max_det": max_det, "plots": False, "verbose": True,
                "project": str(arm_output / "ultralytics"),
                "name": "validation", "exist_ok": True,
            }
            if device is not None:
                kwargs["device"] = device
            metrics = model.val(**kwargs)
            diagnosis = _diagnose_dataset(
                model, benchmark_root / arm, arm_output,
                device=device, imgsz=imgsz, confidence=confidence,
                nms_iou=nms_iou, diagnostic_iou=diagnostic_iou,
                max_det=max_det, batch_size=batch_size,
            )
            report = {
                "format": "coffee_detector.sni21_local_context_evaluation.v1",
                "condition": arm, "run_config": run_config,
                "official_metrics": _metric_summary(metrics, layout.names),
                "diagnosis": diagnosis, "complete": True,
                "development_only": True,
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        reports[arm] = report

    rows = [_report_row(arm, reports[arm]) for arm in ARMS]
    by_arm = {row["condition"]: row for row in rows}
    original = by_arm[ARMS[0]]
    real_repaste = by_arm[ARMS[1]]
    procedural = by_arm[ARMS[2]]
    cutout_attribution = {
        "map50_95_retention": _retention(
            real_repaste["map50_95"], original["map50_95"], "cutout mAP"
        ),
        "conditional_class_accuracy_retention": _retention(
            real_repaste["conditional_class_accuracy"],
            original["conditional_class_accuracy"],
            "cutout class accuracy",
        ),
    }
    background_map_retention = _retention(
        procedural["map50_95"], real_repaste["map50_95"], "background mAP"
    )
    background_class_retention = _retention(
        procedural["conditional_class_accuracy"],
        real_repaste["conditional_class_accuracy"],
        "background class accuracy",
    )
    background_attribution = {
        "map50_95_retention": background_map_retention,
        "conditional_class_accuracy_retention": background_class_retention,
        "interpretation": classify_background_retention(
            background_map_retention, background_class_retention
        ),
    }
    table_path = evaluation_root / "local_context_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format": "coffee_detector.sni21_local_context_summary.v1",
        "protocol": PROTOCOL,
        "setup": str(benchmark_root / "local_context_setup.json"),
        "rows": rows,
        "cutout_attribution": cutout_attribution,
        "background_attribution": background_attribution,
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
    }
    summary_path = evaluation_root / "local_context_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== LOCAL-CONTEXT CONTROL ===", flush=True)
    print(f"SAMPLES       : {setup['samples']} | CLASSES: {setup['classes']}", flush=True)
    print("CUTOUT RETAIN :", cutout_attribution, flush=True)
    print("BACKGROUND    :", background_attribution, flush=True)
    print("TRAINING      : False", flush=True)
    print("TEST ACCESS   : False", flush=True)
    print("SUMMARY       :", summary_path, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired SNI-21 local-context control")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-root", required=True)
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
    run_sni21_local_context_control(
        args.checkpoint, args.real_root, args.source_benchmark_root,
        args.density_evaluation_root, args.benchmark_output_root,
        args.evaluation_output_root, device=args.device, imgsz=args.imgsz,
        batch_size=args.batch_size, confidence=args.confidence,
        nms_iou=args.nms_iou, diagnostic_iou=args.diagnostic_iou,
        max_det=args.max_det,
    )


if __name__ == "__main__":
    main()
