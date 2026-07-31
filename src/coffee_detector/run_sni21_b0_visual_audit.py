from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .prepare_sni_fullscene import SNI21_CLASSES


ORIGINAL_CONDITION = "B0_empirical_mild"
CONTROL_CONDITION = "B0_empirical_mild_native_scale"


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} bukan JSON object: {path}")
    return payload


def _read_records(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction records tidak ditemukan: {path}")
    rows = {}
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            key = Path(str(row["image"])).name
            if key in rows:
                raise RuntimeError(f"Nama image record duplikat: {key}")
            rows[key] = row
    return rows


def _metadata_scenes(data_root: Path) -> dict[str, dict]:
    metadata = _read_json(
        data_root / "metadata" / "instances_synthetic_val.json",
        "Metadata synthetic val",
    )
    annotations: dict[int, list[dict]] = defaultdict(list)
    for row in metadata["annotations"]:
        annotations[int(row["image_id"])].append(row)
    scenes = {}
    for image in metadata["images"]:
        image_id = int(image["id"])
        scene_id = str(image["generation_seed"])
        if scene_id in scenes:
            raise RuntimeError(f"generation_seed duplikat: {scene_id}")
        scenes[scene_id] = {
            "image": image,
            "annotations": sorted(
                annotations[image_id], key=lambda item: int(item["z_order"])
            ),
        }
    return scenes


def _xywh_to_xyxy(box: list[int | float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def _validate_record_alignment(scene: dict, record: dict) -> dict[int, dict]:
    annotations = [
        row for row in scene["annotations"] if not int(row.get("ignore", 0))
    ]
    ground_truth = record["ground_truth"]
    diagnosis = record["ground_truth_diagnosis"]
    if not (len(annotations) == len(ground_truth) == len(diagnosis)):
        raise RuntimeError("Jumlah annotation/GT/diagnosis berbeda")
    diagnosis_by_z_order = {}
    for index, (annotation, gt, result) in enumerate(
        zip(annotations, ground_truth, diagnosis)
    ):
        if int(annotation["category_id"]) != int(gt["class_id"]):
            raise RuntimeError(f"Kelas annotation/record berbeda pada index {index}")
        if int(result["ground_truth_index"]) != index:
            raise RuntimeError("Urutan ground_truth_diagnosis tidak konsisten")
        expected = np.asarray(_xywh_to_xyxy(annotation["bbox"]), dtype=float)
        actual = np.asarray(gt["xyxy"], dtype=float)
        if not np.allclose(expected, actual, atol=0.02):
            raise RuntimeError(f"BBox annotation/record berbeda pada index {index}")
        diagnosis_by_z_order[int(annotation["z_order"])] = result
    return diagnosis_by_z_order


def build_paired_rows(
    original_data_root: str | Path,
    control_data_root: str | Path,
    original_records_path: str | Path,
    control_records_path: str | Path,
) -> list[dict]:
    original_root = Path(original_data_root).expanduser().resolve()
    control_root = Path(control_data_root).expanduser().resolve()
    original_scenes = _metadata_scenes(original_root)
    control_scenes = _metadata_scenes(control_root)
    if set(original_scenes) != set(control_scenes):
        raise RuntimeError("Scene ID B0 original/native-scale berbeda")
    original_records = _read_records(Path(original_records_path).resolve())
    control_records = _read_records(Path(control_records_path).resolve())
    rows = []
    for scene_id in sorted(original_scenes):
        original_scene = original_scenes[scene_id]
        control_scene = control_scenes[scene_id]
        original_name = Path(original_scene["image"]["file_name"]).name
        control_name = Path(control_scene["image"]["file_name"]).name
        original_record = original_records[original_name]
        control_record = control_records[control_name]
        original_diagnosis_by_z = _validate_record_alignment(
            original_scene, original_record
        )
        control_diagnosis_by_z = _validate_record_alignment(
            control_scene, control_record
        )
        if len(original_scene["annotations"]) != len(control_scene["annotations"]):
            raise RuntimeError("Jumlah object paired scene berbeda")
        for index, (original_annotation, control_annotation) in enumerate(
            zip(original_scene["annotations"], control_scene["annotations"])
        ):
            draw_original = (
                int(original_annotation["category_id"]),
                str(original_annotation["source_asset_id"]),
                original_annotation.get("source_parent_id"),
            )
            draw_control = (
                int(control_annotation["category_id"]),
                str(control_annotation["source_asset_id"]),
                control_annotation.get("source_parent_id"),
            )
            if draw_original != draw_control:
                raise RuntimeError("Source draw paired scene tidak identik")
            z_order = int(original_annotation["z_order"])
            if z_order not in original_diagnosis_by_z or z_order not in control_diagnosis_by_z:
                # Visibility can place a source draw below the benchmark's
                # minimum visible fraction in one arm. It is not part of the
                # paired detection estimand because one side has no GT label.
                continue
            original_diagnosis = original_diagnosis_by_z[z_order]
            control_diagnosis = control_diagnosis_by_z[z_order]
            rows.append(
                {
                    "scene_id": scene_id,
                    "object_index": index,
                    "class_id": draw_original[0],
                    "class_name": SNI21_CLASSES[draw_original[0]],
                    "source_asset_id": draw_original[1],
                    "source_parent_id": draw_original[2],
                    "original_image": str(
                        original_root / original_scene["image"]["file_name"]
                    ),
                    "control_image": str(
                        control_root / control_scene["image"]["file_name"]
                    ),
                    "original_bbox": _xywh_to_xyxy(original_annotation["bbox"]),
                    "control_bbox": _xywh_to_xyxy(control_annotation["bbox"]),
                    "original_category": str(original_diagnosis["category"]),
                    "control_category": str(control_diagnosis["category"]),
                    "original_predicted_class": original_diagnosis.get(
                        "best_prediction_class"
                    ),
                    "control_predicted_class": control_diagnosis.get(
                        "best_prediction_class"
                    ),
                    "original_best_iou": float(original_diagnosis["best_iou"]),
                    "control_best_iou": float(control_diagnosis["best_iou"]),
                    "transition": (
                        f"{original_diagnosis['category']} -> "
                        f"{control_diagnosis['category']}"
                    ),
                }
            )
    return rows


def _prediction_name(class_id: int | None) -> str:
    return "none" if class_id is None else SNI21_CLASSES[int(class_id)]


def _checkerboard(size: tuple[int, int], block: int = 16) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    for y in range(0, height, block):
        for x in range(0, width, block):
            shade = 225 if (x // block + y // block) % 2 else 250
            draw.rectangle(
                (x, y, min(x + block, width), min(y + block, height)),
                fill=(shade, shade, shade),
            )
    return canvas


def _asset_panel(asset_path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(asset_path) as source:
        rgba = source.convert("RGBA")
    content = ImageOps.contain(rgba, (size[0] - 20, size[1] - 45))
    panel = _checkerboard(size)
    position = ((size[0] - content.width) // 2, 35 + (size[1] - 35 - content.height) // 2)
    panel.paste(content, position, content)
    ImageDraw.Draw(panel).text((8, 8), "CUTOUT RGBA", fill="black")
    return panel


def _scene_panel(
    image_path: Path,
    bbox: list[float],
    *,
    title: str,
    predicted_class: int | None,
    category: str,
    size: tuple[int, int],
) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    margin = max(width, height) * 0.9
    crop_box = (
        max(0, int(np.floor(x1 - margin))),
        max(0, int(np.floor(y1 - margin))),
        min(image.width, int(np.ceil(x2 + margin))),
        min(image.height, int(np.ceil(y2 + margin))),
    )
    crop = image.crop(crop_box)
    draw = ImageDraw.Draw(crop)
    local_box = (
        int(round(x1 - crop_box[0])),
        int(round(y1 - crop_box[1])),
        int(round(x2 - crop_box[0])),
        int(round(y2 - crop_box[1])),
    )
    draw.rectangle(local_box, outline="#00e676", width=max(2, crop.width // 100))
    content = ImageOps.contain(crop, (size[0] - 10, size[1] - 62))
    panel = Image.new("RGB", size, "white")
    panel.paste(content, ((size[0] - content.width) // 2, 58))
    header = ImageDraw.Draw(panel)
    header.text((8, 6), title, fill="black")
    header.text(
        (8, 25),
        f"{category} | pred={_prediction_name(predicted_class)}",
        fill="#b00020" if category != "localized_correct" else "#006400",
    )
    return panel


def _select_rows(rows: list[dict], samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["transition"]].append(row)
    for values in grouped.values():
        rng.shuffle(values)
    selected = []
    transitions = sorted(grouped, key=lambda key: (-len(grouped[key]), key))
    # First reserve one example for every observed transition.
    for transition in transitions:
        if len(selected) >= min(samples, len(rows)):
            break
        selected.append(grouped[transition].pop())

    # Then spread remaining slots over transition x class buckets so a
    # dominant normal class cannot fill the contact sheet.
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for transition, values in grouped.items():
        for row in values:
            buckets[(transition, int(row["class_id"]))].append(row)
    keys = list(buckets)
    rng.shuffle(keys)
    while keys and len(selected) < min(samples, len(rows)):
        remaining = []
        for key in keys:
            if buckets[key] and len(selected) < samples:
                selected.append(buckets[key].pop())
            if buckets[key]:
                remaining.append(key)
        keys = remaining
    return selected


def run_sni21_b0_visual_audit(
    source_benchmark_root: str | Path,
    density_evaluation_root: str | Path,
    control_benchmark_root: str | Path,
    control_evaluation_root: str | Path,
    output_root: str | Path,
    *,
    samples: int = 18,
    seed: int = 42,
) -> dict:
    source_root = Path(source_benchmark_root).expanduser().resolve()
    density_root = Path(density_evaluation_root).expanduser().resolve()
    control_benchmark_root = Path(control_benchmark_root).expanduser().resolve()
    control_evaluation_root = Path(control_evaluation_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if samples <= 0:
        raise ValueError("samples harus positif")

    scale_summary = _read_json(
        control_evaluation_root / "b0_native_scale_summary.json",
        "Summary native-scale",
    )
    if scale_summary.get("training_executed") is not False:
        raise RuntimeError("Audit menolak artefak yang menjalankan training")
    if scale_summary.get("test_images_accessed") is not False:
        raise RuntimeError("Audit menolak artefak yang mengakses test")
    if scale_summary.get("paired_draws", {}).get("exact_draw_match") is not True:
        raise RuntimeError("Audit pairing native-scale belum PASS")

    original_data = source_root / ORIGINAL_CONDITION
    control_data = control_benchmark_root / CONTROL_CONDITION
    rows = build_paired_rows(
        original_data,
        control_data,
        density_root / ORIGINAL_CONDITION / "prediction_records.jsonl",
        control_evaluation_root / CONTROL_CONDITION / "prediction_records.jsonl",
    )
    transition_counts = Counter(row["transition"] for row in rows)
    selected = _select_rows(rows, samples, seed)

    library = _read_json(
        source_root / "val_object_library" / "object_library.json",
        "Object library",
    )
    assets = {str(row["asset_id"]): row for row in library["assets"]}
    panel_size = (360, 300)
    output_root.mkdir(parents=True, exist_ok=True)
    rows_root = output_root / "rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    rendered = []
    selected_payload = []
    for rank, row in enumerate(selected, 1):
        asset = assets[row["source_asset_id"]]
        asset_path = source_root / "val_object_library" / asset["image"]
        panels = [
            _asset_panel(asset_path, panel_size),
            _scene_panel(
                Path(row["original_image"]),
                row["original_bbox"],
                title="B0 ORIGINAL (small)",
                predicted_class=row["original_predicted_class"],
                category=row["original_category"],
                size=panel_size,
            ),
            _scene_panel(
                Path(row["control_image"]),
                row["control_bbox"],
                title="B0 NATIVE-SCALE",
                predicted_class=row["control_predicted_class"],
                category=row["control_category"],
                size=panel_size,
            ),
        ]
        header_height = 54
        canvas = Image.new(
            "RGB", (panel_size[0] * 3, panel_size[1] + header_height), "white"
        )
        header = ImageDraw.Draw(canvas)
        header.text(
            (8, 7),
            f"#{rank} GT={row['class_name']} | {row['transition']}",
            fill="black",
        )
        header.text(
            (8, 27),
            f"asset={row['source_asset_id']} | parent={row['source_parent_id']}",
            fill="#444444",
        )
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * panel_size[0], header_height))
        target = rows_root / f"{rank:02d}_{row['class_name']}.jpg"
        canvas.save(target, quality=93)
        rendered.append(canvas)
        selected_payload.append({**row, "rendered": str(target)})

    sheet = Image.new(
        "RGB",
        (
            panel_size[0] * 3,
            sum(image.height for image in rendered),
        ),
        "white",
    )
    cursor = 0
    for image in rendered:
        sheet.paste(image, (0, cursor))
        cursor += image.height
    sheet_path = output_root / "paired_contact_sheet.jpg"
    sheet.save(sheet_path, quality=91)

    table_path = output_root / "transition_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("transition", "count", "fraction")
        )
        writer.writeheader()
        for transition, count in transition_counts.most_common():
            writer.writerow(
                {
                    "transition": transition,
                    "count": count,
                    "fraction": count / len(rows),
                }
            )
    payload = {
        "format": "coffee_detector.sni21_b0_paired_visual_audit.v1",
        "objects": len(rows),
        "selected_samples": len(selected_payload),
        "selection_seed": seed,
        "transition_counts": dict(transition_counts.most_common()),
        "transition_table": str(table_path),
        "contact_sheet": str(sheet_path),
        "selected": selected_payload,
        "legend": {
            "green_box": "ground-truth visible bbox",
            "cutout_rgba": "extracted object asset on checkerboard",
            "prediction_category": "best-IoU candidate at diagnostic IoU 0.5",
        },
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
    }
    summary_path = output_root / "visual_audit_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== B0 PAIRED VISUAL AUDIT ===", flush=True)
    print("OBJECTS     :", len(rows), flush=True)
    print("SAMPLES     :", len(selected_payload), flush=True)
    for transition, count in transition_counts.most_common():
        print(f"  {transition}: {count}", flush=True)
    print("CONTACT SHEET:", sheet_path, flush=True)
    print("TRAINING     : False", flush=True)
    print("TEST ACCESS  : False", flush=True)
    print("SUMMARY      :", summary_path, flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a no-training paired visual audit for B0 scale control."
    )
    parser.add_argument("--source-benchmark-root", required=True)
    parser.add_argument("--density-evaluation-root", required=True)
    parser.add_argument("--control-benchmark-root", required=True)
    parser.add_argument("--control-evaluation-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_sni21_b0_visual_audit(
        args.source_benchmark_root,
        args.density_evaluation_root,
        args.control_benchmark_root,
        args.control_evaluation_root,
        args.output_root,
        samples=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
