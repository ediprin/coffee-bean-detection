from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .dataset import IMAGE_SUFFIXES, Box, discover_layout, parse_label


@dataclass(frozen=True)
class VisualSample:
    split: str
    image_path: Path
    target: Box
    boxes: tuple[Box, ...]


def _collect_samples(data_root: Path, split: str, class_id: int) -> list[VisualSample]:
    layout = discover_layout(data_root)
    if "test" in layout.splits or (data_root / "test").exists():
        raise RuntimeError(f"Test tidak boleh tersedia: {data_root}")
    image_root, label_root = layout.splits[split]
    samples = []
    for image_path in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image_path.relative_to(image_root)
        boxes = parse_label(
            (label_root / relative).with_suffix(".txt"), set(layout.names)
        )
        targets = [box for box in boxes if box.class_id == class_id]
        if targets:
            samples.append(VisualSample(split, image_path, targets[0], boxes))
    return samples


def _pixel_box(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    left = (box.x_center - box.width / 2) * width
    top = (box.y_center - box.height / 2) * height
    right = (box.x_center + box.width / 2) * width
    bottom = (box.y_center + box.height / 2) * height
    return left, top, right, bottom


def _render_crop(sample: VisualSample, class_name: str, tile_size: int = 240) -> Image.Image:
    with Image.open(sample.image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    left, top, right, bottom = _pixel_box(sample.target, width, height)
    box_width = max(1.0, right - left)
    box_height = max(1.0, bottom - top)
    side = max(box_width, box_height) * 2.2
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = max(0, int(round(center_x - side / 2)))
    crop_top = max(0, int(round(center_y - side / 2)))
    crop_right = min(width, int(round(center_x + side / 2)))
    crop_bottom = min(height, int(round(center_y + side / 2)))
    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise ValueError(f"Crop target kosong: {sample.image_path}")
    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    draw = ImageDraw.Draw(crop)
    for box in sample.boxes:
        b_left, b_top, b_right, b_bottom = _pixel_box(box, width, height)
        local = (
            b_left - crop_left,
            b_top - crop_top,
            b_right - crop_left,
            b_bottom - crop_top,
        )
        color = (0, 220, 90) if box is sample.target else (255, 180, 0)
        draw.rectangle(local, outline=color, width=max(2, int(side / 100)))
    crop.thumbnail((tile_size, tile_size - 38), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_size, tile_size), "white")
    tile.paste(crop, ((tile_size - crop.width) // 2, 22))
    caption = f"{sample.split} | {sample.image_path.name[:25]}"
    ImageDraw.Draw(tile).text((4, 4), caption, fill="black", font=ImageFont.load_default())
    ImageDraw.Draw(tile).text(
        (4, tile_size - 14), class_name[:34], fill=(0, 100, 40), font=ImageFont.load_default()
    )
    return tile


def _write_sheet(
    train_samples: list[VisualSample],
    val_samples: list[VisualSample],
    class_name: str,
    output: Path,
    *,
    samples_per_split: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    selected = []
    for split_samples in (train_samples, val_samples):
        split_samples = list(split_samples)
        rng.shuffle(split_samples)
        selected.extend(split_samples[:samples_per_split])
    columns = 4
    tile_size = 240
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size + 32), (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        f"{class_name} | green=target, amber=other object",
        fill="black",
        font=ImageFont.load_default(),
    )
    for index, sample in enumerate(selected):
        tile = _render_crop(sample, class_name, tile_size)
        x = (index % columns) * tile_size
        y = 32 + (index // columns) * tile_size
        canvas.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def _select_targets(class_audit: dict) -> dict[str, list[dict]]:
    selected = {}
    for source, summary in class_audit["sources"].items():
        candidates = [
            row
            for row in class_audit["rows"]
            if row["source_dataset"] == source
            and row["train_instances"] >= 50
            and row["val_instances"] >= 10
            and row["map50_95"] is not None
            and row["map50_95"] < 0.35
        ]
        candidates.sort(key=lambda row: (row["map50_95"], -row["val_instances"]))
        targets = candidates[:5]
        shifted = summary.get("largest_train_val_prevalence_shifts", [])
        for row in shifted:
            if row["val_instances"] >= 10 and all(
                item["class_id"] != row["class_id"] for item in targets
            ):
                targets.append(row)
                break
        selected[source] = targets
    return selected


def run_sni21_hard_class_visual_audit(
    separated_root: str | Path,
    class_audit_summary: str | Path,
    output_root: str | Path,
    *,
    samples_per_split: int = 8,
    seed: int = 42,
) -> dict:
    separated_root = Path(separated_root).expanduser().resolve()
    class_audit_summary = Path(class_audit_summary).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    audit = json.loads(class_audit_summary.read_text(encoding="utf-8"))
    if audit.get("training_executed") is not False:
        raise RuntimeError("Class audit mencatat training")
    if audit.get("test_images_accessed") is not False:
        raise RuntimeError("Class audit pernah mengakses test")
    if samples_per_split < 1:
        raise ValueError("samples_per_split minimal 1")

    targets = _select_targets(audit)
    sheets = []
    for source, rows in targets.items():
        data_root = separated_root / source
        for row in rows:
            class_id = int(row["class_id"])
            class_name = row["class_name"]
            train_samples = _collect_samples(data_root, "train", class_id)
            val_samples = _collect_samples(data_root, "val", class_id)
            output = output_root / source / f"{class_id:02d}_{class_name}.jpg"
            _write_sheet(
                train_samples,
                val_samples,
                class_name,
                output,
                samples_per_split=samples_per_split,
                seed=seed + class_id,
            )
            sheets.append(
                {
                    "source_dataset": source,
                    "class_id": class_id,
                    "class_name": class_name,
                    "train_available_images": len(train_samples),
                    "val_available_images": len(val_samples),
                    "train_rendered": min(samples_per_split, len(train_samples)),
                    "val_rendered": min(samples_per_split, len(val_samples)),
                    "contact_sheet": str(output),
                    "selection_ap": row.get("map50_95"),
                    "selection_prevalence_ratio": row.get(
                        "val_to_train_prevalence_ratio"
                    ),
                }
            )
            print(f"VISUAL AUDIT: {source}/{class_name} -> {output}", flush=True)

    summary = {
        "format": "coffee_detector.sni21_hard_class_visual_audit.v1",
        "separated_root": str(separated_root),
        "class_audit_summary": str(class_audit_summary),
        "selection_rule": (
            "AP<0.35, train>=50, val>=10; maksimum lima kelas per source, "
            "ditambah satu prevalence-shift yang belum terpilih."
        ),
        "samples_per_split": samples_per_split,
        "seed": seed,
        "sheets": sheets,
        "training_executed": False,
        "inference_executed": False,
        "test_images_accessed": False,
        "development_only": True,
    }
    summary_path = output_root / "hard_class_visual_audit_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat contact sheet train/val untuk kelas SNI-21 sulit."
    )
    parser.add_argument("--separated-root", required=True)
    parser.add_argument("--class-audit-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-per-split", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_sni21_hard_class_visual_audit(
        args.separated_root,
        args.class_audit_summary,
        args.output_root,
        samples_per_split=args.samples_per_split,
        seed=args.seed,
    )
    print("\n=== HARD-CLASS VISUAL AUDIT ===")
    for row in result["sheets"]:
        print(
            f"{row['source_dataset']}/{row['class_name']}: "
            f"train={row['train_rendered']} val={row['val_rendered']} | "
            f"{row['contact_sheet']}"
        )
    print("TRAINING:", result["training_executed"])
    print("INFERENCE:", result["inference_executed"])
    print("TEST ACCESSED:", result["test_images_accessed"])
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
