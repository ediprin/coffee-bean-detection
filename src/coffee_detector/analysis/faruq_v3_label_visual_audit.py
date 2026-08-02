from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from coffee_detector.analysis.faruq_v3_label_identifiability import (
    LOCAL_DEFECT_CLASSES,
    ORDER,
    SIZE_FAMILIES,
)
from coffee_detector.dataset import IMAGE_SUFFIXES, Box, discover_layout, parse_label


@dataclass(frozen=True)
class VisualObject:
    split: str
    image_path: Path
    target: Box
    class_name: str

    @property
    def normalized_area(self) -> float:
        return float(self.target.width * self.target.height)


def _pixel_box(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    return (
        (box.x_center - box.width / 2) * width,
        (box.y_center - box.height / 2) * height,
        (box.x_center + box.width / 2) * width,
        (box.y_center + box.height / 2) * height,
    )


def _collect_objects(layout, split: str) -> dict[str, list[VisualObject]]:
    image_root, label_root = layout.splits[split]
    valid_ids = set(layout.names)
    by_class = {name: [] for name in layout.names.values()}
    image_paths = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for index, image_path in enumerate(image_paths, 1):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        for box in parse_label(label_path, valid_ids):
            class_name = layout.names[box.class_id]
            by_class[class_name].append(
                VisualObject(split, image_path, box, class_name)
            )
        if index % 500 == 0 or index == len(image_paths):
            print(f"VISUAL AUDIT {split}: {index}/{len(image_paths)}", flush=True)
    return by_class


def _quantile_select(objects: list[VisualObject], count: int) -> list[VisualObject]:
    """Select deterministic area quantiles instead of cherry-picked examples."""

    ordered = sorted(objects, key=lambda item: (item.normalized_area, str(item.image_path)))
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    output = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    output.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return output


def _render_object(item: VisualObject, tile_width: int = 330, tile_height: int = 210) -> Image.Image:
    with Image.open(item.image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    left, top, right, bottom = _pixel_box(item.target, width, height)

    context = image.copy()
    context_draw = ImageDraw.Draw(context)
    context_draw.rectangle((left, top, right, bottom), outline=(0, 210, 80), width=max(3, width // 400))
    context = _fit(context, (155, 155))

    box_width = max(1.0, right - left)
    box_height = max(1.0, bottom - top)
    margin = 0.18 * max(box_width, box_height)
    crop = image.crop(
        (
            max(0, int(left - margin)),
            max(0, int(top - margin)),
            min(width, int(right + margin)),
            min(height, int(bottom + margin)),
        )
    )
    crop = _fit(crop, (155, 155))

    tile = Image.new("RGB", (tile_width, tile_height), (247, 247, 247))
    tile.paste(context, (5, 29))
    tile.paste(crop, (170, 29))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    draw.text((5, 5), f"area={item.normalized_area:.5f}", fill="black", font=font)
    draw.text((5, 188), item.image_path.name[:48], fill=(60, 60, 60), font=font)
    draw.text((5, 174), "context", fill=(0, 110, 45), font=font)
    draw.text((170, 174), "object zoom", fill=(0, 110, 45), font=font)
    return tile


def _write_rows_sheet(
    title: str,
    rows: list[tuple[str, list[VisualObject]]],
    output: Path,
    *,
    samples_per_row: int,
) -> list[dict]:
    tile_width, tile_height = 330, 210
    label_width, header_height = 180, 36
    canvas = Image.new(
        "RGB",
        (label_width + samples_per_row * tile_width, header_height + len(rows) * tile_height),
        (225, 225, 225),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 10), title, fill="black", font=font)
    selected_records = []
    for row_index, (label, objects) in enumerate(rows):
        selected = _quantile_select(objects, samples_per_row)
        y = header_height + row_index * tile_height
        draw.text((8, y + 12), label, fill="black", font=font)
        draw.text((8, y + 30), f"available={len(objects)}", fill=(70, 70, 70), font=font)
        for column, item in enumerate(selected):
            tile = _render_object(item, tile_width, tile_height)
            canvas.paste(tile, (label_width + column * tile_width, y))
            selected_records.append(
                {
                    "row": label,
                    "split": item.split,
                    "class_name": item.class_name,
                    "image": str(item.image_path),
                    "normalized_area": item.normalized_area,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    return selected_records


def audit_faruq_v3_label_visuals(
    data_root: str | Path,
    diagnostic: str | Path,
    output_root: str | Path,
    *,
    samples_per_class: int = 6,
    max_local_pairs: int = 6,
) -> dict:
    """Create data-only contact sheets for label review; never train or infer."""

    layout = discover_layout(data_root)
    if "val" not in layout.splits:
        raise FileNotFoundError("Validation split tidak ditemukan")
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit visual development tidak boleh menyediakan test")
    if samples_per_class < 2:
        raise ValueError("samples_per_class minimal 2")
    if max_local_pairs < 1:
        raise ValueError("max_local_pairs minimal 1")

    diagnostic_path = Path(diagnostic).expanduser().resolve()
    diagnostic_payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostic_payload.get("training_executed") is not False:
        raise RuntimeError("Diagnostic sumber tidak menjamin training=False")
    if diagnostic_payload.get("test_images_accessed") is not False:
        raise RuntimeError("Diagnostic sumber pernah mengakses test")

    output_root = Path(output_root).expanduser().resolve()
    objects = {split: _collect_objects(layout, split) for split in ("train", "val")}

    size_sheets = []
    for split in ("train", "val"):
        for family, levels in SIZE_FAMILIES.items():
            rows = [
                (f"{level}: {class_name}", objects[split].get(class_name, []))
                for level, class_name in levels.items()
            ]
            output = output_root / "size_families" / f"{split}_{family}.jpg"
            selected = _write_rows_sheet(
                f"{split} | {family} | deterministic area quantiles",
                rows,
                output,
                samples_per_row=samples_per_class,
            )
            size_sheets.append(
                {
                    "split": split,
                    "family": family,
                    "contact_sheet": str(output),
                    "available": {
                        level: len(objects[split].get(class_name, []))
                        for level, class_name in levels.items()
                    },
                    "selected": selected,
                }
            )

    local_rows = []
    for raw in diagnostic_payload.get("top_directional_confusions", []):
        expected, predicted = raw["expected"], raw["predicted"]
        if expected in LOCAL_DEFECT_CLASSES and predicted in LOCAL_DEFECT_CLASSES:
            local_rows.append(raw)
    local_rows.sort(key=lambda row: (-int(row["count"]), row["expected"], row["predicted"]))

    pair_sheets = []
    for rank, confusion in enumerate(local_rows[:max_local_pairs], 1):
        expected, predicted = confusion["expected"], confusion["predicted"]
        for split in ("train", "val"):
            rows = [
                (f"expected: {expected}", objects[split].get(expected, [])),
                (f"confused-as: {predicted}", objects[split].get(predicted, [])),
            ]
            output = output_root / "local_pairs" / (
                f"{rank:02d}_{split}_{expected}__vs__{predicted}.jpg"
            )
            selected = _write_rows_sheet(
                f"{split} | confusion count={int(confusion['count'])}",
                rows,
                output,
                samples_per_row=samples_per_class,
            )
            pair_sheets.append(
                {
                    "rank": rank,
                    "split": split,
                    "expected": expected,
                    "predicted": predicted,
                    "confusion_count": int(confusion["count"]),
                    "contact_sheet": str(output),
                    "selected": selected,
                }
            )

    payload = {
        "protocol": "faruq-v3-label-visual-audit-v1",
        "dataset_root": str(layout.root),
        "diagnostic": str(diagnostic_path),
        "selection": "deterministic normalized-area quantiles",
        "samples_per_class": samples_per_class,
        "max_local_pairs": max_local_pairs,
        "size_sheets": size_sheets,
        "local_pair_sheets": pair_sheets,
        "decision": "PENDING_HUMAN_VISUAL_REVIEW",
        "review_questions": [
            "Are small, medium, and large labels visually ordered within each split?",
            "Does each scene provide a stable physical scale reference?",
            "Are local defect cues visible and consistent within each class?",
            "Do train and validation use the same visual interpretation of each label?",
        ],
        "training_executed": False,
        "inference_executed": False,
        "test_images_accessed": False,
        "development_only": True,
    }
    summary = output_root / "label_visual_audit_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 label visual audit without training.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--max-local-pairs", type=int, default=6)
    args = parser.parse_args()
    result = audit_faruq_v3_label_visuals(
        args.data_root,
        args.diagnostic,
        args.output_root,
        samples_per_class=args.samples_per_class,
        max_local_pairs=args.max_local_pairs,
    )
    print(json.dumps({
        "decision": result["decision"],
        "size_sheets": len(result["size_sheets"]),
        "local_pair_sheets": len(result["local_pair_sheets"]),
        "training_executed": result["training_executed"],
        "test_images_accessed": result["test_images_accessed"],
    }, indent=2, ensure_ascii=False))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
