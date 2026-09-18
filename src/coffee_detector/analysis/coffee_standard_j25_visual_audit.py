"""Model-free visual and sibling-consistency audit for Coffee Standard J25."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from coffee_detector.data.prepare_coffee_standard_primary import J25_CLASSES
from coffee_detector.dataset import Box, collect_records, discover_layout


@dataclass(frozen=True)
class AuditObject:
    split: str
    image_path: Path
    box: Box
    class_name: str
    image_boxes: tuple[Box, ...]

    @property
    def area(self) -> float:
        return float(self.box.width * self.box.height)

    @property
    def aspect(self) -> float:
        return float(self.box.width / max(self.box.height, 1e-12))

    @property
    def touches_border(self) -> bool:
        return (
            self.box.x_center - self.box.width / 2 <= 0.002
            or self.box.y_center - self.box.height / 2 <= 0.002
            or self.box.x_center + self.box.width / 2 >= 0.998
            or self.box.y_center + self.box.height / 2 >= 0.998
        )


def _quantile_select(items: list[AuditObject], count: int) -> list[AuditObject]:
    ordered = sorted(items, key=lambda item: (item.area, str(item.image_path)))
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def _pixel_box(box: Box, width: int, height: int) -> tuple[float, float, float, float]:
    return (
        (box.x_center - box.width / 2) * width,
        (box.y_center - box.height / 2) * height,
        (box.x_center + box.width / 2) * width,
        (box.y_center + box.height / 2) * height,
    )


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    output = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    output.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return output


def _render_object(item: AuditObject, tile_size: tuple[int, int] = (300, 200)) -> Image.Image:
    with Image.open(item.image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    context = image.copy()
    draw = ImageDraw.Draw(context)
    for box in item.image_boxes:
        color = (0, 220, 80) if box == item.box else (130, 130, 130)
        draw.rectangle(_pixel_box(box, width, height), outline=color, width=max(2, width // 400))
    left, top, right, bottom = _pixel_box(item.box, width, height)
    margin = 0.20 * max(right - left, bottom - top, 1.0)
    crop = image.crop(
        (
            max(0, int(left - margin)),
            max(0, int(top - margin)),
            min(width, int(right + margin)),
            min(height, int(bottom + margin)),
        )
    )
    context = _fit(context, (140, 140))
    crop = _fit(crop, (140, 140))
    tile = Image.new("RGB", tile_size, (248, 248, 248))
    tile.paste(context, (5, 27))
    tile.paste(crop, (155, 27))
    label = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    label.text((5, 5), f"area={item.area:.5f} ar={item.aspect:.2f}", fill="black", font=font)
    label.text((5, 174), item.image_path.name[:43], fill=(60, 60, 60), font=font)
    return tile


def _write_class_sheet(
    split: str,
    by_class: dict[str, list[AuditObject]],
    output: Path,
    samples_per_class: int,
) -> list[dict]:
    tile_width, tile_height = 300, 200
    label_width, header_height = 220, 35
    canvas = Image.new(
        "RGB",
        (label_width + samples_per_class * tile_width, header_height + len(J25_CLASSES) * tile_height),
        (225, 225, 225),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 10), f"J25 {split} | deterministic box-area quantiles", fill="black", font=font)
    selected_rows = []
    for class_id, class_name in enumerate(J25_CLASSES):
        selected = _quantile_select(by_class[class_name], samples_per_class)
        y = header_height + class_id * tile_height
        draw.text((8, y + 8), f"{class_id}: {class_name}"[:34], fill="black", font=font)
        draw.text((8, y + 25), f"objects={len(by_class[class_name])}", fill=(60, 60, 60), font=font)
        for column, item in enumerate(selected):
            canvas.paste(_render_object(item), (label_width + column * tile_width, y))
            selected_rows.append(
                {
                    "sheet": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "image": str(item.image_path),
                    "area": item.area,
                    "aspect": item.aspect,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    return selected_rows


def _write_flag_sheet(items: list[tuple[str, AuditObject]], output: Path) -> list[dict]:
    columns = 4
    tile_width, tile_height = 300, 225
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, max(1, rows) * tile_height), "white")
    records = []
    for index, (reason, item) in enumerate(items):
        tile = Image.new("RGB", (tile_width, tile_height), (245, 245, 245))
        tile.paste(_render_object(item, (tile_width, 200)), (0, 25))
        ImageDraw.Draw(tile).text((5, 5), reason, fill=(180, 0, 0), font=ImageFont.load_default())
        canvas.paste(tile, ((index % columns) * tile_width, (index // columns) * tile_height))
        records.append(
            {
                "reason": reason,
                "split": item.split,
                "class_name": item.class_name,
                "image": str(item.image_path),
                "area": item.area,
                "aspect": item.aspect,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    return records


def _resolve_yolo_root(root: Path) -> Path:
    candidates = [root, *sorted({path.parent for path in root.rglob("data.yaml")})]
    valid = []
    for candidate in candidates:
        try:
            discover_layout(candidate)
        except (FileNotFoundError, ValueError):
            continue
        valid.append(candidate.resolve())
    valid = sorted(set(valid))
    if len(valid) != 1:
        raise RuntimeError(f"Harus ada tepat satu root YOLO; ditemukan: {valid}")
    return valid[0]


def _sibling_disagreement(source_root: Path) -> dict:
    layout = discover_layout(_resolve_yolo_root(source_root))
    records, errors = collect_records(layout, compute_visual_features=False, progress=True)
    if errors:
        raise RuntimeError(f"Source raw tidak valid: {errors[:10]}")
    by_parent = defaultdict(list)
    for record in records:
        by_parent[record.parent_id].append(record)
    rows = []
    signature_disagreements = 0
    for parent_id, siblings in sorted(by_parent.items()):
        signatures = []
        for sibling in siblings:
            counts = Counter(box.class_id for box in sibling.boxes)
            signatures.append(tuple(sorted(counts.items())))
        if len(set(signatures)) > 1:
            signature_disagreements += 1
            rows.append(
                {
                    "parent_id": parent_id,
                    "siblings": len(siblings),
                    "source_splits": sorted({sibling.split for sibling in siblings}),
                    "signatures": [list(map(list, signature)) for signature in sorted(set(signatures))],
                    "files": [str(sibling.image_path) for sibling in siblings[:12]],
                }
            )
    return {
        "source_images": len(records),
        "parents": len(by_parent),
        "parents_with_multiple_siblings": sum(len(rows) > 1 for rows in by_parent.values()),
        "parents_with_class_count_disagreement": signature_disagreements,
        "disagreement_fraction": signature_disagreements / max(1, len(by_parent)),
        "examples": rows[:100],
    }


def audit_coffee_standard_j25_visuals(
    grouped_root: str | Path,
    source_root: str | Path,
    output_root: str | Path,
    *,
    samples_per_class: int = 3,
    flagged_limit: int = 40,
) -> dict:
    """Create deterministic evidence sheets; no model is imported or executed."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class minimal 1")
    if flagged_limit < 4:
        raise ValueError("flagged_limit minimal 4")
    grouped_root = Path(grouped_root).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    summary_candidates = (
        grouped_root / "coffee_standard_j25_v2_summary.json",
        grouped_root / "coffee_standard_j25_summary.json",
    )
    summary_path = next((path for path in summary_candidates if path.is_file()), None)
    if summary_path is None:
        raise FileNotFoundError(
            f"Ringkasan J25 tidak ditemukan; dicari: {[str(path) for path in summary_candidates]}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("technical_split_ready") is not True:
        raise RuntimeError("J25 grouped belum lolos technical split gate")
    if summary.get("training_authorized") is not False:
        raise RuntimeError("Kontrak input harus masih training_authorized=False")

    layout = discover_layout(grouped_root)
    ordered_names = tuple(layout.names[index] for index in sorted(layout.names))
    if ordered_names != J25_CLASSES:
        raise RuntimeError("Ontologi grouped bukan J25")
    records, errors = collect_records(layout, compute_visual_features=False, progress=True)
    if errors:
        raise RuntimeError(f"Grouped dataset tidak valid: {errors[:10]}")
    by_split_class = {
        split: {name: [] for name in J25_CLASSES} for split in ("train", "val", "test")
    }
    objects = []
    for record in records:
        for box in record.boxes:
            item = AuditObject(
                record.split,
                record.image_path,
                box,
                layout.names[box.class_id],
                record.boxes,
            )
            objects.append(item)
            by_split_class[record.split][item.class_name].append(item)

    sheet_rows = []
    sheets = {}
    for split in ("train", "val", "test"):
        path = output_root / f"j25_{split}_class_review.jpg"
        sheet_rows.extend(
            _write_class_sheet(split, by_split_class[split], path, samples_per_class)
        )
        sheets[split] = str(path)

    by_area = sorted(objects, key=lambda item: (item.area, str(item.image_path)))
    by_aspect = sorted(
        objects,
        key=lambda item: (max(item.aspect, 1 / max(item.aspect, 1e-12)), str(item.image_path)),
        reverse=True,
    )
    border = sorted(
        (item for item in objects if item.touches_border),
        key=lambda item: (item.area, str(item.image_path)),
    )
    quarter = max(1, flagged_limit // 4)
    flags = (
        [("small-area", item) for item in by_area[:quarter]]
        + [("large-area", item) for item in by_area[-quarter:]]
        + [("extreme-aspect", item) for item in by_aspect[:quarter]]
        + [("border-touch", item) for item in border[:quarter]]
    )[:flagged_limit]
    flagged_sheet = output_root / "j25_geometry_flags.jpg"
    flagged_rows = _write_flag_sheet(flags, flagged_sheet)
    sibling = _sibling_disagreement(source_root)

    output_root.mkdir(parents=True, exist_ok=True)
    review_csv = output_root / "j25_visual_review_form.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as stream:
        fields = ["sheet", "class_id", "class_name", "image", "area", "aspect", "review_status", "notes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sheet_rows:
            writer.writerow({**row, "review_status": "", "notes": ""})

    payload = {
        "format": "coffee_detector.coffee_standard_j25_visual_audit.v1",
        "grouped_root": str(grouped_root),
        "source_root": str(source_root),
        "grouped_build_format": summary.get("format"),
        "grouped_selection_policy": summary.get("selection_policy", "v1-most-boxes"),
        "quarantined_identity_components": summary.get("quarantined_identity_components", 0),
        "selection": "deterministic per-class normalized-box-area quantiles",
        "samples_per_class_per_split": samples_per_class,
        "class_review_sheets": sheets,
        "geometry_flag_sheet": str(flagged_sheet),
        "selected_class_review_objects": len(sheet_rows),
        "geometry_flag_objects": len(flagged_rows),
        "sibling_consistency": sibling,
        "review_form": str(review_csv),
        "decision": "PENDING_HUMAN_VISUAL_REVIEW",
        "review_questions": [
            "Does every displayed box tightly cover the intended object?",
            "Is each class visually consistent across train, validation, and candidate test?",
            "Are black, sour/brown, hole, skin/horn, and size subclasses distinguishable under the written ontology?",
            "Do geometry flags reveal systematic truncation, merged objects, or annotation drift?",
            "Are sibling class-count disagreements explained only by augmentation crop/visibility changes?",
        ],
        "training_authorized": False,
        "training_executed": False,
        "model_inference_executed": False,
        "test_model_evaluation_executed": False,
        "next_action": "complete_human_review_form_and_freeze_label_decision",
    }
    output = output_root / "coffee_standard_j25_visual_audit.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build J25 visual-review evidence without training.")
    parser.add_argument("--grouped-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--flagged-limit", type=int, default=40)
    args = parser.parse_args()
    result = audit_coffee_standard_j25_visuals(
        args.grouped_root,
        args.source_root,
        args.output_root,
        samples_per_class=args.samples_per_class,
        flagged_limit=args.flagged_limit,
    )
    print("DECISION:", result["decision"])
    print("SHEETS:", result["class_review_sheets"])
    print("GEOMETRY:", result["geometry_flag_sheet"])
    print("SIBLING CONSISTENCY:", result["sibling_consistency"])
    print("TRAINING AUTHORIZED:", result["training_authorized"])
    print("SUMMARY:", result["summary"])


if __name__ == "__main__":
    main()
