from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from .dataset import IMAGE_SUFFIXES, discover_layout, parse_label


COLORS = {
    "box": "#00b7ff",
    "target": "#ff2d55",
    "header": "#111111",
}


def _load_manifest(data_root: Path) -> dict[Path, dict]:
    path = data_root / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest tidak ditemukan: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in rows:
        image_path = Path(row["output_image"]).expanduser().resolve()
        if image_path in result:
            raise ValueError(f"Output image muncul dua kali di manifest: {image_path}")
        result[image_path] = row
    return result


def _collect_rows(
    data_root: Path,
    splits: tuple[str, ...],
) -> tuple[list[dict], dict[int, str]]:
    layout = discover_layout(data_root)
    manifest = _load_manifest(data_root)
    valid_ids = set(layout.names)
    rows = []
    for split in splits:
        if split not in layout.splits:
            raise FileNotFoundError(f"Split {split} tidak tersedia di {data_root}")
        images_root, labels_root = layout.splits[split]
        image_paths = sorted(
            path
            for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in image_paths:
            relative = image_path.relative_to(images_root)
            label_path = (labels_root / relative).with_suffix(".txt")
            boxes = parse_label(label_path, valid_ids)
            metadata = manifest.get(image_path.resolve())
            if metadata is None:
                raise ValueError(f"Image tidak ada di split manifest: {image_path}")
            if metadata["output_split"] != split:
                raise ValueError(
                    f"Split manifest tidak konsisten untuk {image_path}: "
                    f"{metadata['output_split']} != {split}"
                )
            rows.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "split": split,
                    "dataset": metadata["dataset"],
                    "parent_id": metadata["source_parent_id"],
                    "orientation_action": metadata["orientation_action"],
                    "boxes": boxes,
                }
            )
    return rows, layout.names


def _select_unique_parents(
    rows: list[dict],
    count: int,
    *,
    key,
) -> list[dict]:
    selected = []
    used = set()
    for row in sorted(rows, key=key):
        identity = (row["dataset"], row["parent_id"])
        if identity in used:
            continue
        used.add(identity)
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def _select_dense(rows: list[dict], count: int) -> list[dict]:
    return _select_unique_parents(
        rows,
        count,
        key=lambda row: (
            -len(row["boxes"]),
            row["dataset"],
            row["split"],
            str(row["image_path"]),
        ),
    )


def _select_rotated(rows: list[dict], count: int, seed: int) -> list[dict]:
    candidates = [
        row for row in rows if row["orientation_action"] == "rotate_clockwise"
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    split_order = {"train": 0, "val": 1}
    return _select_unique_parents(
        candidates,
        count,
        key=lambda row: (
            split_order.get(row["split"], 2),
            str(row["image_path"]),
        ),
    )


def _select_classes(
    rows: list[dict], names: dict[int, str]
) -> list[tuple[dict, int]]:
    selected = []
    used_parents = set()
    for class_id in sorted(names):
        candidates = []
        for row in rows:
            target_boxes = [box for box in row["boxes"] if box.class_id == class_id]
            if not target_boxes:
                continue
            largest_area = max(box.width * box.height for box in target_boxes)
            identity = (row["dataset"], row["parent_id"])
            candidates.append(
                (
                    identity in used_parents,
                    -largest_area,
                    len(row["boxes"]),
                    row["dataset"],
                    str(row["image_path"]),
                    row,
                )
            )
        if not candidates:
            continue
        chosen = min(candidates)[-1]
        used_parents.add((chosen["dataset"], chosen["parent_id"]))
        selected.append((chosen, class_id))
    return selected


def _render(
    row: dict,
    names: dict[int, str],
    *,
    target_class: int | None = None,
    max_side: int = 900,
) -> Image.Image:
    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    original_width, original_height = image.size
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    width, height = image.size
    header_height = 42
    canvas = Image.new("RGB", (width, height + header_height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    target_name = names[target_class] if target_class is not None else "-"
    draw.text(
        (6, 5),
        f"{row['dataset']} | {row['split']} | boxes={len(row['boxes'])} "
        f"| target={target_name}",
        fill=COLORS["header"],
    )
    draw.text(
        (6, 21),
        f"parent={row['parent_id'][:72]} | {row['orientation_action']}",
        fill=COLORS["header"],
    )
    scale_x = width / original_width
    scale_y = height / original_height
    line_width = max(2, round(max(width, height) / 350))
    for box in row["boxes"]:
        left = (box.x_center - box.width / 2) * original_width * scale_x
        top = (
            (box.y_center - box.height / 2) * original_height * scale_y
            + header_height
        )
        right = (box.x_center + box.width / 2) * original_width * scale_x
        bottom = (
            (box.y_center + box.height / 2) * original_height * scale_y
            + header_height
        )
        is_target = target_class is not None and box.class_id == target_class
        color = COLORS["target"] if is_target else COLORS["box"]
        draw.rectangle(
            (left, top, right, bottom), outline=color, width=line_width
        )
        if is_target or len(row["boxes"]) <= 8:
            draw.text(
                (left + 2, max(header_height, top - 12)),
                names[box.class_id],
                fill=color,
            )
    return canvas


def _contact_sheet(
    images: list[Image.Image],
    output: Path,
    *,
    columns: int = 4,
    thumb_width: int = 420,
) -> None:
    if not images:
        raise RuntimeError(f"Tidak ada image untuk contact sheet {output.name}")
    columns = min(columns, len(images))
    thumbnails = []
    for image in images:
        ratio = thumb_width / image.width
        thumbnails.append(
            image.resize(
                (thumb_width, max(1, round(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        )
    rows = (len(thumbnails) + columns - 1) // columns
    heights = []
    for row in range(rows):
        items = thumbnails[row * columns : (row + 1) * columns]
        heights.append(max(item.height for item in items))
    sheet = Image.new("RGB", (columns * thumb_width, sum(heights)), "white")
    top = 0
    for row, row_height in enumerate(heights):
        for column, image in enumerate(
            thumbnails[row * columns : (row + 1) * columns]
        ):
            sheet.paste(image, (column * thumb_width, top))
        top += row_height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def run_sni_fullscene_visual_audit(
    data_root: str | Path,
    output_root: str | Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    dense_samples: int = 12,
    rotated_samples: int = 12,
    seed: int = 42,
) -> dict:
    if "test" in splits:
        raise ValueError(
            "Visual audit test dikunci. Gunakan train/val sebelum protokol membuka test."
        )
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows, names = _collect_rows(data_root, splits)
    dense = _select_dense(rows, dense_samples)
    rotated = _select_rotated(rows, rotated_samples, seed)
    per_class = _select_classes(rows, names)

    dense_sheet = output_root / "dense_contact_sheet.jpg"
    rotated_sheet = output_root / "rotated_faruq_contact_sheet.jpg"
    class_sheet = output_root / "class_contact_sheet.jpg"
    _contact_sheet([_render(row, names) for row in dense], dense_sheet)
    _contact_sheet([_render(row, names) for row in rotated], rotated_sheet)
    _contact_sheet(
        [_render(row, names, target_class=class_id) for row, class_id in per_class],
        class_sheet,
        columns=3,
    )

    selected = {
        "dense": [
            {
                "image": str(row["image_path"]),
                "split": row["split"],
                "dataset": row["dataset"],
                "parent_id": row["parent_id"],
                "boxes": len(row["boxes"]),
            }
            for row in dense
        ],
        "rotated": [
            {
                "image": str(row["image_path"]),
                "split": row["split"],
                "dataset": row["dataset"],
                "parent_id": row["parent_id"],
                "boxes": len(row["boxes"]),
            }
            for row in rotated
        ],
        "classes": [
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "image": str(row["image_path"]),
                "split": row["split"],
                "dataset": row["dataset"],
                "parent_id": row["parent_id"],
            }
            for row, class_id in per_class
        ],
    }
    payload = {
        "format": "coffee_detector.sni21_fullscene_visual_audit.v1",
        "data_root": str(data_root),
        "splits_rendered": list(splits),
        "test_rendered": False,
        "images_scanned": len(rows),
        "boxes_scanned": sum(len(row["boxes"]) for row in rows),
        "images_by_split": dict(Counter(row["split"] for row in rows)),
        "images_by_dataset": dict(Counter(row["dataset"] for row in rows)),
        "rotated_images_available": sum(
            row["orientation_action"] == "rotate_clockwise" for row in rows
        ),
        "classes_rendered": len(per_class),
        "missing_classes": [
            names[class_id]
            for class_id in sorted(names)
            if class_id not in {item[1] for item in per_class}
        ],
        "legend": {
            "ordinary_box": COLORS["box"],
            "target_class_box": COLORS["target"],
        },
        "contact_sheets": {
            "dense": str(dense_sheet),
            "rotated_faruq": str(rotated_sheet),
            "classes": str(class_sheet),
        },
        "selected": selected,
        "training_executed": False,
    }
    write_path = output_root / "visual_audit.json"
    write_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render visual QA train/val unified SNI-21. Test tetap terkunci dan "
            "training tidak dijalankan."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dense-samples", type=int, default=12)
    parser.add_argument("--rotated-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_sni_fullscene_visual_audit(
        args.data_root,
        args.output_root,
        dense_samples=args.dense_samples,
        rotated_samples=args.rotated_samples,
        seed=args.seed,
    )
    print("=== VISUAL AUDIT SNI-21 SELESAI ===")
    print(f"Images scanned : {result['images_scanned']}")
    print(f"Boxes scanned  : {result['boxes_scanned']}")
    print(f"Classes        : {result['classes_rendered']}/21")
    for name, path in result["contact_sheets"].items():
        print(f"{name:14s}: {path}")
    print("TEST TETAP TERKUNCI. TRAINING BELUM DIJALANKAN.")


if __name__ == "__main__":
    main()
