from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    "visible": "#00b7ff",
    "full": "#ffd60a",
    "focus": "#ff2d55",
}


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    width: int = 2,
    dash: int = 7,
) -> None:
    left, top, right, bottom = box
    for start in range(left, right, dash * 2):
        draw.line((start, top, min(start + dash, right), top), fill=fill, width=width)
        draw.line((start, bottom, min(start + dash, right), bottom), fill=fill, width=width)
    for start in range(top, bottom, dash * 2):
        draw.line((left, start, left, min(start + dash, bottom)), fill=fill, width=width)
        draw.line((right, start, right, min(start + dash, bottom)), fill=fill, width=width)


def _xywh_to_xyxy(box: list[int | float]) -> tuple[int, int, int, int]:
    x, y, width, height = (int(round(value)) for value in box)
    return x, y, x + width, y + height


def _select_images(images: list[dict], samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in images:
        grouped[str(row.get("target_visibility_bin") or "naive")].append(row)
    for rows in grouped.values():
        rng.shuffle(rows)
    selected = []
    names = sorted(grouped)
    while len(selected) < min(samples, len(images)) and names:
        remaining = []
        for name in names:
            if grouped[name] and len(selected) < samples:
                selected.append(grouped[name].pop())
            if grouped[name]:
                remaining.append(name)
        names = remaining
    return selected


def _contact_sheet(
    images: list[Image.Image],
    output: Path,
    *,
    columns: int = 4,
    thumb_width: int = 360,
) -> None:
    if not images:
        return
    columns = min(columns, len(images))
    thumbs = []
    for item in images:
        ratio = thumb_width / item.width
        thumbs.append(
            item.resize(
                (thumb_width, max(1, int(round(item.height * ratio)))),
                Image.Resampling.LANCZOS,
            )
        )
    rows = (len(thumbs) + columns - 1) // columns
    row_height = max(item.height for item in thumbs)
    sheet = Image.new("RGB", (columns * thumb_width, rows * row_height), "white")
    for index, item in enumerate(thumbs):
        sheet.paste(
            item,
            ((index % columns) * thumb_width, (index // columns) * row_height),
        )
    sheet.save(output, quality=90)


def run_vadcp_visual_audit(
    data_root: str | Path,
    output_root: str | Path,
    *,
    samples: int = 16,
    seed: int = 42,
) -> dict:
    data_root = Path(data_root).expanduser().resolve()
    metadata_path = data_root / "metadata" / "instances_synthetic_train.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata VA-DCP tidak ditemukan: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]) for row in metadata["categories"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for row in metadata["annotations"]:
        annotations[int(row["image_id"])].append(row)
    selected = _select_images(metadata["images"], samples, seed)
    output_root = Path(output_root).expanduser().resolve()
    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    rendered = []
    raw_images = []
    selected_rows = []
    for rank, image_row in enumerate(selected, 1):
        image_path = data_root / image_row["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        raw_images.append(image.copy())
        header = 30
        canvas = Image.new("RGB", (image.width, image.height + header), "white")
        canvas.paste(image, (0, header))
        draw = ImageDraw.Draw(canvas)
        target = image_row.get("target_visibility_bin") or "naive"
        draw.text(
            (6, 8),
            f"target={target} hit={image_row.get('target_visibility_hit')}",
            fill="black",
        )
        for row in annotations[int(image_row["id"])]:
            if row.get("full_bbox"):
                full = _xywh_to_xyxy(row["full_bbox"])
                full = (full[0], full[1] + header, full[2], full[3] + header)
                _dashed_rectangle(draw, full, COLORS["full"])
            if row.get("bbox") and not int(row.get("ignore", 0)):
                visible = _xywh_to_xyxy(row["bbox"])
                visible = (
                    visible[0],
                    visible[1] + header,
                    visible[2],
                    visible[3] + header,
                )
                color = COLORS["focus"] if row.get("is_focus") else COLORS["visible"]
                draw.rectangle(visible, outline=color, width=2)
                label = (
                    f"{categories[int(row['category_id'])]} "
                    f"v={float(row['visibility_ratio']):.2f}"
                )
                draw.text((visible[0] + 2, max(header, visible[1] - 11)), label, fill=color)
        target_path = images_root / f"{rank:02d}_{image_path.stem}.jpg"
        canvas.save(target_path, quality=92)
        rendered.append(canvas)
        selected_rows.append(
            {
                "image_id": image_row["id"],
                "source": str(image_path),
                "rendered": str(target_path),
                "target_visibility_bin": target,
                "target_visibility_hit": image_row.get("target_visibility_hit"),
            }
        )

    contact_sheet = output_root / "contact_sheet.jpg"
    raw_contact_sheet = output_root / "contact_sheet_raw.jpg"
    _contact_sheet(rendered, contact_sheet)
    _contact_sheet(raw_images, raw_contact_sheet)
    payload = {
        "data_root": str(data_root),
        "samples": len(selected_rows),
        "legend": {
            "visible_bbox": COLORS["visible"],
            "full_bbox_dashed": COLORS["full"],
            "focus_visible_bbox": COLORS["focus"],
        },
        "selected": selected_rows,
        "contact_sheet": str(contact_sheet),
        "raw_contact_sheet": str(raw_contact_sheet),
    }
    (output_root / "visual_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat contact sheet full/visible bbox untuk QA data VA-DCP."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_vadcp_visual_audit(
        args.data_root, args.output_root, samples=args.samples, seed=args.seed
    )
    print("=== VISUAL AUDIT VA-DCP ===")
    print(f"Samples      : {result['samples']}")
    print(f"Contact sheet: {result['contact_sheet']}")
    print(f"Raw sheet    : {result['raw_contact_sheet']}")


if __name__ == "__main__":
    main()
