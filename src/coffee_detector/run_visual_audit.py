from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .dataset import discover_layout, parse_label


COLORS = {"ground_truth": "#0088ff", "prediction": "#ff3b30"}


def select_audit_rows(rows: list[dict], samples: int, seed: int) -> list[dict]:
    if samples <= 0:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (
            -row["absolute_count_error"],
            row["minimum_confidence"],
            row["image"],
        ),
    )
    selected = ordered[: min(samples, len(ordered))]
    if selected and all(row["absolute_count_error"] == 0 for row in selected):
        # All counts agree: keep the lowest-confidence half and add a seeded
        # random half so the contact sheet is not only one unusual scene type.
        hard_count = min((samples + 1) // 2, len(ordered))
        hard = ordered[:hard_count]
        hard_paths = {row["image"] for row in hard}
        pool = [row for row in rows if row["image"] not in hard_paths]
        rng = random.Random(seed)
        rng.shuffle(pool)
        selected = hard + pool[: max(0, samples - len(hard))]
    return selected


def _draw_ground_truth(image: Image.Image, boxes, names: dict[int, str]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for box in boxes:
        left = (box.x_center - box.width / 2) * width
        top = (box.y_center - box.height / 2) * height
        right = (box.x_center + box.width / 2) * width
        bottom = (box.y_center + box.height / 2) * height
        draw.rectangle((left, top, right, bottom), outline=COLORS["ground_truth"], width=3)
        draw.text((left + 2, max(0, top - 11)), names[box.class_id], fill=COLORS["ground_truth"])
    return canvas


def _draw_predictions(image: Image.Image, result, names: dict[int, str]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    if result.boxes is None:
        return canvas
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    for coordinates, class_id, confidence in zip(xyxy, classes, confidences):
        left, top, right, bottom = coordinates
        draw.rectangle((left, top, right, bottom), outline=COLORS["prediction"], width=3)
        label = f"{names[int(class_id)]} {confidence:.2f}"
        draw.text((left + 2, max(0, top - 11)), label, fill=COLORS["prediction"])
    return canvas


def _comparison_image(image_path: Path, boxes, result, names: dict[int, str]) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    ground_truth = _draw_ground_truth(image, boxes, names)
    prediction = _draw_predictions(image, result, names)
    header_height = 24
    output = Image.new("RGB", (image.width * 2, image.height + header_height), "white")
    output.paste(ground_truth, (0, header_height))
    output.paste(prediction, (image.width, header_height))
    draw = ImageDraw.Draw(output)
    draw.text((6, 6), "GROUND TRUTH (blue)", fill=COLORS["ground_truth"])
    draw.text((image.width + 6, 6), "PREDICTION (red)", fill=COLORS["prediction"])
    return output


def _make_contact_sheet(images: list[Image.Image], output: Path, columns: int = 2) -> None:
    if not images:
        return
    thumb_width = 900
    thumbnails = []
    for image in images:
        ratio = thumb_width / image.width
        thumbnails.append(image.resize((thumb_width, max(1, round(image.height * ratio)))))
    rows = (len(thumbnails) + columns - 1) // columns
    row_heights = []
    for row in range(rows):
        items = thumbnails[row * columns : (row + 1) * columns]
        row_heights.append(max(item.height for item in items))
    sheet = Image.new("RGB", (thumb_width * columns, sum(row_heights)), "white")
    y = 0
    for row, row_height in enumerate(row_heights):
        for column, image in enumerate(thumbnails[row * columns : (row + 1) * columns]):
            sheet.paste(image, (column * thumb_width, y))
        y += row_height
    sheet.save(output, quality=90)


def run_visual_audit(
    checkpoint: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    samples: int = 12,
    seed: int = 42,
    device: str | None = None,
    confidence: float = 0.25,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang. Jalankan `pip install -e .`.") from error

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    layout = discover_layout(data_root)
    if "test" not in layout.splits:
        raise FileNotFoundError("Split test tidak tersedia pada dataset")
    image_root, label_root = layout.splits["test"]
    image_paths = sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    output_root = Path(output_root).expanduser().resolve()
    pairs_root = output_root / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)

    kwargs = {"source": [str(path) for path in image_paths], "conf": confidence, "stream": True, "verbose": False}
    if device is not None:
        kwargs["device"] = device
    model = YOLO(str(checkpoint))
    rows = []
    valid_ids = set(layout.names)
    print(f"PREDICT TEST: {len(image_paths)} gambar", flush=True)
    for index, result in enumerate(model.predict(**kwargs), 1):
        image_path = Path(result.path).resolve()
        relative = image_path.relative_to(image_root.resolve())
        label_path = (label_root / relative).with_suffix(".txt")
        ground_truth = parse_label(label_path, valid_ids)
        confidences = [] if result.boxes is None else result.boxes.conf.detach().cpu().tolist()
        prediction_count = len(confidences)
        row = {
            "image": str(image_path),
            "label": str(label_path),
            "ground_truth_count": len(ground_truth),
            "prediction_count": prediction_count,
            "absolute_count_error": abs(prediction_count - len(ground_truth)),
            "minimum_confidence": min(confidences, default=0.0),
            "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        }
        rows.append(row)
        if index % 50 == 0 or index == len(image_paths):
            print(f"  {index}/{len(image_paths)}", flush=True)

    selected = select_audit_rows(rows, samples, seed)
    comparison_images = []
    selected_kwargs = {
        "source": [row["image"] for row in selected],
        "conf": confidence,
        "stream": False,
        "verbose": False,
    }
    if device is not None:
        selected_kwargs["device"] = device
    selected_results = model.predict(**selected_kwargs) if selected else []
    for rank, (row, result) in enumerate(zip(selected, selected_results), 1):
        ground_truth = parse_label(Path(row["label"]), valid_ids)
        comparison = _comparison_image(Path(row["image"]), ground_truth, result, layout.names)
        target = pairs_root / f"{rank:02d}_{Path(row['image']).stem}.jpg"
        comparison.save(target, quality=92)
        row["comparison"] = str(target)
        comparison_images.append(comparison)
    contact_sheet = output_root / "contact_sheet.jpg"
    _make_contact_sheet(comparison_images, contact_sheet)

    exact_count_matches = sum(row["absolute_count_error"] == 0 for row in rows)
    payload = {
        "checkpoint": str(checkpoint),
        "data_root": str(layout.root),
        "test_images": len(rows),
        "ground_truth_boxes": sum(row["ground_truth_count"] for row in rows),
        "predicted_boxes": sum(row["prediction_count"] for row in rows),
        "exact_count_match_images": exact_count_matches,
        "exact_count_match_rate": exact_count_matches / len(rows) if rows else 0.0,
        "images_with_count_error": len(rows) - exact_count_matches,
        "mean_absolute_count_error": (
            sum(row["absolute_count_error"] for row in rows) / len(rows) if rows else 0.0
        ),
        "confidence_threshold": confidence,
        "selected": selected,
        "rows": rows,
        "contact_sheet": str(contact_sheet),
    }
    summary_path = output_root / "visual_audit.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SAVED: {summary_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit visual GT vs prediksi pada test detector.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    payload = run_visual_audit(
        args.checkpoint,
        args.data_root,
        args.output_root,
        samples=args.samples,
        seed=args.seed,
        device=args.device,
        confidence=args.confidence,
    )
    print("\n=== VISUAL AUDIT ===")
    print(f"Test images            : {payload['test_images']}")
    print(f"Ground-truth boxes     : {payload['ground_truth_boxes']}")
    print(f"Predicted boxes        : {payload['predicted_boxes']}")
    print(f"Exact count match      : {payload['exact_count_match_rate'] * 100:.2f}%")
    print(f"Images with count error: {payload['images_with_count_error']}")
    print(f"Mean absolute error    : {payload['mean_absolute_count_error']:.4f}")
    print(f"Contact sheet          : {payload['contact_sheet']}")


if __name__ == "__main__":
    main()
