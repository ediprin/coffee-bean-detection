from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEVELOPMENT_SPLITS = {"train": "train", "valid": "val", "val": "val"}
TRANSFORMS = {
    "identity": None,
    "rotate_ccw": Image.Transpose.ROTATE_90,
    "rotate_180": Image.Transpose.ROTATE_180,
    "rotate_cw": Image.Transpose.ROTATE_270,
    "flip_horizontal": Image.Transpose.FLIP_LEFT_RIGHT,
    "flip_vertical": Image.Transpose.FLIP_TOP_BOTTOM,
    "transpose": Image.Transpose.TRANSPOSE,
    "transverse": Image.Transpose.TRANSVERSE,
}


def _annotation_files(root: Path) -> list[tuple[str, Path]]:
    outputs = []
    for alias, split in DEVELOPMENT_SPLITS.items():
        split_root = root / alias
        if not split_root.is_dir():
            continue
        candidates = sorted(split_root.glob("*.json"))
        candidates.extend(sorted(split_root.glob("*.coco.json")))
        unique = []
        seen = set()
        for candidate in candidates:
            if candidate.is_file() and candidate.resolve() not in seen:
                seen.add(candidate.resolve())
                unique.append(candidate)
        if len(unique) != 1:
            raise RuntimeError(
                f"Diharapkan satu COCO JSON pada {split_root}, ditemukan {len(unique)}"
            )
        outputs.append((split, unique[0]))
    if not outputs:
        raise FileNotFoundError(f"COCO train/valid tidak ditemukan: {root}")
    if (root / "test").exists():
        print("TEST TERDETEKSI TETAPI TIDAK DIBACA", flush=True)
    return outputs


def _polygons(annotations: list[dict]) -> list[list[float]]:
    polygons = []
    for annotation in annotations:
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            continue
        if segmentation and all(isinstance(value, (int, float)) for value in segmentation):
            segmentation = [segmentation]
        for polygon in segmentation:
            if (
                isinstance(polygon, list)
                and len(polygon) >= 6
                and len(polygon) % 2 == 0
            ):
                polygons.append([float(value) for value in polygon])
    return polygons


def _mask_from_polygons(
    polygons: list[list[float]],
    expected_size: tuple[int, int],
    score_size: tuple[int, int],
) -> np.ndarray:
    expected_width, expected_height = expected_size
    score_width, score_height = score_size
    mask = Image.new("L", score_size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        points = [
            (
                polygon[index] / expected_width * score_width,
                polygon[index + 1] / expected_height * score_height,
            )
            for index in range(0, len(polygon), 2)
        ]
        draw.polygon(points, fill=255)
    return np.asarray(mask) > 0


def _candidate_images(
    image: Image.Image, expected_size: tuple[int, int]
) -> dict[str, Image.Image]:
    rgb = image.convert("RGB")
    candidates = {}
    for name, transform in TRANSFORMS.items():
        candidate = rgb.copy() if transform is None else rgb.transpose(transform)
        if candidate.size == expected_size:
            candidates[name] = candidate
        else:
            candidate.close()
    if not candidates:
        rgb.close()
        raise ValueError(
            f"Tidak ada transformasi yang cocok: raw={image.size}, expected={expected_size}"
        )
    rgb.close()
    return candidates


def _alignment_score(candidate: Image.Image, mask: np.ndarray) -> float:
    score_height, score_width = mask.shape
    resized = candidate.resize((score_width, score_height), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    border = np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]), axis=0
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(pixels - background, axis=2) / 441.67295593
    if mask.sum() < 4 or (~mask).sum() < 4:
        return float("-inf")
    inside = float(distance[mask].mean())
    outside = float(distance[~mask].mean())
    return inside - outside


def _current_transform(raw_size: tuple[int, int], expected_size: tuple[int, int]) -> str:
    if raw_size == expected_size:
        return "identity"
    if raw_size == (expected_size[1], expected_size[0]):
        return "rotate_cw"
    return "invalid"


def _render_overlay(
    image: Image.Image,
    polygons: list[list[float]],
    expected_size: tuple[int, int],
    title: str,
    tile_size: tuple[int, int] = (320, 260),
) -> Image.Image:
    tile_width, tile_height = tile_size
    preview_height = tile_height - 28
    preview = image.copy()
    preview.thumbnail((tile_width, preview_height), Image.Resampling.LANCZOS)
    scale_x = preview.width / expected_size[0]
    scale_y = preview.height / expected_size[1]
    draw = ImageDraw.Draw(preview)
    for polygon in polygons:
        points = [
            (polygon[index] * scale_x, polygon[index + 1] * scale_y)
            for index in range(0, len(polygon), 2)
        ]
        draw.line(points + [points[0]], fill=(0, 255, 80), width=2)
    tile = Image.new("RGB", tile_size, "white")
    tile.paste(preview, ((tile_width - preview.width) // 2, 24))
    ImageDraw.Draw(tile).text((5, 5), title[:52], fill="black", font=ImageFont.load_default())
    preview.close()
    return tile


def _write_contact_sheet(records: list[dict], output: Path, limit: int) -> None:
    flagged = [record for record in records if record["flagged_orientation"]][:limit]
    if not flagged:
        return
    tile_width, tile_height = 320, 260
    canvas = Image.new("RGB", (tile_width * 2, tile_height * len(flagged)), (230, 230, 230))
    for row, record in enumerate(flagged):
        with Image.open(record["image_path"]) as image:
            candidates = _candidate_images(image, tuple(record["expected_size"]))
        current = candidates.get(record["current_transform"])
        best = candidates[record["best_transform"]]
        if current is not None:
            left = _render_overlay(
                current,
                record["polygons"],
                tuple(record["expected_size"]),
                f"current={record['current_transform']} {record['current_score']:.3f}",
            )
            canvas.paste(left, (0, row * tile_height))
            left.close()
        right = _render_overlay(
            best,
            record["polygons"],
            tuple(record["expected_size"]),
            f"best={record['best_transform']} {record['best_score']:.3f}",
        )
        canvas.paste(right, (tile_width, row * tile_height))
        right.close()
        for candidate in candidates.values():
            candidate.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)
    canvas.close()


def audit_faruq_mask_geometry(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    score_long_side: int = 192,
    min_improvement: float = 0.02,
    contact_sheet_limit: int = 24,
) -> dict:
    raw_root = Path(raw_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    annotation_files = _annotation_files(raw_root)
    records = []
    counters = Counter()
    for split, annotation_path in annotation_files:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        by_image: dict[int | str, list[dict]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            by_image[annotation["image_id"]].append(annotation)
        images = payload.get("images", [])
        for index, image_record in enumerate(images, 1):
            image_path = annotation_path.parent / str(image_record["file_name"])
            if not image_path.is_file():
                counters["missing_images"] += 1
                continue
            expected_size = (int(image_record["width"]), int(image_record["height"]))
            polygons = _polygons(by_image.get(image_record["id"], []))
            if not polygons:
                counters["images_without_polygon"] += 1
                continue
            scale = score_long_side / max(expected_size)
            score_size = (
                max(16, int(round(expected_size[0] * scale))),
                max(16, int(round(expected_size[1] * scale))),
            )
            mask = _mask_from_polygons(polygons, expected_size, score_size)
            with Image.open(image_path) as opened:
                raw_size = opened.size
                candidates = _candidate_images(opened, expected_size)
            scores = {
                name: _alignment_score(candidate, mask)
                for name, candidate in candidates.items()
            }
            best_transform = max(scores, key=scores.get)
            current_transform = _current_transform(raw_size, expected_size)
            current_score = scores.get(current_transform, float("-inf"))
            best_score = scores[best_transform]
            improvement = best_score - current_score
            flagged = (
                best_transform != current_transform
                and np.isfinite(improvement)
                and improvement >= min_improvement
            )
            counters["audited_images"] += 1
            counters[f"current:{current_transform}"] += 1
            counters[f"best:{best_transform}"] += 1
            counters["flagged_orientation"] += int(flagged)
            records.append(
                {
                    "split": split,
                    "image_id": image_record["id"],
                    "image_path": str(image_path),
                    "raw_size": list(raw_size),
                    "expected_size": list(expected_size),
                    "polygons": polygons,
                    "current_transform": current_transform,
                    "best_transform": best_transform,
                    "current_score": float(current_score),
                    "best_score": float(best_score),
                    "best_improvement": float(improvement),
                    "flagged_orientation": bool(flagged),
                }
            )
            for candidate in candidates.values():
                candidate.close()
            if index % 250 == 0 or index == len(images):
                print(
                    f"GEOMETRY {split}: {index}/{len(images)} | "
                    f"flagged={counters['flagged_orientation']}",
                    flush=True,
                )

    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "faruq_geometry_records.json"
    records_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    sheet_path = output_root / "flagged_orientation_contact_sheet.jpg"
    _write_contact_sheet(records, sheet_path, contact_sheet_limit)
    summary = {
        "format": "coffee_detector.faruq_mask_geometry_audit.v1",
        "raw_root": str(raw_root),
        "splits_accessed": sorted({split for split, _ in annotation_files}),
        "test_images_accessed": False,
        "training_executed": False,
        "inference_executed": False,
        "score_long_side": score_long_side,
        "min_improvement": min_improvement,
        "counters": dict(counters),
        "flagged_fraction": (
            counters["flagged_orientation"] / counters["audited_images"]
            if counters["audited_images"]
            else 0.0
        ),
        "records": str(records_path),
        "contact_sheet": str(sheet_path) if sheet_path.is_file() else None,
        "decision": "AUDIT_ONLY",
        "next_action": (
            "Review overlay current vs best. Do not rewrite images/annotations "
            "until transform selection is visually confirmed."
        ),
    }
    summary_path = output_root / "faruq_geometry_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit orientasi Faruq train/val terhadap polygon mask tanpa repair."
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--score-long-side", type=int, default=192)
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--contact-sheet-limit", type=int, default=24)
    args = parser.parse_args()
    result = audit_faruq_mask_geometry(
        args.raw_root,
        args.output_root,
        score_long_side=args.score_long_side,
        min_improvement=args.min_improvement,
        contact_sheet_limit=args.contact_sheet_limit,
    )
    print("\n=== FARUQ MASK-GEOMETRY AUDIT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
