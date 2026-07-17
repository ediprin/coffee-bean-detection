from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from ..dataset import IMAGE_SUFFIXES, collect_records, discover_layout
from .masks import crop_to_mask, estimate_foreground_mask
from .types import Cutout


SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


def _stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _infer_class_and_split(path: Path, root: Path) -> tuple[str, str]:
    directories = list(path.relative_to(root).parts[:-1])
    split = "unspecified"
    class_candidates = []
    for directory in directories:
        normalized = directory.strip().lower()
        if normalized in SPLIT_NAMES:
            split = SPLIT_NAMES[normalized]
        elif normalized not in {"images", "image", "data", "dataset"}:
            class_candidates.append(directory)
    if not class_candidates:
        raise ValueError(f"Nama kelas tidak dapat ditentukan dari path: {path}")
    return class_candidates[-1], split


def _write_asset(
    output_root: Path,
    class_name: str,
    source_id: str,
    source_split: str,
    source_path: Path,
    image: Image.Image,
    mask_threshold: float,
    padding: int,
    minimum_fraction: float,
    maximum_fraction: float,
) -> tuple[dict | None, str | None]:
    try:
        mask = estimate_foreground_mask(image, threshold=mask_threshold)
        fraction = float(mask.mean())
        if not (minimum_fraction <= fraction <= maximum_fraction):
            raise ValueError(
                f"foreground_fraction={fraction:.4f} di luar "
                f"[{minimum_fraction:.4f}, {maximum_fraction:.4f}]"
            )
        if (
            np.any(mask[0])
            or np.any(mask[-1])
            or np.any(mask[:, 0])
            or np.any(mask[:, -1])
        ):
            raise ValueError(
                "foreground menyentuh batas crop; full/amodal mask berpotensi terpotong"
            )
        rgba, cropped_mask = crop_to_mask(image, mask, padding=padding)
        asset_digest = hashlib.sha256(rgba.tobytes()).hexdigest()
        asset_id = _stable_id(class_name, source_id, asset_digest)
        class_folder = output_root / "assets" / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        target = class_folder / f"{asset_id}.png"
        rgba.save(target)
        cropped_fraction = float(cropped_mask.mean())
        return (
            {
                "asset_id": asset_id,
                "class_name": class_name,
                "image": target.relative_to(output_root).as_posix(),
                "source_id": source_id,
                "source_split": source_split,
                "source_path": str(source_path),
                "source_foreground_fraction": fraction,
                "cropped_foreground_fraction": cropped_fraction,
                "width": rgba.width,
                "height": rgba.height,
                "sha256_rgba": asset_digest,
            },
            None,
        )
    except (OSError, ValueError) as error:
        return None, f"{source_path}: {error}"


def _finalize_library(
    output_root: Path,
    assets: list[dict],
    failures: list[str],
    source: dict,
) -> dict:
    if not assets:
        raise RuntimeError("Tidak ada cutout valid yang dihasilkan")
    class_names = sorted({str(item["class_name"]) for item in assets})
    class_ids = {name: index for index, name in enumerate(class_names)}
    seen: dict[str, str] = {}
    unique_assets = []
    duplicate_assets = 0
    for item in assets:
        digest = str(item["sha256_rgba"])
        existing_class = seen.get(digest)
        if existing_class is not None:
            if existing_class != item["class_name"]:
                raise RuntimeError(
                    "Cutout pixel-identik memiliki kelas berbeda: "
                    f"{existing_class} vs {item['class_name']}"
                )
            Path(output_root / item["image"]).unlink(missing_ok=True)
            duplicate_assets += 1
            continue
        seen[digest] = str(item["class_name"])
        item["class_id"] = class_ids[str(item["class_name"])]
        unique_assets.append(item)

    payload = {
        "format": "coffee_detector.object_library.v1",
        "root": str(output_root),
        "source": source,
        "classes": {str(index): name for name, index in class_ids.items()},
        "assets": unique_assets,
        "audit": {
            "assets": len(unique_assets),
            "classes": len(class_names),
            "assets_by_class": dict(
                sorted(Counter(item["class_name"] for item in unique_assets).items())
            ),
            "assets_by_source_split": dict(
                sorted(Counter(item["source_split"] for item in unique_assets).items())
            ),
            "duplicate_assets_removed": duplicate_assets,
            "failures": len(failures),
            "failure_examples": failures[:100],
        },
    }
    path = output_root / "object_library.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def prepare_classification_library(
    image_root: str | Path,
    output_root: str | Path,
    *,
    allowed_splits: tuple[str, ...] = ("train", "unspecified"),
    mask_threshold: float = 24.0,
    padding: int = 2,
    minimum_fraction: float = 0.005,
    maximum_fraction: float = 0.92,
    max_assets_per_class: int | None = 500,
    seed: int = 42,
) -> dict:
    image_root = Path(image_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Folder klasifikasi tidak ditemukan: {image_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output object library tidak kosong: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    allowed = {SPLIT_NAMES.get(item.lower(), item.lower()) for item in allowed_splits}
    assets: list[dict] = []
    failures: list[str] = []
    skipped_splits = Counter()
    paths = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    print(f"INDEX CLASSIFICATION: {len(paths)} gambar", flush=True)
    random.Random(seed).shuffle(paths)
    accepted_by_class = Counter()
    for path in paths:
        try:
            class_name, source_split = _infer_class_and_split(path, image_root)
        except ValueError as error:
            failures.append(str(error))
            continue
        if source_split not in allowed:
            skipped_splits[source_split] += 1
            continue
        if (
            max_assets_per_class is not None
            and accepted_by_class[class_name] >= max_assets_per_class
        ):
            continue
        source_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source_image:
            image = source_image.convert("RGBA")
        item, failure = _write_asset(
            output_root,
            class_name,
            source_digest,
            source_split,
            path,
            image,
            mask_threshold,
            padding,
            minimum_fraction,
            maximum_fraction,
        )
        if item:
            assets.append(item)
            accepted_by_class[class_name] += 1
            if len(assets) % 100 == 0:
                print(f"  cutout valid: {len(assets)}", flush=True)
        if failure:
            failures.append(failure)
    return _finalize_library(
        output_root,
        assets,
        failures,
        {
            "type": "classification",
            "root": str(image_root),
            "allowed_splits": sorted(allowed),
            "skipped_by_split": dict(skipped_splits),
            "max_assets_per_class": max_assets_per_class,
            "seed": seed,
        },
    )


def prepare_yolo_library(
    data_root: str | Path,
    output_root: str | Path,
    *,
    source_split: str = "train",
    mask_threshold: float = 24.0,
    padding: int = 2,
    box_padding_fraction: float = 0.04,
    minimum_fraction: float = 0.03,
    maximum_fraction: float = 0.96,
    max_assets_per_class: int | None = 500,
    seed: int = 42,
) -> dict:
    layout = discover_layout(data_root)
    print("INDEX YOLO DATASET: membaca gambar dan label...", flush=True)
    records, errors = collect_records(layout)
    if errors:
        raise RuntimeError("Dataset YOLO tidak valid:\n- " + "\n- ".join(errors[:20]))
    source_split = SPLIT_NAMES.get(source_split.lower(), source_split.lower())
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output object library tidak kosong: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    assets: list[dict] = []
    failures: list[str] = []
    candidates = [
        (record, box_index, box)
        for record in records
        if record.split == source_split
        for box_index, box in enumerate(record.boxes)
    ]
    print(
        f"INDEX SELESAI: {len(records)} gambar, {len(candidates)} kandidat box {source_split}",
        flush=True,
    )
    random.Random(seed).shuffle(candidates)
    accepted_by_class = Counter()
    image_cache: dict[Path, Image.Image] = {}
    for record, box_index, box in candidates:
        class_name = layout.names[box.class_id]
        if (
            max_assets_per_class is not None
            and accepted_by_class[class_name] >= max_assets_per_class
        ):
            continue
        if record.image_path not in image_cache:
            with Image.open(record.image_path) as source_image:
                image_cache[record.image_path] = source_image.convert("RGBA")
            if len(image_cache) > 32:
                image_cache.pop(next(iter(image_cache)))
        image = image_cache[record.image_path]
        left = (box.x_center - box.width / 2.0) * image.width
        top = (box.y_center - box.height / 2.0) * image.height
        right = (box.x_center + box.width / 2.0) * image.width
        bottom = (box.y_center + box.height / 2.0) * image.height
        pad = max(right - left, bottom - top) * box_padding_fraction
        pixel_box = (
            max(0, int(np.floor(left - pad))),
            max(0, int(np.floor(top - pad))),
            min(image.width, int(np.ceil(right + pad))),
            min(image.height, int(np.ceil(bottom + pad))),
        )
        crop = image.crop(pixel_box)
        source_id = _stable_id(
            record.sha256,
            box_index,
            box.class_id,
            *(round(value, 8) for value in (box.x_center, box.y_center, box.width, box.height)),
        )
        item, failure = _write_asset(
            output_root,
            class_name,
            source_id,
            source_split,
            record.image_path,
            crop,
            mask_threshold,
            padding,
            minimum_fraction,
            maximum_fraction,
        )
        if item:
            item["source_box_index"] = box_index
            item["source_parent_id"] = record.parent_id
            assets.append(item)
            accepted_by_class[class_name] += 1
            if len(assets) % 100 == 0:
                print(f"  cutout valid: {len(assets)}", flush=True)
        if failure:
            failures.append(failure)
    return _finalize_library(
        output_root,
        assets,
        failures,
        {
            "type": "yolo_detection",
            "root": str(layout.root),
            "source_split": source_split,
            "max_assets_per_class": max_assets_per_class,
            "seed": seed,
        },
    )


def load_object_library(path: str | Path, *, train_only: bool = True) -> tuple[dict[int, str], list[Cutout], dict]:
    path = Path(path).expanduser().resolve()
    manifest_path = path / "object_library.json" if path.is_dir() else path
    if not manifest_path.is_file():
        raise FileNotFoundError(f"object_library.json tidak ditemukan: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "coffee_detector.object_library.v1":
        raise ValueError(f"Format object library tidak dikenal: {manifest_path}")
    root = manifest_path.parent
    classes = {int(key): str(value) for key, value in payload["classes"].items()}
    cutouts = []
    rejected = Counter()
    for row in payload["assets"]:
        source_split = str(row.get("source_split", "unspecified"))
        if train_only and source_split not in {"train", "unspecified"}:
            rejected[source_split] += 1
            continue
        cutouts.append(
            Cutout(
                asset_id=str(row["asset_id"]),
                class_id=int(row["class_id"]),
                class_name=str(row["class_name"]),
                image_path=(root / row["image"]).resolve(),
                source_id=str(row["source_id"]),
                source_split=source_split,
            )
        )
    if not cutouts:
        raise RuntimeError("Object library tidak memiliki aset train/unspecified")
    return classes, cutouts, {"manifest": payload, "rejected_splits": dict(rejected)}
