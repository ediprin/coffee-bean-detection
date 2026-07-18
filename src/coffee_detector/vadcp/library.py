from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..dataset import (
    IMAGE_SUFFIXES,
    Box,
    DatasetLayout,
    discover_layout,
    image_sha256,
    parse_label,
    roboflow_parent_id,
)
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


@dataclass(frozen=True)
class _BoxCandidate:
    image_path: Path
    box_index: int
    box: Box
    parent_id: str


def _sample_yolo_candidates(
    layout: DatasetLayout,
    source_split: str,
    *,
    max_assets_per_class: int | None,
    candidate_multiplier: int,
    max_assets_per_image_class: int,
    seed: int,
) -> tuple[list[_BoxCandidate], dict]:
    """Reservoir-sample boxes without opening or hashing every image.

    The previous implementation reused ``collect_records`` and therefore
    calculated SHA-256, dHash, and mean RGB for every image before sampling.
    Object-library preparation only needs labels at this stage, so a lightweight
    label pass is both equivalent and substantially faster.
    """
    if source_split not in layout.splits:
        raise FileNotFoundError(f"Split sumber tidak ditemukan: {source_split}")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier harus positif")
    if max_assets_per_image_class <= 0:
        raise ValueError("max_assets_per_image_class harus positif")
    capacity = (
        None
        if max_assets_per_class is None
        else max_assets_per_class * candidate_multiplier
    )
    image_root, label_root = layout.splits[source_split]
    image_paths = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    valid_ids = set(layout.names)
    reservoirs: dict[int, list[_BoxCandidate]] = {
        class_id: [] for class_id in layout.names
    }
    seen = Counter()
    eligible = Counter()
    skipped_per_image_cap = Counter()
    errors: list[str] = []
    rng = random.Random(seed)
    started = time.perf_counter()
    for image_index, image_path in enumerate(image_paths, 1):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        try:
            boxes = parse_label(label_path, valid_ids)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        per_image = Counter()
        parent_id = roboflow_parent_id(image_path)
        for box_index, box in enumerate(boxes):
            seen[box.class_id] += 1
            if per_image[box.class_id] >= max_assets_per_image_class:
                skipped_per_image_cap[box.class_id] += 1
                continue
            per_image[box.class_id] += 1
            eligible[box.class_id] += 1
            candidate = _BoxCandidate(image_path, box_index, box, parent_id)
            reservoir = reservoirs[box.class_id]
            if capacity is None or len(reservoir) < capacity:
                reservoir.append(candidate)
            else:
                replacement = rng.randrange(eligible[box.class_id])
                if replacement < capacity:
                    reservoir[replacement] = candidate
        if image_index % 1000 == 0 or image_index == len(image_paths):
            elapsed = time.perf_counter() - started
            print(
                f"  label index: {image_index}/{len(image_paths)} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
    candidates = [item for items in reservoirs.values() for item in items]
    rng.shuffle(candidates)
    stats = {
        "images_indexed": len(image_paths),
        "boxes_seen_by_class": {
            layout.names[key]: seen[key] for key in sorted(layout.names)
        },
        "eligible_by_class": {
            layout.names[key]: eligible[key] for key in sorted(layout.names)
        },
        "sampled_candidates_by_class": {
            layout.names[key]: len(reservoirs[key]) for key in sorted(layout.names)
        },
        "skipped_by_per_image_cap": {
            layout.names[key]: skipped_per_image_cap[key] for key in sorted(layout.names)
        },
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return candidates, stats


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
    box_padding_fraction: float = 0.12,
    minimum_fraction: float = 0.03,
    maximum_fraction: float = 0.96,
    max_assets_per_class: int | None = 500,
    seed: int = 42,
    candidate_multiplier: int = 2,
    max_assets_per_image_class: int = 3,
) -> dict:
    layout = discover_layout(data_root)
    source_split = SPLIT_NAMES.get(source_split.lower(), source_split.lower())
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output object library tidak kosong: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    assets: list[dict] = []
    failures: list[str] = []
    print("INDEX YOLO DATASET: reservoir-sampling label train...", flush=True)
    candidates, index_stats = _sample_yolo_candidates(
        layout,
        source_split,
        max_assets_per_class=max_assets_per_class,
        candidate_multiplier=candidate_multiplier,
        max_assets_per_image_class=max_assets_per_image_class,
        seed=seed,
    )
    if index_stats["errors"]:
        raise RuntimeError(
            "Dataset YOLO tidak valid:\n- "
            + "\n- ".join(index_stats["errors"][:20])
        )
    print(
        f"INDEX SELESAI: {index_stats['images_indexed']} gambar, "
        f"{len(candidates)} kandidat terpilih ({index_stats['elapsed_seconds']:.1f}s)",
        flush=True,
    )
    accepted_by_class = Counter()
    image_cache: dict[Path, Image.Image] = {}
    image_hash_cache: dict[Path, str] = {}
    extraction_started = time.perf_counter()
    for candidate in candidates:
        image_path = candidate.image_path
        box_index = candidate.box_index
        box = candidate.box
        class_name = layout.names[box.class_id]
        if (
            max_assets_per_class is not None
            and accepted_by_class[class_name] >= max_assets_per_class
        ):
            continue
        if image_path not in image_cache:
            with Image.open(image_path) as source_image:
                image_cache[image_path] = source_image.convert("RGBA")
            if len(image_cache) > 32:
                image_cache.pop(next(iter(image_cache)))
        image = image_cache[image_path]
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
        if image_path not in image_hash_cache:
            image_hash_cache[image_path] = image_sha256(image_path)
        source_id = _stable_id(
            image_hash_cache[image_path],
            box_index,
            box.class_id,
            *(round(value, 8) for value in (box.x_center, box.y_center, box.width, box.height)),
        )
        item, failure = _write_asset(
            output_root,
            class_name,
            source_id,
            source_split,
            image_path,
            crop,
            mask_threshold,
            padding,
            minimum_fraction,
            maximum_fraction,
        )
        if item:
            item["source_box_index"] = box_index
            item["source_parent_id"] = candidate.parent_id
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
            "candidate_multiplier": candidate_multiplier,
            "max_assets_per_image_class": max_assets_per_image_class,
            "index": index_stats,
            "extraction_elapsed_seconds": time.perf_counter() - extraction_started,
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
