from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..dataset import (
    IMAGE_SUFFIXES,
    Box,
    DatasetLayout,
    discover_layout,
    image_sha256,
    parse_label,
    roboflow_parent_id,
)
from .masks import (
    crop_to_mask,
    estimate_foreground_mask,
    fill_holes,
    largest_component,
    mask_bbox,
    principal_mask_geometry,
)
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
    preferred_center: tuple[float, float] | None = None,
    explicit_mask: np.ndarray | None = None,
    border_trim_fraction: float = 0.0,
    background_model: str = "median",
) -> tuple[dict | None, str | None]:
    try:
        mask = (
            np.asarray(explicit_mask, dtype=bool)
            if explicit_mask is not None
            else estimate_foreground_mask(
                image,
                threshold=mask_threshold,
                preferred_point=preferred_center,
                background_model=background_model,
            )
        )
        if mask.shape != (image.height, image.width):
            raise ValueError("Ukuran explicit mask tidak sama dengan crop")
        if explicit_mask is None and border_trim_fraction > 0:
            trim = max(
                1,
                int(round(min(image.width, image.height) * border_trim_fraction)),
            )
            mask = mask.copy()
            mask[:trim] = False
            mask[-trim:] = False
            mask[:, :trim] = False
            mask[:, -trim:] = False
            mask = fill_holes(
                largest_component(mask, preferred_point=preferred_center)
            )
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
        centroid_y, centroid_x = np.mean(np.argwhere(mask), axis=0)
        centroid_distance_fraction = None
        if preferred_center is not None:
            distance = np.hypot(
                centroid_x - preferred_center[0],
                centroid_y - preferred_center[1],
            )
            centroid_distance_fraction = float(distance / max(image.size))
            if centroid_distance_fraction > 0.35:
                raise ValueError(
                    "centroid foreground terlalu jauh dari pusat bbox: "
                    f"{centroid_distance_fraction:.3f}"
                )
        rgba, cropped_mask = crop_to_mask(image, mask, padding=padding)
        cropped_box = mask_bbox(cropped_mask)
        if cropped_box is None:
            raise ValueError("mask hasil crop kosong")
        intrinsic_major, intrinsic_minor, _ = principal_mask_geometry(cropped_mask)
        intrinsic_aspect_ratio = intrinsic_major / max(intrinsic_minor, 1e-6)
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
                "centroid_distance_fraction": centroid_distance_fraction,
                "width": rgba.width,
                "height": rgba.height,
                "intrinsic_aspect_ratio": intrinsic_aspect_ratio,
                "sha256_rgba": asset_digest,
                "mask_source": "segmentation_polygon" if explicit_mask is not None else "estimated_foreground",
                "mask_background_model": (
                    None if explicit_mask is not None else background_model
                ),
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
            None,
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


def prepare_sni_crop_manifest_library(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    source_split: str = "train",
    normal_class: str = "biji_normal",
    max_normal_assets: int = 300,
    max_defect_assets_per_class: int = 60,
    candidate_multiplier: int = 1,
    mask_threshold: float = 40.0,
    padding: int = 2,
    minimum_fraction: float = 0.03,
    maximum_fraction: float = 0.96,
    seed: int = 42,
    shard_cache_root: str | Path | None = None,
) -> dict:
    """Build a train-only cutout library from the sharded SNI crop package.

    The crop package intentionally stores JPEG crops rather than alpha masks.
    Foreground masks are therefore estimated and must pass the same boundary
    checks as ordinary classification assets.  Selection is grouped by source
    identity before filling any remaining capacity, reducing repeated views
    from one parent scene.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    source_split = SPLIT_NAMES.get(source_split.lower(), source_split.lower())
    manifest_path = dataset_root / "manifest.csv"
    shard_root = dataset_root / "shards"
    complete_path = dataset_root / "complete.json"
    audit_path = dataset_root / "audit.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest crop tidak ditemukan: {manifest_path}")
    if not shard_root.is_dir():
        raise FileNotFoundError(f"Folder shard tidak ditemukan: {shard_root}")
    if not complete_path.is_file():
        raise FileNotFoundError(f"Marker complete tidak ditemukan: {complete_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit dataset tidak ditemukan: {audit_path}")
    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise RuntimeError(f"Dataset crop belum lengkap: {complete_path}")
    if max_normal_assets <= 0 or max_defect_assets_per_class <= 0:
        raise ValueError("Batas aset per kelas harus positif")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier harus positif")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output object library tidak kosong: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    required = {
        "dataset",
        "archive_split",
        "generated_split",
        "image_id",
        "source_identity",
        "canonical_class",
        "crop_sha256",
        "crop_path",
    }
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    split_counts = Counter()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError("Kolom manifest belum lengkap: " + ", ".join(missing))
        for row in reader:
            split = SPLIT_NAMES.get(
                str(row["generated_split"]).strip().lower(),
                str(row["generated_split"]).strip().lower(),
            )
            split_counts[split] += 1
            if split == source_split:
                rows_by_class[str(row["canonical_class"])].append(row)
    if not rows_by_class:
        raise RuntimeError(f"Manifest tidak memiliki crop split {source_split}")
    if normal_class not in rows_by_class:
        raise ValueError(f"Kelas normal tidak ditemukan: {normal_class}")

    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    source_audits = audit_payload.get("source_audits")
    if not isinstance(source_audits, dict) or not source_audits:
        raise ValueError(f"source_audits tidak tersedia: {audit_path}")
    archive_offsets: dict[tuple[str, str], tuple[int, int]] = {}
    global_offset = 0
    for dataset_name, source_audit in source_audits.items():
        archive_counts = source_audit.get("archive_counts", {})
        if not isinstance(archive_counts, dict) or not archive_counts:
            raise ValueError(
                f"archive_counts tidak tersedia untuk {dataset_name}"
            )
        dataset_images = 0
        for archive_split, split_audit in archive_counts.items():
            split_images = int(split_audit["images"])
            archive_offsets[(str(dataset_name), str(archive_split))] = (
                global_offset + dataset_images,
                split_images,
            )
            dataset_images += split_images
        declared_images = int(source_audit.get("images", dataset_images))
        if declared_images != dataset_images:
            raise ValueError(
                f"Jumlah image {dataset_name} tidak konsisten: "
                f"{declared_images} vs {dataset_images}"
            )
        global_offset += dataset_images

    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    selected_by_class = Counter()
    for class_name in sorted(rows_by_class):
        capacity = (
            max_normal_assets
            if class_name == normal_class
            else max_defect_assets_per_class
        )
        candidate_capacity = capacity * candidate_multiplier
        by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows_by_class[class_name]:
            by_parent[str(row["source_identity"])].append(row)
        parent_ids = sorted(by_parent)
        rng.shuffle(parent_ids)
        class_rows: list[dict[str, str]] = []
        for parent_id in parent_ids:
            candidates = by_parent[parent_id]
            class_rows.append(rng.choice(candidates))
            if len(class_rows) >= candidate_capacity:
                break
        if len(class_rows) < candidate_capacity:
            used_paths = {row["crop_path"] for row in class_rows}
            remaining = [
                row
                for row in rows_by_class[class_name]
                if row["crop_path"] not in used_paths
            ]
            rng.shuffle(remaining)
            class_rows.extend(remaining[: candidate_capacity - len(class_rows)])
        selected.extend(class_rows)
        selected_by_class[class_name] = len(class_rows)

    wanted = {str(row["crop_path"]): row for row in selected}
    payloads: dict[str, bytes] = {}
    shard_paths = sorted(shard_root.glob("*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"Shard TAR tidak ditemukan: {shard_root}")
    shard_ranges = []
    for shard_path in shard_paths:
        match = re.fullmatch(
            r"crop_shard_(\d+)_(\d+)\.tar", shard_path.name
        )
        if match is None:
            raise ValueError(f"Nama shard tidak dikenal: {shard_path.name}")
        shard_ranges.append((int(match.group(1)), int(match.group(2)), shard_path))
    wanted_by_shard: dict[Path, set[str]] = defaultdict(set)
    for crop_path, row in wanted.items():
        try:
            image_id = int(row["image_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"image_id tidak valid untuk {crop_path}") from error
        archive_key = (str(row["dataset"]), str(row["archive_split"]))
        archive_position = archive_offsets.get(archive_key)
        if archive_position is None:
            raise ValueError(
                f"Dataset/archive_split tidak ada dalam audit: {archive_key}"
            )
        archive_offset, archive_images = archive_position
        if not 0 <= image_id < archive_images:
            raise ValueError(
                f"image_id={image_id} di luar rentang {archive_key}: "
                f"0-{archive_images - 1}"
            )
        # Shard ranges use a one-based global source-image ordinal. COCO
        # image_id restarts from zero for every dataset/archive split.
        shard_ordinal = archive_offset + image_id + 1
        matched_shard = next(
            (
                shard_path
                for first, last, shard_path in shard_ranges
                if first <= shard_ordinal <= last
            ),
            None,
        )
        if matched_shard is None:
            raise FileNotFoundError(
                f"Shard untuk image_id={image_id} "
                f"(ordinal={shard_ordinal}) tidak ditemukan: {crop_path}"
            )
        wanted_by_shard[matched_shard].add(crop_path)
    cache_root = (
        Path(shard_cache_root).expanduser().resolve()
        if shard_cache_root is not None
        else None
    )
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
    print(
        f"INDEX SNI CROP: {len(wanted)} crop {source_split} dipilih dari "
        f"{len(wanted_by_shard)}/{len(shard_paths)} shard",
        flush=True,
    )
    relevant_shards = sorted(wanted_by_shard)
    for shard_index, source_shard in enumerate(relevant_shards, 1):
        shard_path = source_shard
        if cache_root is not None:
            cached = cache_root / source_shard.name
            if not cached.is_file() or cached.stat().st_size != source_shard.stat().st_size:
                print(
                    f"  cache shard {shard_index}/{len(relevant_shards)}: "
                    f"{source_shard.name} ({source_shard.stat().st_size / 1e6:.1f} MB)",
                    flush=True,
                )
                temporary = cached.with_suffix(cached.suffix + ".part")
                shutil.copy2(source_shard, temporary)
                temporary.replace(cached)
            shard_path = cached
        targets = wanted_by_shard[source_shard]
        with tarfile.open(shard_path, "r") as archive:
            for member in archive:
                if (
                    member.name not in targets
                    or member.name in payloads
                    or not member.isfile()
                ):
                    continue
                extracted = archive.extractfile(member)
                if extracted is not None:
                    payloads[member.name] = extracted.read()
        print(
            f"  baca shard {shard_index}/{len(relevant_shards)} | "
            f"crop ditemukan {len(payloads)}/{len(wanted)}",
            flush=True,
        )
    missing_paths = sorted(set(wanted) - set(payloads))
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} crop terpilih tidak ditemukan dalam shard; "
            f"contoh: {missing_paths[0]}"
        )

    assets: list[dict] = []
    failures: list[str] = []
    accepted_by_class = Counter()
    for index, row in enumerate(selected, 1):
        crop_path = str(row["crop_path"])
        class_name = str(row["canonical_class"])
        target_assets = (
            max_normal_assets
            if class_name == normal_class
            else max_defect_assets_per_class
        )
        if accepted_by_class[class_name] >= target_assets:
            continue
        try:
            with Image.open(io.BytesIO(payloads[crop_path])) as source_image:
                image = source_image.convert("RGBA")
        except OSError as error:
            failures.append(f"{crop_path}: {error}")
            continue
        item, failure = _write_asset(
            output_root,
            class_name,
            str(row["crop_sha256"]),
            source_split,
            Path(crop_path),
            image,
            mask_threshold,
            padding,
            minimum_fraction,
            maximum_fraction,
            preferred_center=(image.width / 2.0, image.height / 2.0),
            explicit_mask=None,
            border_trim_fraction=0.03,
            background_model="spatial_prototypes",
        )
        if item:
            item["source_parent_id"] = str(row["source_identity"])
            item["source_dataset"] = str(row["dataset"])
            assets.append(item)
            accepted_by_class[class_name] += 1
        if failure:
            failures.append(failure)
        if index % 100 == 0 or index == len(selected):
            print(
                f"  mask crop: {index}/{len(selected)} | valid={len(assets)}",
                flush=True,
            )
    return _finalize_library(
        output_root,
        assets,
        failures,
        {
            "type": "sni_crop_manifest",
            "root": str(dataset_root),
            "manifest": str(manifest_path),
            "source_split": source_split,
            "normal_class": normal_class,
            "selected_by_class": dict(sorted(selected_by_class.items())),
            "accepted_by_class_before_dedup": dict(
                sorted(accepted_by_class.items())
            ),
            "split_counts": dict(sorted(split_counts.items())),
            "max_normal_assets": max_normal_assets,
            "max_defect_assets_per_class": max_defect_assets_per_class,
            "candidate_multiplier": candidate_multiplier,
            "relevant_shards": len(relevant_shards),
            "global_source_images": global_offset,
            "shard_cache_root": str(cache_root) if cache_root is not None else None,
            "seed": seed,
            "mask_provenance": (
                "estimated from stored JPEG crop with spatial border "
                f"prototypes and RGB distance threshold {mask_threshold:g}; "
                "original polygon/alpha is not present in crop package"
            ),
            "mask_config": {
                "background_model": "spatial_prototypes",
                "distance_threshold": float(mask_threshold),
                "border_trim_fraction": 0.03,
                "matte": "inward_feathered_premultiplied",
            },
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
    label_line_cache: dict[Path, list[str]] = {}
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
        relative = image_path.relative_to(layout.splits[source_split][0])
        label_path = (layout.splits[source_split][1] / relative).with_suffix(".txt")
        if label_path not in label_line_cache:
            label_line_cache[label_path] = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(label_line_cache) > 64:
                label_line_cache.pop(next(iter(label_line_cache)))
        fields = label_line_cache[label_path][box_index].split()
        values = [float(value) for value in fields[1:]]
        explicit_mask = None
        if len(values) > 4:
            polygon = [
                (
                    values[offset] * image.width - pixel_box[0],
                    values[offset + 1] * image.height - pixel_box[1],
                )
                for offset in range(0, len(values), 2)
            ]
            polygon_image = Image.new("L", crop.size, 0)
            ImageDraw.Draw(polygon_image).polygon(polygon, fill=255)
            explicit_mask = np.asarray(polygon_image, dtype=np.uint8) > 0
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
            (
                box.x_center * image.width - pixel_box[0],
                box.y_center * image.height - pixel_box[1],
            ),
            explicit_mask,
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
            "type": "yolo_detection_or_segmentation",
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
                source_parent_id=(
                    str(row["source_parent_id"])
                    if row.get("source_parent_id") is not None
                    else None
                ),
                intrinsic_aspect_ratio=(
                    float(row["intrinsic_aspect_ratio"])
                    if row.get("intrinsic_aspect_ratio") is not None
                    else None
                ),
            )
        )
    if not cutouts:
        raise RuntimeError("Object library tidak memiliki aset train/unspecified")
    return classes, cutouts, {"manifest": payload, "rejected_splits": dict(rejected)}


def backfill_intrinsic_geometry(
    path: str | Path,
    *,
    progress_every: int = 100,
) -> dict[str, int]:
    """Persist rotation-invariant silhouette capability in an old library.

    Older object libraries remain valid, but their manifests predate the
    `intrinsic_aspect_ratio` field. Persisting it once avoids rescanning every
    PNG independently in the A1 and A2 generation subprocesses.
    """
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "object_library.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = payload.get("assets", [])
    missing = [row for row in assets if row.get("intrinsic_aspect_ratio") is None]
    if not missing:
        return {"assets": len(assets), "profiled": 0, "remaining": 0}

    library_root = manifest_path.parent
    total = len(missing)
    print(f"PROFIL GEOMETRI CUTOUT: 0/{total}")
    for index, row in enumerate(missing, start=1):
        image_path = library_root / str(row["image"])
        with Image.open(image_path) as source:
            alpha = np.asarray(
                source.convert("RGBA").getchannel("A"), dtype=np.uint8
            )
        major, minor, _ = principal_mask_geometry(alpha >= 32)
        row["intrinsic_aspect_ratio"] = major / max(minor, 1e-6)
        if index % max(progress_every, 1) == 0 or index == total:
            print(f"PROFIL GEOMETRI CUTOUT: {index}/{total}")

    payload["geometry_profile"] = {
        "method": "principal_mask_extent.v1",
        "profiled_assets": total,
    }
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(manifest_path)
    return {"assets": len(assets), "profiled": total, "remaining": 0}
