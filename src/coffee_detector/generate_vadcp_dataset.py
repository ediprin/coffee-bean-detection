from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from .dataset import IMAGE_SUFFIXES, discover_layout
from .vadcp.compositor import CompositionSpec, compose_scene, load_background
from .vadcp.library import load_object_library
from .vadcp.masks import binary_mask_rle, mask_bbox
from .vadcp.profile import (
    SceneCalibration,
    build_scene_calibration,
    calibration_summary,
    load_scene_calibration,
)
from .vadcp.types import Cutout


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _remap_cutouts(
    cutouts: list[Cutout],
    target_names: dict[int, str],
) -> list[Cutout]:
    target_by_name: dict[str, tuple[int, str]] = {}
    for class_id, name in target_names.items():
        key = _normalized_name(name)
        if key in target_by_name:
            raise ValueError(f"Nama kelas ambigu setelah normalisasi: {name}")
        target_by_name[key] = (class_id, name)
    remapped = []
    missing = set()
    for item in cutouts:
        match = target_by_name.get(_normalized_name(item.class_name))
        if match is None:
            missing.add(item.class_name)
            continue
        remapped.append(replace(item, class_id=match[0], class_name=match[1]))
    if missing:
        raise ValueError(
            "Kelas object library tidak ditemukan pada data nyata: "
            + ", ".join(sorted(missing))
        )
    present = {item.class_id for item in remapped}
    absent = [target_names[index] for index in sorted(target_names) if index not in present]
    if absent:
        raise ValueError(
            "Object library belum mencakup semua kelas data nyata: " + ", ".join(absent)
        )
    return remapped


def _materialize(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_real_split(
    image_root: Path,
    label_root: Path,
    output_root: Path,
    split: str,
) -> int:
    image_paths = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    total = len(image_paths)
    started = time.perf_counter()
    print(f"REAL SPLIT {split}: mulai 0/{total}", flush=True)
    count = 0
    for image_path in image_paths:
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Label real tidak ditemukan: {label_path}")
        _materialize(image_path, output_root / split / "images" / relative)
        _materialize(label_path, output_root / split / "labels" / relative.with_suffix(".txt"))
        count += 1
        if count % 500 == 0 or count == total:
            elapsed = time.perf_counter() - started
            print(
                f"REAL SPLIT {split}: {count}/{total} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
    return count


def _background_paths(root: str | Path | None) -> list[Path]:
    if root is None:
        return []
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder background tidak ditemukan: {root}")
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _yolo_line(class_id: int, bbox: tuple[int, int, int, int], size: tuple[int, int]) -> str:
    x, y, width, height = bbox
    image_width, image_height = size
    return (
        f"{class_id} "
        f"{(x + width / 2) / image_width:.8f} "
        f"{(y + height / 2) / image_height:.8f} "
        f"{width / image_width:.8f} "
        f"{height / image_height:.8f}"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calibrated_canvas_size(
    long_side: int,
    calibration: SceneCalibration,
) -> tuple[int, int]:
    if long_side <= 0:
        raise ValueError("canvas_size harus positif")
    ratio = float(np.median(calibration.canvas_width_height_ratios))
    if ratio >= 1.0:
        return long_side, max(1, int(round(long_side / ratio)))
    return max(1, int(round(long_side * ratio))), long_side


def generate_vadcp_dataset(
    real_data_root: str | Path,
    object_library: str | Path,
    output_root: str | Path,
    *,
    synthetic_images: int = 2000,
    seed: int = 42,
    mode: str = "visibility",
    background_root: str | Path | None = None,
    canvas_size: int = 640,
    object_range: tuple[int, int] = (12, 30),
    object_scale: tuple[float, float] = (0.08, 0.18),
    minimum_visibility: float = 0.10,
    include_real_train: bool = True,
    use_shadows: bool = True,
    scene_profile: str | Path | SceneCalibration | None = None,
) -> dict:
    if synthetic_images <= 0:
        raise ValueError("synthetic_images harus positif")
    layout = discover_layout(real_data_root)
    missing_splits = sorted({"train", "val", "test"} - set(layout.splits))
    if missing_splits:
        raise FileNotFoundError(
            "Dataset nyata harus memiliki train/val/test: " + ", ".join(missing_splits)
        )
    _, cutouts, library_info = load_object_library(object_library, train_only=True)
    cutouts = _remap_cutouts(cutouts, layout.names)
    invalid_sources = sorted(
        {item.source_split for item in cutouts if item.source_split not in {"train", "unspecified"}}
    )
    if invalid_sources:
        raise RuntimeError(
            "Synthetic train memuat aset non-train: " + ", ".join(invalid_sources)
        )

    if isinstance(scene_profile, SceneCalibration):
        calibration = scene_profile
        scene_profile_path = None
    elif scene_profile is not None:
        scene_profile_path = Path(scene_profile).expanduser().resolve()
        calibration = load_scene_calibration(scene_profile_path)
    else:
        scene_profile_path = None
        print("KALIBRASI REAL TRAIN: menghitung prior scene...", flush=True)
        calibration = build_scene_calibration(layout, split="train", seed=seed)

    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output VA-DCP tidak kosong: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    real_counts = {}
    for split in ("train", "val", "test"):
        if split == "train" and not include_real_train:
            (output_root / split / "images").mkdir(parents=True, exist_ok=True)
            (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
            real_counts[split] = 0
            continue
        image_root, label_root = layout.splits[split]
        real_counts[split] = _copy_real_split(
            image_root, label_root, output_root, split
        )
        print(
            f"REAL SPLIT {split}: {real_counts[split]} gambar dimaterialisasi",
            flush=True,
        )

    calibrated_canvas = _calibrated_canvas_size(canvas_size, calibration)
    print(f"CANVAS SINTETIS: {calibrated_canvas[0]}x{calibrated_canvas[1]}", flush=True)
    spec = CompositionSpec(
        canvas_size=calibrated_canvas,
        object_range=object_range,
        object_scale=object_scale,
        minimum_visibility=minimum_visibility,
        mode=mode,
        use_shadows=use_shadows,
    )
    backgrounds = _background_paths(background_root)
    train_images = output_root / "train" / "images"
    train_labels = output_root / "train" / "labels"
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    annotations = []
    images = []
    annotation_id = 1
    visibility_counts = Counter()
    class_counts = Counter()
    focus_targets = Counter()
    focus_hits = Counter()
    ignored_instances = 0
    scene_modes = Counter()
    repeated_assets = 0
    geometry_targets = 0
    geometry_hits = 0
    geometry_fallbacks = 0

    generation_started = time.perf_counter()
    progress_every = max(1, min(25, synthetic_images // 20 or 1))
    print(
        f"VA-DCP {mode}: mulai 0/{synthetic_images} gambar",
        flush=True,
    )
    for scene_index in range(synthetic_images):
        # A per-scene RNG keeps source selection and backgrounds paired across
        # A1/A2 even though the two placement algorithms consume different
        # numbers of random draws inside a scene.
        scene_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{scene_index}".encode("utf-8")).digest()[:8],
            "big",
        )
        scene_rng = random.Random(scene_seed)
        background_path = scene_rng.choice(backgrounds) if backgrounds else None
        background = load_background(
            background_path, spec.canvas_size, scene_rng, calibration
        )
        scene = compose_scene(
            background, cutouts, spec, scene_rng, calibration=calibration
        )
        stem = f"{mode}_seed{seed}_{scene_index:06d}"
        image_path = train_images / f"{stem}.jpg"
        label_path = train_labels / f"{stem}.txt"
        scene.image.save(image_path, quality=94, subsampling=0)
        label_lines = []
        image_id = scene_index + 1
        if scene.target_visibility_bin:
            focus_targets[scene.target_visibility_bin] += scene.controlled_instances
            focus_hits[scene.target_visibility_bin] += scene.controlled_hits
        scene_modes[scene.scene_mode] += 1
        repeated_assets += scene.repeated_assets
        geometry_targets += scene.geometry_targets
        geometry_hits += scene.geometry_hits
        geometry_fallbacks += scene.geometry_fallbacks
        for instance in scene.instances:
            assert instance.visible_mask is not None
            visible_bbox = mask_bbox(instance.visible_mask)
            full_bbox = mask_bbox(instance.full_mask)
            ignored = (
                instance.visibility_ratio < minimum_visibility
                or visible_bbox is None
                or full_bbox is None
            )
            if ignored:
                ignored_instances += 1
            else:
                label_lines.append(
                    _yolo_line(instance.cutout.class_id, visible_bbox, spec.canvas_size)
                )
                visibility_counts[instance.visibility_bin] += 1
                class_counts[instance.cutout.class_name] += 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": instance.cutout.class_id,
                    "bbox": list(visible_bbox) if visible_bbox else None,
                    "full_bbox": list(full_bbox) if full_bbox else None,
                    "area": int(instance.visible_mask.sum()),
                    "full_area": int(instance.full_mask.sum()),
                    "segmentation": binary_mask_rle(instance.visible_mask),
                    "full_segmentation": binary_mask_rle(instance.full_mask),
                    "iscrowd": 0,
                    "ignore": int(ignored),
                    "source_asset_id": instance.cutout.asset_id,
                    "source_id": instance.cutout.source_id,
                    "source_split": instance.cutout.source_split,
                    "z_order": instance.z_order,
                    "visibility_ratio": instance.visibility_ratio,
                    "visibility_bin": instance.visibility_bin,
                    "is_focus": instance.is_focus,
                    "target_bbox_ratio": instance.target_bbox_ratio,
                    "achieved_bbox_ratio": instance.achieved_bbox_ratio,
                    "intrinsic_aspect_ratio": instance.cutout.intrinsic_aspect_ratio,
                    "geometry_reachable": (
                        max(
                            float(instance.target_bbox_ratio),
                            1.0 / float(instance.target_bbox_ratio),
                        )
                        <= float(instance.cutout.intrinsic_aspect_ratio) / 0.97
                        if instance.target_bbox_ratio is not None
                        and instance.cutout.intrinsic_aspect_ratio is not None
                        else None
                    ),
                    "geometry_log_error": (
                        abs(
                            math.log(
                                float(instance.achieved_bbox_ratio)
                                / float(instance.target_bbox_ratio)
                            )
                        )
                        if instance.target_bbox_ratio is not None
                        and instance.achieved_bbox_ratio is not None
                        else None
                    ),
                }
            )
            annotation_id += 1
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        images.append(
            {
                "id": image_id,
                "file_name": f"train/images/{image_path.name}",
                "width": spec.canvas_size[0],
                "height": spec.canvas_size[1],
                "background": str(background_path) if background_path else "procedural",
                "target_visibility_bin": scene.target_visibility_bin,
                "target_visibility_hit": scene.target_visibility_hit,
                "controlled_instances": scene.controlled_instances,
                "controlled_hits": scene.controlled_hits,
                "scene_mode": scene.scene_mode,
                "repeated_assets": scene.repeated_assets,
                "geometry_targets": scene.geometry_targets,
                "geometry_hits": scene.geometry_hits,
                "geometry_fallbacks": scene.geometry_fallbacks,
                "generation_seed": scene_seed,
                "sha256": _file_sha256(image_path),
            }
        )
        completed = scene_index + 1
        if completed % progress_every == 0 or completed == synthetic_images:
            elapsed = time.perf_counter() - generation_started
            rate = completed / max(elapsed, 1e-8)
            eta = (synthetic_images - completed) / max(rate, 1e-8)
            print(
                f"VA-DCP {mode}: {completed}/{synthetic_images} gambar | "
                f"{rate:.2f} img/s | ETA {eta / 60:.1f} menit",
                flush=True,
            )

    names = {int(index): name for index, name in layout.names.items()}
    metadata = {
        "info": {
            "format": "coffee_detector.vadcp.v2",
            "mode": mode,
            "seed": seed,
            "real_data_root": str(layout.root),
            "object_library": str(Path(object_library).expanduser().resolve()),
            "background_root": str(Path(background_root).expanduser().resolve()) if background_root else None,
            "claim_scope": "Synthetic augmentation; validation and test remain real.",
            "generation_model": "physics-informed 2.5D projected packing",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": names[index]} for index in sorted(names)
        ],
    }
    metadata_path = metadata_dir / "instances_synthetic_train.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    yaml_payload = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    target_hit_rates = {
        name: focus_hits[name] / count if count else None
        for name, count in sorted(focus_targets.items())
    }
    manifest = {
        "format": "coffee_detector.vadcp_manifest.v2",
        "output": str(output_root),
        "mode": mode,
        "seed": seed,
        "synthetic_images": synthetic_images,
        "real_images": real_counts,
        "include_real_train": include_real_train,
        "classes": names,
        "spec": asdict(spec),
        "library_audit": library_info["manifest"].get("audit", {}),
        "background_images": len(backgrounds),
        "procedural_background": not bool(backgrounds),
        "scene_calibration": {
            "path": str(scene_profile_path) if scene_profile_path else None,
            "summary": calibration_summary(calibration),
        },
        "scene_modes": dict(sorted(scene_modes.items())),
        "repeated_assets": repeated_assets,
        "geometry_targets": geometry_targets,
        "geometry_hits": geometry_hits,
        "geometry_target_hit_rate": (
            geometry_hits / geometry_targets if geometry_targets else None
        ),
        "geometry_fallbacks": geometry_fallbacks,
        "instances_by_visibility": dict(sorted(visibility_counts.items())),
        "instances_by_class": dict(sorted(class_counts.items())),
        "ignored_instances": ignored_instances,
        "focus_targets": dict(sorted(focus_targets.items())),
        "focus_target_hit_rate": target_hit_rates,
        "metadata": str(metadata_path),
    }
    manifest_path = metadata_dir / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialisasi real-train + synthetic VA-DCP; val/test tetap real."
    )
    parser.add_argument("--real-data-root", required=True)
    parser.add_argument("--object-library", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--background-root")
    parser.add_argument("--synthetic-images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("naive", "visibility"), default="visibility")
    parser.add_argument("--canvas-size", type=int, default=640)
    parser.add_argument("--objects-min", type=int, default=12)
    parser.add_argument("--objects-max", type=int, default=30)
    parser.add_argument("--scale-min", type=float, default=0.08)
    parser.add_argument("--scale-max", type=float, default=0.18)
    parser.add_argument("--minimum-visibility", type=float, default=0.10)
    parser.add_argument("--synthetic-only-train", action="store_true")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument(
        "--scene-profile",
        help="Profil empiris train nyata dari coffee_detector.profile_vadcp_source.",
    )
    args = parser.parse_args()
    result = generate_vadcp_dataset(
        args.real_data_root,
        args.object_library,
        args.output_root,
        synthetic_images=args.synthetic_images,
        seed=args.seed,
        mode=args.mode,
        background_root=args.background_root,
        canvas_size=args.canvas_size,
        object_range=(args.objects_min, args.objects_max),
        object_scale=(args.scale_min, args.scale_max),
        minimum_visibility=args.minimum_visibility,
        include_real_train=not args.synthetic_only_train,
        use_shadows=not args.no_shadows,
        scene_profile=args.scene_profile,
    )
    print("\n=== VA-DCP DATASET SELESAI ===")
    print(f"Output       : {result['output']}")
    print(f"Mode         : {result['mode']}")
    print(f"Synthetic    : {result['synthetic_images']}")
    print(f"Real         : {result['real_images']}")
    print(f"Visibility   : {result['instances_by_visibility']}")
    print(f"Target hit   : {result['focus_target_hit_rate']}")
    print(f"Scene modes  : {result['scene_modes']}")
    print(f"Repeated     : {result['repeated_assets']}")
    print(f"Geometry hit : {result['geometry_target_hit_rate']}")
    print(f"Geom fallback: {result['geometry_fallbacks']}")
    print(f"Ignored      : {result['ignored_instances']}")


if __name__ == "__main__":
    main()
