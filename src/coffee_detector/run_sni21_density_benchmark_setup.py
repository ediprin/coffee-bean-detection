from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .audit_vadcp import audit_vadcp_dataset
from .generate_vadcp_dataset import generate_vadcp_dataset
from .prepare_sni_fullscene import SNI21_CLASSES
from .sni_crop_manifest import build_sni_crop_calibration
from .vadcp.library import (
    load_object_library,
    prepare_sni_crop_manifest_library,
)
from .vadcp.profile import save_scene_calibration


DENSITY_LADDER = {
    "B0": (1, 5),
    "B1": (10, 25),
    "B2": (50, 100),
    "B3": (220, 300),
}

STAGE_VARIANTS = {
    "core": (("empirical", "mild"),),
    "clear": (("empirical", "clear"),),
    "balanced": (("balanced", "mild"),),
    "severe": (("empirical", "severe"),),
    "all": (
        ("empirical", "clear"),
        ("empirical", "mild"),
        ("empirical", "severe"),
        ("balanced", "mild"),
    ),
}


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} bukan JSON object: {path}")
    return payload


def audit_validation_library_provenance(
    crop_dataset_root: str | Path,
    library_root: str | Path,
) -> dict:
    """Verify selected validation assets have no train/test identity overlap.

    This reads manifest metadata only. It does not open train, validation, or
    test source images.
    """
    crop_dataset_root = Path(crop_dataset_root).expanduser().resolve()
    library_root = Path(library_root).expanduser().resolve()
    manifest_path = crop_dataset_root / "manifest.csv"
    library_path = library_root / "object_library.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest crop tidak ditemukan: {manifest_path}")
    library = _load_json(library_path, "Object library validation")

    parent_splits: dict[str, set[str]] = defaultdict(set)
    crop_splits: dict[str, set[str]] = defaultdict(set)
    class_by_crop: dict[str, str] = {}
    required = {
        "generated_split",
        "source_identity",
        "crop_sha256",
        "canonical_class",
    }
    rows = 0
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                "Kolom manifest belum lengkap: " + ", ".join(missing)
            )
        for row in reader:
            rows += 1
            split = str(row["generated_split"]).strip().lower()
            parent = str(row["source_identity"])
            digest = str(row["crop_sha256"])
            parent_splits[parent].add(split)
            crop_splits[digest].add(split)
            class_by_crop[digest] = str(row["canonical_class"])

    errors = []
    selected_parents = set()
    selected_crops = set()
    selected_assets = set()
    selected_classes = Counter()
    for row in library.get("assets", []):
        asset_id = str(row["asset_id"])
        source_split = str(row.get("source_split", ""))
        source_id = str(row["source_id"])
        parent_id = str(row.get("source_parent_id", ""))
        class_name = str(row["class_name"])
        selected_assets.add(asset_id)
        selected_crops.add(source_id)
        selected_parents.add(parent_id)
        selected_classes[class_name] += 1
        if source_split != "val":
            errors.append(f"{asset_id}: source_split={source_split}, expected=val")
        if crop_splits.get(source_id) != {"val"}:
            errors.append(
                f"{asset_id}: crop identity muncul pada "
                f"{sorted(crop_splits.get(source_id, set()))}"
            )
        if parent_splits.get(parent_id) != {"val"}:
            errors.append(
                f"{asset_id}: parent identity muncul pada "
                f"{sorted(parent_splits.get(parent_id, set()))}"
            )
        if class_by_crop.get(source_id) != class_name:
            errors.append(
                f"{asset_id}: kelas library/manifest berbeda "
                f"{class_name}/{class_by_crop.get(source_id)}"
            )
    classes = {
        int(key): str(value)
        for key, value in library.get("classes", {}).items()
    }
    expected = {index: name for index, name in enumerate(SNI21_CLASSES)}
    if classes != expected:
        errors.append("Urutan/nama kelas library bukan canonical SNI-21")
    report = {
        "format": "coffee_detector.sni21_val_library_audit.v1",
        "manifest_rows": rows,
        "library_assets": len(selected_assets),
        "source_parents": len(selected_parents),
        "assets_by_class": dict(sorted(selected_classes.items())),
        "selected_source_split": "val",
        "parent_identity_overlap": {
            split: sum(split in parent_splits[parent] for parent in selected_parents)
            for split in ("train", "test")
        },
        "crop_identity_overlap": {
            split: sum(split in crop_splits[digest] for digest in selected_crops)
            for split in ("train", "test")
        },
        "test_images_opened": False,
        "errors": errors[:200],
        "error_count": len(errors),
        "safe_for_development_benchmark": not errors,
    }
    return report


def _reuse_limits(
    cutouts,
    calibration,
    *,
    scenes: int,
    density: tuple[int, int],
    balanced: bool,
    requested_asset_limit: int | None,
    requested_parent_limit: int | None,
) -> tuple[int, int, dict]:
    maximum_instances = scenes * density[1]
    by_class = Counter(item.class_id for item in cutouts)
    parents_by_class: dict[int, set[str]] = defaultdict(set)
    for item in cutouts:
        if item.source_parent_id is not None:
            parents_by_class[item.class_id].add(item.source_parent_id)
    parent_count = len(
        {
            item.source_parent_id
            for item in cutouts
            if item.source_parent_id is not None
        }
    )
    if balanced:
        probabilities = {
            class_id: 1.0 / len(by_class) for class_id in by_class
        }
    else:
        mass = sum(calibration.class_probabilities.values())
        probabilities = {
            class_id: calibration.class_probabilities.get(class_id, 0.0)
            / max(mass, 1e-12)
            for class_id in by_class
        }
    class_expected = {
        class_id: maximum_instances * probabilities.get(class_id, 0.0)
        for class_id in by_class
    }
    # A mean-only capacity bound fails for rare classes: a valid empirical
    # draw can exceed its expectation by several instances even in B0. Use a
    # conservative binomial envelope before distributing capacity over source
    # identities. The additional factor absorbs geometry-driven preference for
    # particular cutouts without removing the explicit reuse ceiling.
    class_upper_instances = {
        class_id: math.ceil(
            expected
            + 8.0
            * math.sqrt(
                max(
                    expected
                    * (1.0 - probabilities.get(class_id, 0.0)),
                    0.0,
                )
            )
            + 2.0
        )
        for class_id, expected in class_expected.items()
    }
    class_lower_bounds = {
        class_id: math.ceil(
            class_upper_instances[class_id] / max(asset_count, 1)
        )
        for class_id, asset_count in by_class.items()
    }
    # An asset is normally selected at most once per scene. Use the scene count
    # as the automatic ceiling so concentrated rare classes cannot terminate
    # generation midway. Actual reuse remains recorded and is handled as a
    # grouped source-identity dependency during uncertainty estimation.
    automatic_asset = max(
        2,
        scenes,
        max(class_lower_bounds.values(), default=1) * 2,
    )
    parent_class_lower_bounds = {
        class_id: math.ceil(
            class_upper_instances[class_id] / max(len(parents), 1)
        )
        for class_id, parents in parents_by_class.items()
    }
    automatic_parent = max(
        2,
        maximum_instances,
        math.ceil(maximum_instances / max(parent_count, 1)) * 4,
        max(parent_class_lower_bounds.values(), default=1) * 2,
    )
    asset_limit = (
        requested_asset_limit
        if requested_asset_limit is not None
        else automatic_asset
    )
    parent_limit = (
        requested_parent_limit
        if requested_parent_limit is not None
        else automatic_parent
    )
    if asset_limit < max(class_lower_bounds.values(), default=1):
        raise ValueError(
            "max_asset_reuse lebih kecil dari lower bound kapasitas kelas"
        )
    if parent_limit * max(parent_count, 1) < maximum_instances:
        raise ValueError(
            "max_parent_reuse tidak cukup untuk jumlah instance maksimum"
        )
    return asset_limit, parent_limit, {
        "maximum_instances": maximum_instances,
        "assets": len(cutouts),
        "parents": parent_count,
        "class_lower_bounds": {
            str(key): value for key, value in sorted(class_lower_bounds.items())
        },
        "class_upper_instances": {
            str(key): value
            for key, value in sorted(class_upper_instances.items())
        },
        "parent_class_lower_bounds": {
            str(key): value
            for key, value in sorted(parent_class_lower_bounds.items())
        },
        "automatic_asset_limit": automatic_asset,
        "automatic_parent_limit": automatic_parent,
        "reuse_policy": (
            "Natural generation ceilings prevent rare concentrated source "
            "identities from aborting synthesis. Report actual reuse and use "
            "source-parent grouped uncertainty for model comparisons."
        ),
        "selected_asset_limit": asset_limit,
        "selected_parent_limit": parent_limit,
    }


def _write_resampling_units(arm_root: Path) -> dict:
    manifest = _load_json(
        arm_root / "metadata" / "generation_manifest.json",
        "Manifest benchmark",
    )
    split = str(manifest["synthetic_split"])
    metadata = _load_json(
        arm_root / "metadata" / f"instances_synthetic_{split}.json",
        "Metadata benchmark",
    )
    image_by_id = {int(row["id"]): row for row in metadata["images"]}
    rows_by_image: dict[int, list[dict]] = defaultdict(list)
    for row in metadata["annotations"]:
        rows_by_image[int(row["image_id"])].append(row)
    asset_scenes: dict[str, set[str]] = defaultdict(set)
    parent_scenes: dict[str, set[str]] = defaultdict(set)
    scenes = []
    for image_id, image in sorted(image_by_id.items()):
        scene_id = str(image["generation_seed"])
        rows = rows_by_image.get(image_id, [])
        assets = sorted({str(row["source_asset_id"]) for row in rows})
        parents = sorted(
            {
                str(row["source_parent_id"])
                for row in rows
                if row.get("source_parent_id") is not None
            }
        )
        scenes.append(
            {
                "scene_id": scene_id,
                "image_id": image_id,
                "file_name": image["file_name"],
                "source_asset_ids": assets,
                "source_parent_ids": parents,
            }
        )
        for asset_id in assets:
            asset_scenes[asset_id].add(scene_id)
        for parent_id in parents:
            parent_scenes[parent_id].add(scene_id)
    payload = {
        "format": "coffee_detector.sni21_resampling_units.v1",
        "scenes": scenes,
        "source_asset_clusters": {
            key: sorted(value) for key, value in sorted(asset_scenes.items())
        },
        "source_parent_clusters": {
            key: sorted(value) for key, value in sorted(parent_scenes.items())
        },
        "performance_uncertainty": {
            "scene_bootstrap": (
                "Resample scene_id and recompute the paired metric delta."
            ),
            "grouped_identity": (
                "Use source-identity cluster resampling/weighting because repeated "
                "placements are not independent observations."
            ),
            "computed_now": False,
            "reason": "Model predictions do not exist during dataset generation.",
        },
    }
    output = arm_root / "metadata" / "resampling_units.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "path": str(output),
        "scenes": len(scenes),
        "asset_clusters": len(asset_scenes),
        "parent_clusters": len(parent_scenes),
    }


def run_sni21_density_benchmark_setup(
    crop_dataset_root: str | Path,
    output_root: str | Path,
    *,
    stage: str = "core",
    scenes_per_condition: int = 200,
    seed: int = 42,
    canvas_size: int = 1024,
    object_scale: tuple[float, float] = (0.025, 0.055),
    max_asset_reuse: int | None = None,
    max_parent_reuse: int | None = None,
    object_library_root: str | Path | None = None,
    shard_cache_root: str | Path | None = None,
) -> dict:
    if stage not in STAGE_VARIANTS:
        raise ValueError(f"Stage tidak dikenal: {stage}")
    if scenes_per_condition <= 0:
        raise ValueError("scenes_per_condition harus positif")
    crop_root = Path(crop_dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    library_root = (
        Path(object_library_root).expanduser().resolve()
        if object_library_root is not None
        else output_root / "val_object_library"
    )
    if not (library_root / "object_library.json").is_file():
        print("[1/4] Membuat object library dari validation identities...", flush=True)
        prepare_sni_crop_manifest_library(
            crop_root,
            library_root,
            source_split="val",
            max_normal_assets=100_000,
            max_defect_assets_per_class=100_000,
            seed=seed,
            shard_cache_root=shard_cache_root,
        )
    else:
        print(f"[1/4] Reuse validation object library: {library_root}", flush=True)

    provenance = audit_validation_library_provenance(crop_root, library_root)
    provenance_path = output_root / "validation_library_audit.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not provenance["safe_for_development_benchmark"]:
        raise RuntimeError(
            f"Validation library gagal audit: {provenance_path}"
        )
    print(
        f"[2/4] Provenance aman: {provenance['library_assets']} aset val, "
        "test image tidak dibuka.",
        flush=True,
    )

    names, calibration, empirical = build_sni_crop_calibration(
        crop_root,
        policy="source_empirical",
        source_split="val",
        objects_min=1,
        objects_max=300,
    )
    expected_names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    if names != expected_names:
        raise ValueError("Kalibrasi validation bukan canonical SNI-21")
    profile_path = save_scene_calibration(
        calibration, output_root / "val_scene_profile.json"
    )
    _, cutouts, _ = load_object_library(library_root, train_only=False)
    cutouts = [item for item in cutouts if item.source_split == "val"]

    arms = {}
    variants = STAGE_VARIANTS[stage]
    total_arms = len(DENSITY_LADDER) * len(variants)
    arm_index = 0
    for density_code, density in DENSITY_LADDER.items():
        for prior, visibility in variants:
            arm_index += 1
            balanced = prior == "balanced"
            asset_limit, parent_limit, reuse_plan = _reuse_limits(
                cutouts,
                calibration,
                scenes=scenes_per_condition,
                density=density,
                balanced=balanced,
                requested_asset_limit=max_asset_reuse,
                requested_parent_limit=max_parent_reuse,
            )
            arm_name = f"{density_code}_{prior}_{visibility}"
            arm_root = output_root / arm_name
            manifest_path = arm_root / "metadata" / "generation_manifest.json"
            if manifest_path.is_file():
                manifest = _load_json(manifest_path, f"Manifest {arm_name}")
                manifest_spec = manifest.get("spec", {})
                expected = {
                    "artifact_role": "development_benchmark",
                    "library_source_split": "val",
                    "synthetic_split": "val",
                    "synthetic_images": scenes_per_condition,
                    "seed": seed,
                    "mode": "visibility",
                    "preset": "sni_spread",
                }
                conflicts = {
                    key: (manifest.get(key), value)
                    for key, value in expected.items()
                    if manifest.get(key) != value
                }
                expected_spec = {
                    "object_range": list(density),
                    "object_scale": list(object_scale),
                    "class_balanced": balanced,
                    "target_bin_weights": {
                        "clear": 1.0 if visibility == "clear" else 0.0,
                        "mild": 1.0 if visibility == "mild" else 0.0,
                        "severe": 1.0 if visibility == "severe" else 0.0,
                    },
                }
                conflicts.update(
                    {
                        f"spec.{key}": (manifest_spec.get(key), value)
                        for key, value in expected_spec.items()
                        if manifest_spec.get(key) != value
                    }
                )
                expected_reuse = {
                    "asset_reuse.limit": asset_limit,
                    "parent_reuse.limit": parent_limit,
                }
                actual_reuse = {
                    "asset_reuse.limit": manifest.get("asset_reuse", {}).get(
                        "limit"
                    ),
                    "parent_reuse.limit": manifest.get("parent_reuse", {}).get(
                        "limit"
                    ),
                }
                conflicts.update(
                    {
                        key: (actual_reuse[key], value)
                        for key, value in expected_reuse.items()
                        if actual_reuse[key] != value
                    }
                )
                if conflicts:
                    raise RuntimeError(
                        f"Output {arm_name} konflik: "
                        + json.dumps(conflicts, ensure_ascii=False)
                    )
                print(
                    f"[3/4] Reuse {arm_name} ({arm_index}/{total_arms})",
                    flush=True,
                )
            else:
                if arm_root.exists() and any(arm_root.iterdir()):
                    print(
                        f"[3/4] Hapus output parsial {arm_name}, lalu generate "
                        "ulang.",
                        flush=True,
                    )
                    shutil.rmtree(arm_root)
                print(
                    f"[3/4] Generate {arm_name} ({arm_index}/{total_arms}) | "
                    f"{density[0]}-{density[1]} objek",
                    flush=True,
                )
                manifest = generate_vadcp_dataset(
                    None,
                    library_root,
                    arm_root,
                    synthetic_images=scenes_per_condition,
                    seed=seed,
                    mode="visibility",
                    preset="sni_spread",
                    canvas_size=canvas_size,
                    object_range=density,
                    object_scale=object_scale,
                    include_real_train=False,
                    materialize_real_splits=False,
                    scene_profile=calibration,
                    target_names=names,
                    artifact_role="development_benchmark",
                    library_source_split="val",
                    synthetic_split="val",
                    class_balanced=balanced,
                    target_visibility_bin=visibility,
                    max_asset_reuse=asset_limit,
                    max_parent_reuse=parent_limit,
                )
            audit = audit_vadcp_dataset(
                arm_root, arm_root / "metadata" / "vadcp_audit.json"
            )
            if not audit["safe_for_training"]:
                raise RuntimeError(
                    f"Audit benchmark gagal: {arm_root / 'metadata/vadcp_audit.json'}"
                )
            resampling = _write_resampling_units(arm_root)
            arms[arm_name] = {
                "root": str(arm_root),
                "density": list(density),
                "prior": prior,
                "visibility": visibility,
                "manifest": str(manifest_path),
                "audit": str(arm_root / "metadata" / "vadcp_audit.json"),
                "reuse_plan": reuse_plan,
                "resampling_units": resampling,
            }

    summary = {
        "format": "coffee_detector.sni21_density_benchmark_setup.v1",
        "protocol": "docs/SNI21_DENSITY_BENCHMARK_PROTOCOL.md",
        "stage": stage,
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "crop_dataset_root": str(crop_root),
        "output_root": str(output_root),
        "validation_object_library": str(library_root),
        "validation_library_audit": str(provenance_path),
        "scene_profile": str(profile_path),
        "empirical_source_prior": empirical,
        "density_ladder": {
            key: list(value) for key, value in DENSITY_LADDER.items()
        },
        "constant_object_scale": list(object_scale),
        "constant_scale_reason": (
            "Hold object scale fixed across B0-B3 so density count is not "
            "confounded with a changing scale distribution."
        ),
        "scenes_per_condition": scenes_per_condition,
        "seed": seed,
        "arms": arms,
        "ready_for_evaluation": bool(arms),
        "performance_uncertainty_computed": False,
        "next_action": (
            "Evaluate the same frozen checkpoints; then compute paired scene "
            "and grouped-source uncertainty from predictions."
        ),
    }
    summary_path = output_root / f"setup_{stage}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== HELD-OUT DENSITY BENCHMARK SIAP ===", flush=True)
    print("TRAINING        : TIDAK DIJALANKAN", flush=True)
    print("TEST IMAGE      : TIDAK DIAKSES", flush=True)
    print("DEVELOPMENT ONLY:", True, flush=True)
    print("SUMMARY         :", summary_path, flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bangun benchmark sintetis B0-B3 dari validation identities; "
            "tidak training dan tidak membuka test image."
        )
    )
    parser.add_argument("--crop-dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_VARIANTS),
        default="core",
    )
    parser.add_argument("--scenes-per-condition", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--scale-min", type=float, default=0.025)
    parser.add_argument("--scale-max", type=float, default=0.055)
    parser.add_argument("--max-asset-reuse", type=int)
    parser.add_argument("--max-parent-reuse", type=int)
    parser.add_argument("--object-library-root")
    parser.add_argument("--shard-cache-root")
    args = parser.parse_args()
    run_sni21_density_benchmark_setup(
        args.crop_dataset_root,
        args.output_root,
        stage=args.stage,
        scenes_per_condition=args.scenes_per_condition,
        seed=args.seed,
        canvas_size=args.canvas_size,
        object_scale=(args.scale_min, args.scale_max),
        max_asset_reuse=args.max_asset_reuse,
        max_parent_reuse=args.max_parent_reuse,
        object_library_root=args.object_library_root,
        shard_cache_root=args.shard_cache_root,
    )


if __name__ == "__main__":
    main()
