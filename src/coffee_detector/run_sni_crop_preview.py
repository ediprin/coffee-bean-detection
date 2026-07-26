from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate_vadcp_dataset import generate_vadcp_dataset
from .run_cutout_visual_audit import run_cutout_visual_audit
from .run_vadcp_visual_audit import run_vadcp_visual_audit
from .sni_crop_manifest import (
    COMPOSITION_POLICIES,
    build_sni_crop_calibration,
)
from .vadcp.library import prepare_sni_crop_manifest_library
from .vadcp.profile import save_scene_calibration


def run_sni_crop_preview(
    crop_dataset_root: str | Path,
    output_root: str | Path,
    *,
    images: int = 4,
    seed: int = 42,
    objects_min: int = 220,
    objects_max: int = 300,
    canvas_size: int = 1024,
    enriched_normal_fraction: float = 0.55,
    max_normal_assets: int = 300,
    max_defect_assets_per_class: int = 60,
    shard_cache_root: str | Path | None = None,
) -> dict:
    """Generate four visual-only SNI composition arms from sharded crop data."""
    if images <= 0:
        raise ValueError("images harus positif")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    library_root = output_root / "object_library"
    library_manifest = library_root / "object_library.json"
    if library_manifest.is_file():
        print(f"[1/6] Object library digunakan ulang: {library_root}", flush=True)
    else:
        if library_root.exists() and any(library_root.iterdir()):
            raise FileExistsError(
                f"Object library parsial/tidak valid: {library_root}"
            )
        print("[1/6] Membuat object library train-only dari shard crop...", flush=True)
        prepare_sni_crop_manifest_library(
            crop_dataset_root,
            library_root,
            source_split="train",
            max_normal_assets=max_normal_assets,
            max_defect_assets_per_class=max_defect_assets_per_class,
            seed=seed,
            shard_cache_root=shard_cache_root,
        )

    print("[2/6] Membuat audit visual tepi cutout...", flush=True)
    cutout_audit = run_cutout_visual_audit(
        library_root,
        output_root / "cutout_visual",
        samples=42,
        seed=seed,
    )

    manifests = {}
    visuals = {}
    policies = {}
    arm_number = 0
    for policy in COMPOSITION_POLICIES:
        names, calibration, policy_report = build_sni_crop_calibration(
            crop_dataset_root,
            policy=policy,
            enriched_normal_fraction=enriched_normal_fraction,
            objects_min=objects_min,
            objects_max=objects_max,
        )
        policy_root = output_root / policy
        policy_root.mkdir(parents=True, exist_ok=True)
        profile_path = save_scene_calibration(
            calibration, policy_root / "scene_profile.json"
        )
        policy_report["scene_profile"] = str(profile_path)
        policies[policy] = policy_report
        for arm, mode in (("A1", "naive"), ("A2", "visibility")):
            arm_number += 1
            key = f"{policy}/{arm}"
            arm_root = policy_root / arm
            print(
                f"[{arm_number + 2}/6] {key}: {images} scene, "
                f"{objects_min}-{objects_max} objek...",
                flush=True,
            )
            manifests[key] = generate_vadcp_dataset(
                None,
                library_root,
                arm_root,
                synthetic_images=images,
                seed=seed,
                mode=mode,
                preset="sni_spread",
                canvas_size=canvas_size,
                object_range=(objects_min, objects_max),
                include_real_train=False,
                materialize_real_splits=False,
                scene_profile=calibration,
                target_names=names,
            )
            visuals[key] = run_vadcp_visual_audit(
                arm_root,
                policy_root / f"{arm}_visual",
                samples=images,
                seed=seed,
            )

    compositions = {}
    for key, manifest in manifests.items():
        class_counts = {
            name: int(value)
            for name, value in manifest["instances_by_class"].items()
        }
        total = sum(class_counts.values())
        normal = class_counts.get("biji_normal", 0)
        compositions[key] = {
            "labeled_instances": total,
            "normal_instances": normal,
            "normal_fraction": normal / total if total else None,
            "instances_by_class": class_counts,
            "repeated_assets": manifest["repeated_assets"],
            "ignored_instances": manifest["ignored_instances"],
            "geometry_target_hit_rate": manifest["geometry_target_hit_rate"],
        }

    summary = {
        "format": "coffee_detector.sni_crop_preview.v1",
        "training_executed": False,
        "crop_dataset_root": str(Path(crop_dataset_root).expanduser().resolve()),
        "object_library": str(library_root),
        "seed": seed,
        "images_per_arm": images,
        "objects_per_scene": [objects_min, objects_max],
        "canvas_long_side": canvas_size,
        "mass_claim": False,
        "prevalence_claim": False,
        "claim_note": (
            "source_empirical mengikuti komposisi train paket crop, bukan "
            "prevalensi populasi 300 g. defect_enriched adalah augmentasi "
            "training terkontrol; val/test final harus nyata."
        ),
        "policies": policies,
        "compositions": compositions,
        "cutout_contact_sheet": cutout_audit["contact_sheet"],
        "raw_contact_sheets": {
            key: value["raw_contact_sheet"] for key, value in visuals.items()
        },
        "overlay_contact_sheets": {
            key: value["contact_sheet"] for key, value in visuals.items()
        },
        "manifests": manifests,
        "ready_for_visual_review": True,
        "training_ready": False,
    }
    summary_path = output_root / "preview_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== PREVIEW SNI CROP SELESAI — TRAINING TIDAK DIJALANKAN ===", flush=True)
    print(f"Cutout sheet : {summary['cutout_contact_sheet']}", flush=True)
    for key, path in summary["raw_contact_sheets"].items():
        normal_fraction = compositions[key]["normal_fraction"]
        print(
            f"{key:28s}: normal={normal_fraction:.2%} | raw={path}",
            flush=True,
        )
    print(f"Ringkasan    : {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Buat pilot scene 300 g dari coffee-sni-instance-crop-v1; "
            "tidak menjalankan training."
        )
    )
    parser.add_argument("--crop-dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects-min", type=int, default=220)
    parser.add_argument("--objects-max", type=int, default=300)
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--enriched-normal-fraction", type=float, default=0.55)
    parser.add_argument("--max-normal-assets", type=int, default=300)
    parser.add_argument("--max-defect-assets-per-class", type=int, default=60)
    parser.add_argument("--shard-cache-root")
    args = parser.parse_args()
    run_sni_crop_preview(
        args.crop_dataset_root,
        args.output_root,
        images=args.images,
        seed=args.seed,
        objects_min=args.objects_min,
        objects_max=args.objects_max,
        canvas_size=args.canvas_size,
        enriched_normal_fraction=args.enriched_normal_fraction,
        max_normal_assets=args.max_normal_assets,
        max_defect_assets_per_class=args.max_defect_assets_per_class,
        shard_cache_root=args.shard_cache_root,
    )


if __name__ == "__main__":
    main()
