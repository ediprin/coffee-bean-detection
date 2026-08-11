from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_vadcp import audit_vadcp_dataset, print_audit_summary
from .dataset import discover_layout
from .generate_vadcp_dataset import generate_vadcp_dataset
from .prepare_sni_fullscene import SNI21_CLASSES
from .run_vadcp_visual_audit import run_vadcp_visual_audit
from .sni_crop_manifest import build_sni_crop_calibration
from .vadcp.library import prepare_sni_crop_manifest_library
from .vadcp.profile import save_scene_calibration


ARM_MODES = {"A1": "naive", "A2": "visibility"}


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} bukan JSON object: {path}")
    return payload


def _validate_real_dataset(real_data_root: str | Path) -> tuple[Path, dict[int, str]]:
    root = Path(real_data_root).expanduser().resolve()
    layout = discover_layout(root)
    missing = sorted({"train", "val", "test"} - set(layout.splits))
    if missing:
        raise FileNotFoundError(
            "Dataset real belum memiliki split: " + ", ".join(missing)
        )
    expected = {index: name for index, name in enumerate(SNI21_CLASSES)}
    if layout.names != expected:
        raise ValueError(
            "Nama/urutan kelas dataset real bukan canonical SNI-21"
        )
    materialization = _load_json(root / "audit.json", "Audit materialisasi")
    post_audit = _load_json(
        root / "post_materialization_audit.json",
        "Audit pascamaterialisasi",
    )
    if not materialization.get("training_ready"):
        raise RuntimeError("Dataset real belum TRAINING_READY")
    if materialization.get("test_locked") is not True:
        raise RuntimeError("Dataset real tidak mencatat test_locked=true")
    if not post_audit.get("safe_for_training"):
        raise RuntimeError("Audit pascamaterialisasi belum aman")
    if int(post_audit.get("cross_split_duplicate_components", -1)) != 0:
        raise RuntimeError("Dataset real masih memiliki duplicate lintas split")
    return root, layout.names


def _validate_library(
    library_root: Path,
    *,
    seed: int,
    max_normal_assets: int,
    max_defect_assets_per_class: int,
) -> dict | None:
    manifest_path = library_root / "object_library.json"
    if not manifest_path.is_file():
        return None
    payload = _load_json(manifest_path, "Manifest object library")
    source = payload.get("source", {})
    expected = {
        "type": "sni_crop_manifest",
        "source_split": "train",
        "seed": seed,
        "max_normal_assets": max_normal_assets,
        "max_defect_assets_per_class": max_defect_assets_per_class,
    }
    conflicts = {
        key: (source.get(key), value)
        for key, value in expected.items()
        if source.get(key) != value
    }
    if conflicts:
        raise RuntimeError(
            "Object library ada tetapi tidak cocok dengan protokol: "
            + json.dumps(conflicts, ensure_ascii=False)
        )
    audit = payload.get("audit", {})
    if int(audit.get("assets", 0)) <= 0:
        raise RuntimeError("Object library tidak memiliki aset valid")
    if set(audit.get("assets_by_source_split", {})) != {"train"}:
        raise RuntimeError("Object library tidak murni berasal dari train")
    classes = {
        int(key): str(value)
        for key, value in payload.get("classes", {}).items()
    }
    if classes != {
        index: name for index, name in enumerate(SNI21_CLASSES)
    }:
        raise RuntimeError("Object library tidak mencakup canonical SNI-21")
    return payload


def _validate_existing_arm(
    arm_root: Path,
    *,
    arm: str,
    mode: str,
    seed: int,
    synthetic_images: int,
    names: dict[int, str],
) -> dict | None:
    manifest_path = arm_root / "metadata" / "generation_manifest.json"
    if not manifest_path.is_file():
        return None
    payload = _load_json(manifest_path, f"Manifest {arm}")
    actual_classes = {
        int(key): str(value)
        for key, value in payload.get("classes", {}).items()
    }
    expected = {
        "mode": mode,
        "preset": "sni_spread",
        "seed": seed,
        "synthetic_images": synthetic_images,
        "include_real_train": False,
        "materialize_real_splits": False,
    }
    conflicts = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if conflicts:
        raise RuntimeError(
            f"Output {arm} ada tetapi tidak cocok dengan protokol: "
            + json.dumps(conflicts, ensure_ascii=False)
        )
    if actual_classes != names:
        raise RuntimeError(
            f"Output {arm} ada tetapi nama/urutan kelasnya berbeda"
        )
    return payload


def run_sni21_vadcp_setup(
    real_data_root: str | Path,
    crop_dataset_root: str | Path,
    output_root: str | Path,
    *,
    synthetic_images: int = 2000,
    seed: int = 42,
    objects_min: int = 220,
    objects_max: int = 300,
    canvas_size: int = 1024,
    max_normal_assets: int = 300,
    max_defect_assets_per_class: int = 60,
    shard_cache_root: str | Path | None = None,
    object_library_root: str | Path | None = None,
    visual_samples: int = 12,
) -> dict:
    """Materialize protocol-locked synthetic-only A1/A2; never train a model."""
    if synthetic_images <= 0:
        raise ValueError("synthetic_images harus positif")
    if objects_min <= 0 or objects_min > objects_max:
        raise ValueError("Rentang objects tidak valid")
    if canvas_size <= 0:
        raise ValueError("canvas_size harus positif")
    if visual_samples <= 0:
        raise ValueError("visual_samples harus positif")

    real_root, real_names = _validate_real_dataset(real_data_root)
    crop_root = Path(crop_dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    library_root = (
        Path(object_library_root).expanduser().resolve()
        if object_library_root is not None
        else output_root / "object_library"
    )

    print("[1/5] Validasi A0 real SNI-21: LULUS", flush=True)
    library = _validate_library(
        library_root,
        seed=seed,
        max_normal_assets=max_normal_assets,
        max_defect_assets_per_class=max_defect_assets_per_class,
    )
    if library is None:
        if library_root.exists() and any(library_root.iterdir()):
            raise RuntimeError(
                f"Object library parsial; gunakan output baru: {library_root}"
            )
        print("[2/5] Membuat object library train-only...", flush=True)
        library = prepare_sni_crop_manifest_library(
            crop_root,
            library_root,
            source_split="train",
            max_normal_assets=max_normal_assets,
            max_defect_assets_per_class=max_defect_assets_per_class,
            seed=seed,
            shard_cache_root=shard_cache_root,
        )
    else:
        print(f"[2/5] REUSE object library: {library_root}", flush=True)

    names, calibration, composition = build_sni_crop_calibration(
        crop_root,
        policy="source_empirical",
        objects_min=objects_min,
        objects_max=objects_max,
    )
    if names != real_names:
        raise ValueError(
            "Nama/urutan kelas crop calibration berbeda dari A0 real"
        )
    profile_path = save_scene_calibration(
        calibration, output_root / "scene_profile_source_empirical.json"
    )

    manifests: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    visuals: dict[str, dict] = {}
    for step, (arm, mode) in enumerate(ARM_MODES.items(), 3):
        arm_root = output_root / arm
        manifest = _validate_existing_arm(
            arm_root,
            arm=arm,
            mode=mode,
            seed=seed,
            synthetic_images=synthetic_images,
            names=names,
        )
        if manifest is None:
            if arm_root.exists() and any(arm_root.iterdir()):
                raise RuntimeError(
                    f"Output {arm} parsial; gunakan output baru: {arm_root}"
                )
            print(
                f"[{step}/5] GENERATE {arm}: {synthetic_images} scene | "
                f"{objects_min}-{objects_max} objek | mode={mode}",
                flush=True,
            )
            manifest = generate_vadcp_dataset(
                None,
                library_root,
                arm_root,
                synthetic_images=synthetic_images,
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
        else:
            print(f"[{step}/5] REUSE {arm}: {arm_root}", flush=True)

        audit = audit_vadcp_dataset(
            arm_root, arm_root / "metadata" / "vadcp_audit.json"
        )
        print_audit_summary(audit, label=arm)
        visual = run_vadcp_visual_audit(
            arm_root,
            output_root / "visual" / arm,
            samples=visual_samples,
            seed=seed,
        )
        manifests[arm] = manifest
        audits[arm] = audit
        visuals[arm] = visual

    training_ready = all(
        audit.get("safe_for_training") and audit.get("geometry_ready")
        for audit in audits.values()
    )
    print("[5/5] Menulis ringkasan setup dan keputusan readiness...", flush=True)
    summary = {
        "format": "coffee_detector.sni21_vadcp_setup.v1",
        "protocol": "docs/SNI21_VADCP_SCREENING_PROTOCOL.md",
        "training_executed": False,
        "test_accessed": False,
        "real_data_root": str(real_root),
        "crop_dataset_root": str(crop_root),
        "output_root": str(output_root),
        "object_library": str(library_root),
        "scene_profile": str(profile_path),
        "composition": composition,
        "seed": seed,
        "synthetic_images_per_arm": synthetic_images,
        "objects_per_scene": [objects_min, objects_max],
        "canvas_size": canvas_size,
        "arms": {
            arm: {
                "mode": ARM_MODES[arm],
                "root": str(output_root / arm),
                "manifest": str(
                    output_root
                    / arm
                    / "metadata"
                    / "generation_manifest.json"
                ),
                "audit": str(
                    output_root / arm / "metadata" / "vadcp_audit.json"
                ),
                "raw_contact_sheet": visuals[arm]["raw_contact_sheet"],
                "overlay_contact_sheet": visuals[arm]["contact_sheet"],
                "safe_for_training": audits[arm]["safe_for_training"],
                "geometry_ready": audits[arm]["geometry_ready"],
            }
            for arm in ARM_MODES
        },
        "training_ready": training_ready,
        "next_action": (
            "Run validation-only seed-42 A0/A1/A2 screening."
            if training_ready
            else "Stop; inspect A1/A2 audit and visual sheets."
        ),
    }
    summary_path = output_root / "setup_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== SETUP SNI-21 A1/A2 SELESAI ===", flush=True)
    print(f"TRAINING_READY : {training_ready}", flush=True)
    print("TRAINING       : BELUM DIJALANKAN", flush=True)
    print("TEST           : TIDAK DIAKSES", flush=True)
    print(f"SUMMARY        : {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dan audit A1/A2 SNI-21; tidak menjalankan training."
        )
    )
    parser.add_argument("--real-data-root", required=True)
    parser.add_argument("--crop-dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--synthetic-images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects-min", type=int, default=220)
    parser.add_argument("--objects-max", type=int, default=300)
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--max-normal-assets", type=int, default=300)
    parser.add_argument("--max-defect-assets-per-class", type=int, default=60)
    parser.add_argument("--shard-cache-root")
    parser.add_argument(
        "--object-library-root",
        help="Library bersama agar smoke/full tidak membaca shard dua kali.",
    )
    parser.add_argument("--visual-samples", type=int, default=12)
    args = parser.parse_args()
    run_sni21_vadcp_setup(
        args.real_data_root,
        args.crop_dataset_root,
        args.output_root,
        synthetic_images=args.synthetic_images,
        seed=args.seed,
        objects_min=args.objects_min,
        objects_max=args.objects_max,
        canvas_size=args.canvas_size,
        max_normal_assets=args.max_normal_assets,
        max_defect_assets_per_class=args.max_defect_assets_per_class,
        shard_cache_root=args.shard_cache_root,
        object_library_root=args.object_library_root,
        visual_samples=args.visual_samples,
    )


if __name__ == "__main__":
    main()
