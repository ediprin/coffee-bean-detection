from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .generate_vadcp_dataset import generate_vadcp_dataset
from .run_vadcp_visual_audit import run_vadcp_visual_audit
from .vadcp.profile import build_scene_calibration, save_scene_calibration
from .dataset import discover_layout


def run_sni_spread_preview(
    real_data_root: str | Path,
    object_library: str | Path,
    output_root: str | Path,
    *,
    images: int = 4,
    seed: int = 42,
    objects_min: int = 220,
    objects_max: int = 300,
    canvas_size: int = 768,
) -> dict:
    """Generate paired high-count A1/A2 previews without copying or training data."""
    if images <= 0:
        raise ValueError("images harus positif")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    profile_path = output_root / "scene_profile.json"
    print("[1/4] Membaca prior geometri dan distribusi kelas train nyata...", flush=True)
    calibration = build_scene_calibration(
        discover_layout(real_data_root), split="train", seed=seed
    )
    save_scene_calibration(calibration, profile_path)

    manifests = {}
    for step, (arm, mode) in enumerate((("A1", "naive"), ("A2", "visibility")), 2):
        arm_root = output_root / arm
        print(
            f"[{step}/4] Membuat {arm}: {images} scene, "
            f"{objects_min}-{objects_max} biji/scene...",
            flush=True,
        )
        manifests[arm] = generate_vadcp_dataset(
            real_data_root,
            object_library,
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
        )

    print("[4/4] Membuat contact sheet A1 dan A2...", flush=True)
    visuals = {
        arm: run_vadcp_visual_audit(
            output_root / arm,
            output_root / f"{arm}_visual",
            samples=images,
            seed=seed,
        )
        for arm in ("A1", "A2")
    }
    audits = {}
    for arm in ("A1", "A2"):
        metadata_path = Path(manifests[arm]["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        counts = Counter(int(row["image_id"]) for row in metadata["annotations"])
        observed = [counts[int(row["id"])] for row in metadata["images"]]
        audits[arm] = {
            "scene_count_min": min(observed),
            "scene_count_max": max(observed),
            "scene_count_mean": sum(observed) / len(observed),
            "within_requested_density": all(
                objects_min <= value <= objects_max for value in observed
            ),
            "contains_pile_scene": bool(manifests[arm]["scene_modes"].get("pile", 0)),
            "scene_modes": manifests[arm]["scene_modes"],
            "visibility": manifests[arm]["instances_by_visibility"],
            "repeated_assets": manifests[arm]["repeated_assets"],
        }
    ready_for_visual_review = all(
        item["within_requested_density"] and not item["contains_pile_scene"]
        for item in audits.values()
    )
    summary = {
        "format": "coffee_detector.sni_spread_preview.v1",
        "training_executed": False,
        "preset": "sni_spread",
        "mass_claim": False,
        "mass_note": (
            "Rentang jumlah adalah pilot visual. Hubungan dengan 300 g harus "
            "dikalibrasi dari hitungan biji pada foto sampel 300 g nyata."
        ),
        "seed": seed,
        "images_per_arm": images,
        "objects_per_scene": [objects_min, objects_max],
        "canvas_long_side": canvas_size,
        "scene_profile": str(profile_path),
        "audit": audits,
        "ready_for_visual_review": ready_for_visual_review,
        "manifests": manifests,
        "contact_sheets": {
            arm: visuals[arm]["raw_contact_sheet"] for arm in visuals
        },
        "overlay_contact_sheets": {
            arm: visuals[arm]["contact_sheet"] for arm in visuals
        },
    }
    summary_path = output_root / "preview_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== PREVIEW SNI 300G SELESAI — TRAINING TIDAK DIJALANKAN ===", flush=True)
    print(f"A1 raw : {summary['contact_sheets']['A1']}", flush=True)
    print(f"A2 raw : {summary['contact_sheets']['A2']}", flush=True)
    print(f"Density: {audits}", flush=True)
    print(f"SIAP REVIEW VISUAL: {'YA' if ready_for_visual_review else 'BELUM'}", flush=True)
    print(f"Ringkas: {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview cepat copy-paste scene kopi high-count; tanpa training."
    )
    parser.add_argument("--real-data-root", required=True)
    parser.add_argument("--object-library", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects-min", type=int, default=220)
    parser.add_argument("--objects-max", type=int, default=300)
    parser.add_argument("--canvas-size", type=int, default=768)
    args = parser.parse_args()
    run_sni_spread_preview(
        args.real_data_root,
        args.object_library,
        args.output_root,
        images=args.images,
        seed=args.seed,
        objects_min=args.objects_min,
        objects_max=args.objects_max,
        canvas_size=args.canvas_size,
    )


if __name__ == "__main__":
    main()
