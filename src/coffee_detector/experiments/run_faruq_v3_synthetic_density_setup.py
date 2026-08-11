"""Build a group-safe synthetic density diagnostic from Faruq-v3 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from coffee_detector.audit_vadcp import audit_vadcp_dataset
from coffee_detector.dataset import discover_layout
from coffee_detector.generate_vadcp_dataset import generate_vadcp_dataset
from coffee_detector.prepare_sni_fullscene import (
    SNI21_CLASSES,
    canonical_source_identity,
)
from coffee_detector.run_sni21_density_benchmark_setup import (
    DENSITY_LADDER,
    _reuse_limits,
    _write_resampling_units,
)
from coffee_detector.vadcp.library import (
    _finalize_library,
    _stable_id,
    _write_asset,
    load_object_library,
)
from coffee_detector.vadcp.profile import build_scene_calibration, save_scene_calibration


ARTIFACT_ROLE = "development_benchmark"


def _load_json(path: str | Path, label: str) -> dict | list:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coco_polygons(annotation: dict) -> list[list[float]]:
    segmentation = annotation.get("segmentation", [])
    if not isinstance(segmentation, list):
        return []
    if segmentation and isinstance(segmentation[0], (int, float)):
        segmentation = [segmentation]
    polygons = []
    for polygon in segmentation:
        if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
            polygons.append([float(value) for value in polygon])
    return polygons


def _prepare_faruq_polygon_library(
    polygon_root: Path,
    grouped_manifest: Path,
    output_root: Path,
) -> dict:
    """Extract val-authority cutouts from repaired Faruq COCO polygons.

    Faruq-v3 stores detection boxes for training, while Faruq-v2 also retains
    the audited polygons. Parent authority is therefore taken from the v3
    grouped manifest and pixels/masks from the corresponding v2 record.
    """

    manifest = _load_json(grouped_manifest, "Faruq grouped manifest")
    allowed_parents = {
        str(row["source_parent_id"])
        for row in manifest
        if row["output_split"] == "val"
    }
    annotation_files = sorted(polygon_root.glob("*/_annotations.coco.json"))
    if len(annotation_files) != 2:
        raise FileNotFoundError(
            f"Diharapkan dua COCO development di {polygon_root}; "
            f"ditemukan {len(annotation_files)}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    assets = []
    failures = []
    selected_by_class = Counter()
    selected_parents = set()
    for annotation_path in annotation_files:
        payload = _load_json(annotation_path, "Faruq polygon COCO")
        categories = {
            int(row["id"]): str(row["name"])
            for row in payload.get("categories", [])
        }
        expected = {index: name for index, name in enumerate(SNI21_CLASSES)}
        if categories != expected:
            raise RuntimeError(f"Mapping COCO bukan canonical SNI-21: {annotation_path}")
        annotations_by_image = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations_by_image[str(annotation["image_id"])].append(annotation)
        for image_record in payload.get("images", []):
            parent_id = canonical_source_identity(Path(image_record["file_name"]).name)
            if parent_id not in allowed_parents:
                continue
            image_path = annotation_path.parent / str(image_record["file_name"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as opened:
                image = opened.convert("RGBA")
            image_digest = _sha256(image_path)
            selected_parents.add(parent_id)
            for annotation in annotations_by_image.get(str(image_record["id"]), []):
                class_id = int(annotation["category_id"])
                polygons = _coco_polygons(annotation)
                if not polygons:
                    failures.append(f"{image_path}: annotation {annotation['id']} tanpa polygon")
                    continue
                xs = [value for polygon in polygons for value in polygon[0::2]]
                ys = [value for polygon in polygons for value in polygon[1::2]]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                pad = max(x2 - x1, y2 - y1) * 0.12
                pixel_box = (
                    max(0, int(np.floor(x1 - pad))),
                    max(0, int(np.floor(y1 - pad))),
                    min(image.width, int(np.ceil(x2 + pad))),
                    min(image.height, int(np.ceil(y2 + pad))),
                )
                crop = image.crop(pixel_box)
                mask_image = Image.new("L", crop.size, 0)
                drawer = ImageDraw.Draw(mask_image)
                for polygon in polygons:
                    points = [
                        (polygon[index] - pixel_box[0], polygon[index + 1] - pixel_box[1])
                        for index in range(0, len(polygon), 2)
                    ]
                    drawer.polygon(points, fill=255)
                explicit_mask = np.asarray(mask_image, dtype=np.uint8) > 0
                source_id = _stable_id(image_digest, annotation["id"], class_id)
                item, failure = _write_asset(
                    output_root,
                    SNI21_CLASSES[class_id],
                    source_id,
                    "val",
                    image_path,
                    crop,
                    24.0,
                    2,
                    0.005,
                    0.99,
                    preferred_center=None,
                    explicit_mask=explicit_mask,
                )
                if item:
                    item["source_parent_id"] = parent_id
                    item["source_annotation_id"] = str(annotation["id"])
                    assets.append(item)
                    selected_by_class[SNI21_CLASSES[class_id]] += 1
                if failure:
                    failures.append(failure)
            image.close()
    missing_parents = allowed_parents - selected_parents
    if missing_parents:
        raise RuntimeError(
            f"{len(missing_parents)} parent validation tidak ditemukan dalam COCO polygon"
        )
    return _finalize_library(
        output_root,
        assets,
        failures,
        {
            "type": "faruq_repaired_coco_polygon",
            "root": str(polygon_root),
            "source_split": "val",
            "source_group_split": "faruq_v3_validation",
            "selected_parents": len(selected_parents),
            "selected_by_class": dict(sorted(selected_by_class.items())),
        },
    )


def _audit_library(
    data_root: Path, grouped_manifest: Path, library_root: Path
) -> dict:
    manifest = _load_json(grouped_manifest, "Faruq grouped manifest")
    library = _load_json(library_root / "object_library.json", "Object library")
    train_parents = {
        str(row["source_parent_id"])
        for row in manifest
        if row["output_split"] == "train"
    }
    val_parents = {
        str(row["source_parent_id"])
        for row in manifest
        if row["output_split"] == "val"
    }
    assets = library.get("assets", [])
    asset_parents = {str(row.get("source_parent_id")) for row in assets}
    classes = {int(key): str(value) for key, value in library.get("classes", {}).items()}
    expected = {index: name for index, name in enumerate(SNI21_CLASSES)}
    gates = {
        "source_is_faruq_v3_validation": (
            library.get("source", {}).get("source_split") == "val"
            and library.get("source", {}).get("source_group_split")
            == "faruq_v3_validation"
        ),
        "zero_train_parent_overlap": not (asset_parents & train_parents),
        "all_library_parents_are_validation": asset_parents <= val_parents,
        "all_21_classes_present": classes == expected,
        "no_test_split_available": not (data_root / "test").exists(),
    }
    return {
        "format": "coffee_detector.faruq_v3_density_library_audit.v1",
        "assets": len(assets),
        "asset_parents": len(asset_parents),
        "train_parents": len(train_parents),
        "validation_parents": len(val_parents),
        "train_parent_overlap": sorted(asset_parents & train_parents),
        "non_validation_parents": sorted(asset_parents - val_parents),
        "gates": gates,
        "safe_for_development_diagnostic": all(gates.values()),
        "training_executed": False,
        "test_images_accessed": False,
    }


def run_faruq_v3_synthetic_density_setup(
    data_root: str | Path,
    polygon_root: str | Path,
    grouped_summary: str | Path,
    grouped_manifest: str | Path,
    output_root: str | Path,
    *,
    scenes_per_condition: int = 100,
    seed: int = 42,
    canvas_size: int = 1024,
    object_scale: tuple[float, float] = (0.025, 0.055),
    library_cache_root: str | Path | None = None,
) -> dict:
    data_root = Path(data_root).expanduser().resolve()
    polygon_root = Path(polygon_root).expanduser().resolve()
    grouped_manifest = Path(grouped_manifest).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    summary = _load_json(grouped_summary, "Faruq grouped summary")
    if (
        summary.get("training_ready") is not True
        or summary.get("test_images_accessed") is not False
        or summary.get("test_locked") is not True
        or not all(summary.get("gates", {}).values())
    ):
        raise RuntimeError("Faruq-v3 grouped development belum aman")
    layout = discover_layout(data_root)
    if "train" not in layout.splits or "val" not in layout.splits:
        raise FileNotFoundError("Faruq-v3 memerlukan train dan val")
    if "test" in layout.splits or (data_root / "test").exists():
        raise RuntimeError("Test tidak boleh tersedia pada setup density")
    expected = {index: name for index, name in enumerate(SNI21_CLASSES)}
    if layout.names != expected:
        raise RuntimeError("Mapping kelas Faruq-v3 bukan canonical SNI-21")

    output_root.mkdir(parents=True, exist_ok=True)
    library_root = output_root / "val_object_library"
    library_manifest = library_root / "object_library.json"
    persistent_library_valid = False
    if library_manifest.is_file():
        cached_library = _load_json(library_manifest, "Object library")
        persistent_library_valid = (
            cached_library.get("source", {}).get("type")
            == "faruq_repaired_coco_polygon"
        )
    work_library_root = (
        Path(library_cache_root).expanduser().resolve()
        if library_cache_root is not None
        else library_root
    )
    if work_library_root != library_root:
        if work_library_root.exists():
            shutil.rmtree(work_library_root)
        if persistent_library_valid:
            print("CACHE object library Drive ke disk lokal", flush=True)
            shutil.copytree(library_root, work_library_root)
        else:
            print("EXTRACT polygon library langsung ke disk lokal", flush=True)
            _prepare_faruq_polygon_library(
                polygon_root, grouped_manifest, work_library_root
            )
            if library_root.exists():
                shutil.rmtree(library_root)
            print("BACKUP object library ke Drive", flush=True)
            shutil.copytree(work_library_root, library_root)
    elif not persistent_library_valid:
        if library_root.exists():
            shutil.rmtree(library_root)
        _prepare_faruq_polygon_library(polygon_root, grouped_manifest, library_root)
    library_audit = _audit_library(data_root, grouped_manifest, library_root)
    audit_path = output_root / "validation_library_audit.json"
    audit_path.write_text(
        json.dumps(library_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not library_audit["safe_for_development_diagnostic"]:
        raise RuntimeError(f"Object library validation tidak aman: {audit_path}")

    calibration = build_scene_calibration(layout, split="val", seed=seed)
    if set(calibration.class_probabilities) != set(expected):
        raise RuntimeError("Calibration validation kehilangan kelas")
    profile_path = save_scene_calibration(
        calibration, output_root / "faruq_v3_val_scene_profile.json"
    )
    names, cutouts, _ = load_object_library(work_library_root, train_only=False)
    if names != expected or {item.class_id for item in cutouts} != set(expected):
        raise RuntimeError("Cutout validation kehilangan kelas SNI-21")

    arms = {}
    for index, (density_code, density) in enumerate(DENSITY_LADDER.items(), 1):
        arm_name = f"{density_code}_balanced_mild"
        arm_root = output_root / arm_name
        asset_limit, parent_limit, reuse_plan = _reuse_limits(
            cutouts,
            calibration,
            scenes=scenes_per_condition,
            density=density,
            balanced=True,
            requested_asset_limit=None,
            requested_parent_limit=None,
        )
        manifest_path = arm_root / "metadata/generation_manifest.json"
        if not manifest_path.is_file():
            if arm_root.exists() and any(arm_root.iterdir()):
                shutil.rmtree(arm_root)
            print(
                f"[{index}/4] GENERATE {arm_name}: {density[0]}-{density[1]} objek",
                flush=True,
            )
            generate_vadcp_dataset(
                None,
                work_library_root,
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
                artifact_role=ARTIFACT_ROLE,
                library_source_split="val",
                synthetic_split="val",
                class_balanced=True,
                target_visibility_bin="mild",
                max_asset_reuse=asset_limit,
                max_parent_reuse=parent_limit,
            )
        else:
            print(f"[{index}/4] REUSE {arm_name}", flush=True)
        audit = audit_vadcp_dataset(arm_root, arm_root / "metadata/vadcp_audit.json")
        if not audit["safe_for_training"]:
            raise RuntimeError(f"Audit geometry gagal: {arm_name}")
        resampling = _write_resampling_units(arm_root)
        arms[arm_name] = {
            "root": str(arm_root),
            "density": list(density),
            "prior": "balanced_diagnostic",
            "visibility": "mild",
            "manifest": str(manifest_path),
            "audit": str(arm_root / "metadata/vadcp_audit.json"),
            "reuse_plan": reuse_plan,
            "resampling_units": resampling,
        }

    payload = {
        "format": "coffee_detector.faruq_v3_synthetic_density_setup.v1",
        "status": "complete",
        "development_only": True,
        "source_split": "faruq_v3_validation",
        "source_correlated_with_model_selection": True,
        "scenes_per_condition": scenes_per_condition,
        "seed": seed,
        "constant_object_scale": list(object_scale),
        "class_prior": "balanced_diagnostic",
        "training_executed": False,
        "test_images_accessed": False,
        "further_tuning_authorized": False,
        "validation_library_audit": str(audit_path),
        "persistent_object_library": str(library_root),
        "scene_profile": str(profile_path),
        "arms": arms,
        "ready_for_frozen_screening": True,
    }
    summary_path = output_root / "setup_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(summary_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Faruq-v3 validation density diagnostic")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--polygon-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--grouped-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scenes-per-condition", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--library-cache-root")
    args = parser.parse_args()
    result = run_faruq_v3_synthetic_density_setup(
        args.data_root,
        args.polygon_root,
        args.grouped_summary,
        args.grouped_manifest,
        args.output_root,
        scenes_per_condition=args.scenes_per_condition,
        seed=args.seed,
        library_cache_root=args.library_cache_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
