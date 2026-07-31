import csv
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from coffee_detector.audit_vadcp import audit_vadcp_dataset
from coffee_detector.generate_vadcp_dataset import generate_vadcp_dataset
from coffee_detector.run_sni_crop_preview import run_sni_crop_preview
from coffee_detector.run_sni21_density_benchmark_setup import (
    _reuse_limits,
    audit_validation_library_provenance,
)
from coffee_detector.sni_crop_manifest import build_sni_crop_calibration
from coffee_detector.vadcp.library import prepare_sni_crop_manifest_library


def test_density_reuse_limit_reserves_rare_class_sampling_variance() -> None:
    cutouts = [
        SimpleNamespace(class_id=0, source_parent_id=f"common-{index}")
        for index in range(99)
    ]
    cutouts.append(
        SimpleNamespace(class_id=1, source_parent_id="rare-parent")
    )
    calibration = SimpleNamespace(
        class_probabilities={0: 0.999, 1: 0.001}
    )

    asset_limit, parent_limit, plan = _reuse_limits(
        cutouts,
        calibration,
        scenes=200,
        density=(1, 5),
        balanced=False,
        requested_asset_limit=None,
        requested_parent_limit=None,
    )

    assert plan["class_upper_instances"]["1"] > 2
    assert asset_limit >= plan["class_lower_bounds"]["1"]
    assert parent_limit >= plan["parent_class_lower_bounds"]["1"]
    with pytest.raises(ValueError, match="lower bound"):
        _reuse_limits(
            cutouts,
            calibration,
            scenes=200,
            density=(1, 5),
            balanced=False,
            requested_asset_limit=2,
            requested_parent_limit=None,
        )


def _write_crop_package(root: Path) -> None:
    rows = []
    payloads = {}
    sample_id = 0
    for split, count in (("train", 8), ("val", 2), ("test", 2)):
        archive_image_id = 0
        for class_name, color in (
            ("biji_normal", (130, 100, 55)),
            ("biji_hitam", (40, 30, 25)),
        ):
            for index in range(count):
                sample_id += 1
                crop_path = (
                    f"source/{split}/{class_name}/"
                    f"sample_{sample_id:04d}.jpg"
                )
                image = Image.new("RGB", (64, 64), (238, 235, 230))
                offset = index % 4
                varied_color = tuple(
                    min(255, channel + index % 7) for channel in color
                )
                ImageDraw.Draw(image).ellipse(
                    (14 + offset, 20, 50 + offset, 44),
                    fill=varied_color,
                )
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)
                payloads[crop_path] = buffer.getvalue()
                rows.append(
                    {
                        "dataset": "fixture",
                        "archive_split": split,
                        "generated_split": split,
                        "image_id": str(archive_image_id),
                        "source_identity": f"{split}-{class_name}-{index}",
                        "canonical_class": class_name,
                        "bbox_width": "36",
                        "bbox_height": "24",
                        "crop_sha256": f"digest-{sample_id}",
                        "crop_path": crop_path,
                    }
                )
                archive_image_id += 1
    root.mkdir(parents=True)
    (root / "shards").mkdir()
    (root / "complete.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (root / "audit.json").write_text(
        json.dumps(
            {
                "source_audits": {
                    "fixture": {
                        "images": 24,
                        "archive_counts": {
                            "train": {"images": 16},
                            "val": {"images": 4},
                            "test": {"images": 4},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with (root / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with tarfile.open(root / "shards" / "crop_shard_00001_00024.tar", "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_sni_crop_library_is_train_only_and_parent_aware(tmp_path: Path) -> None:
    source = tmp_path / "crop-package"
    output = tmp_path / "library"
    _write_crop_package(source)

    result = prepare_sni_crop_manifest_library(
        source,
        output,
        max_normal_assets=4,
        max_defect_assets_per_class=3,
        seed=7,
    )

    assert result["audit"]["assets_by_class"] == {
        "biji_hitam": 3,
        "biji_normal": 4,
    }
    assert result["audit"]["assets_by_source_split"] == {"train": 7}
    assert all(item["source_parent_id"] for item in result["assets"])
    assert all(
        item["mask_source"] == "estimated_foreground"
        for item in result["assets"]
    )
    assert all(
        item["mask_background_model"] == "spatial_prototypes"
        for item in result["assets"]
    )
    assert result["source"]["mask_config"] == {
        "background_model": "spatial_prototypes",
        "distance_threshold": 40.0,
        "border_trim_fraction": 0.03,
        "matte": "inward_feathered_premultiplied",
    }
    assert result["source"]["split_counts"] == {
        "test": 4,
        "train": 16,
        "val": 4,
    }


def test_sni_crop_shards_use_global_dataset_offsets(tmp_path: Path) -> None:
    source = tmp_path / "multi-dataset-crops"
    output = tmp_path / "library"
    (source / "shards").mkdir(parents=True)
    (source / "complete.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (source / "audit.json").write_text(
        json.dumps(
            {
                "source_audits": {
                    "dataset_a": {
                        "images": 1,
                        "archive_counts": {"train": {"images": 1}},
                    },
                    "dataset_b": {
                        "images": 1,
                        "archive_counts": {"train": {"images": 1}},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for ordinal, (dataset, class_name, color) in enumerate(
        (
            ("dataset_a", "biji_normal", (130, 100, 55)),
            ("dataset_b", "biji_hitam", (35, 30, 25)),
        ),
        1,
    ):
        crop_path = f"source/train/{class_name}/{dataset}__train__0000000.jpg"
        image = Image.new("RGB", (64, 64), (238, 235, 230))
        ImageDraw.Draw(image).ellipse((14, 20, 50, 44), fill=color)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        with tarfile.open(
            source / "shards" / f"crop_shard_{ordinal:05d}_{ordinal:05d}.tar",
            "w",
        ) as archive:
            info = tarfile.TarInfo(crop_path)
            info.size = len(buffer.getvalue())
            archive.addfile(info, io.BytesIO(buffer.getvalue()))
        rows.append(
            {
                "dataset": dataset,
                "archive_split": "train",
                "generated_split": "train",
                "image_id": "0",
                "source_identity": f"{dataset}-source",
                "canonical_class": class_name,
                "bbox_width": "36",
                "bbox_height": "24",
                "crop_sha256": f"{dataset}-digest",
                "crop_path": crop_path,
            }
        )
    with (source / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = prepare_sni_crop_manifest_library(
        source,
        output,
        max_normal_assets=1,
        max_defect_assets_per_class=1,
    )

    assert result["audit"]["assets_by_class"] == {
        "biji_hitam": 1,
        "biji_normal": 1,
    }
    assert result["source"]["global_source_images"] == 2


def test_sni_crop_composition_keeps_normal_dominant(tmp_path: Path) -> None:
    source = tmp_path / "crop-package"
    _write_crop_package(source)

    names, source_calibration, source_report = build_sni_crop_calibration(
        source,
        policy="source_empirical",
        objects_min=20,
        objects_max=30,
    )
    _, enriched_calibration, enriched_report = build_sni_crop_calibration(
        source,
        policy="defect_enriched",
        enriched_normal_fraction=0.60,
        objects_min=20,
        objects_max=30,
    )
    normal_id = next(index for index, name in names.items() if name == "biji_normal")

    assert source_report["source_normal_fraction"] == 0.5
    assert source_calibration.class_probabilities[normal_id] == 0.5
    assert enriched_calibration.class_probabilities[normal_id] == 0.60
    assert enriched_report["requested_normal_fraction"] == 0.60

    _, validation_calibration, validation_report = build_sni_crop_calibration(
        source,
        policy="source_empirical",
        source_split="val",
        objects_min=1,
        objects_max=5,
    )
    assert validation_calibration.split == "val_crop_manifest"
    assert validation_report["source_split"] == "val"
    assert validation_report["source_crops"] == 4


def test_validation_library_provenance_uses_only_val_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "crop-package"
    library = tmp_path / "val-library"
    _write_crop_package(source)
    prepare_sni_crop_manifest_library(
        source,
        library,
        source_split="val",
        max_normal_assets=2,
        max_defect_assets_per_class=2,
        seed=7,
    )

    report = audit_validation_library_provenance(source, library)

    assert report["selected_source_split"] == "val"
    assert report["parent_identity_overlap"] == {"train": 0, "test": 0}
    assert report["crop_identity_overlap"] == {"train": 0, "test": 0}
    assert report["test_images_opened"] is False
    # The tiny fixture has two classes rather than canonical SNI-21, so the
    # complete production gate must remain closed.
    assert not report["safe_for_development_benchmark"]
    assert "canonical SNI-21" in report["errors"][-1]


def test_synthetic_preview_can_run_without_real_detection_dataset(
    tmp_path: Path,
) -> None:
    source = tmp_path / "crop-package"
    library = tmp_path / "library"
    output = tmp_path / "preview"
    _write_crop_package(source)
    prepare_sni_crop_manifest_library(
        source,
        library,
        max_normal_assets=4,
        max_defect_assets_per_class=4,
    )
    names, calibration, _ = build_sni_crop_calibration(
        source,
        policy="defect_enriched",
        enriched_normal_fraction=0.60,
        objects_min=8,
        objects_max=8,
    )

    manifest = generate_vadcp_dataset(
        None,
        library,
        output,
        synthetic_images=1,
        seed=5,
        mode="naive",
        preset="sni_spread",
        canvas_size=256,
        object_range=(8, 8),
        include_real_train=False,
        materialize_real_splits=False,
        use_shadows=False,
        scene_profile=calibration,
        target_names=names,
    )

    assert manifest["synthetic_images"] == 1
    assert manifest["real_images"] == {"train": 0, "val": 0, "test": 0}
    assert sum(manifest["instances_by_class"].values()) == 8
    assert (output / "train" / "images" / "naive_seed5_000000.jpg").is_file()


def test_development_benchmark_uses_val_assets_and_val_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "crop-package"
    library = tmp_path / "val-library"
    output = tmp_path / "benchmark"
    _write_crop_package(source)
    prepare_sni_crop_manifest_library(
        source,
        library,
        source_split="val",
        max_normal_assets=2,
        max_defect_assets_per_class=2,
        seed=5,
    )
    names, calibration, _ = build_sni_crop_calibration(
        source,
        policy="source_empirical",
        source_split="val",
        objects_min=2,
        objects_max=2,
    )

    manifest = generate_vadcp_dataset(
        None,
        library,
        output,
        synthetic_images=1,
        seed=5,
        mode="visibility",
        preset="sni_spread",
        canvas_size=256,
        object_range=(2, 2),
        include_real_train=False,
        materialize_real_splits=False,
        use_shadows=False,
        scene_profile=calibration,
        target_names=names,
        artifact_role="development_benchmark",
        library_source_split="val",
        synthetic_split="val",
        target_visibility_bin="mild",
        max_asset_reuse=2,
        max_parent_reuse=2,
    )
    audit = audit_vadcp_dataset(output)

    assert manifest["artifact_role"] == "development_benchmark"
    assert manifest["library_source_split"] == "val"
    assert manifest["synthetic_split"] == "val"
    assert manifest["asset_reuse"]["maximum"] <= 2
    assert audit["safe_for_training"], audit["errors"]
    assert audit["library_source_split"] == "val"
    assert (output / "val/images/visibility_seed5_000000.jpg").is_file()
    assert (output / "metadata/instances_synthetic_val.json").is_file()


def test_sni_crop_preview_records_locked_mask_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "crop-package"
    output = tmp_path / "preview"
    _write_crop_package(source)

    summary = run_sni_crop_preview(
        source,
        output,
        images=1,
        seed=5,
        objects_min=8,
        objects_max=8,
        canvas_size=256,
        enriched_normal_fraction=0.60,
        max_normal_assets=4,
        max_defect_assets_per_class=4,
    )

    assert summary["format"] == "coffee_detector.sni_crop_preview.v2"
    assert summary["mask_config"] == {
        "background_model": "spatial_prototypes",
        "distance_threshold": 40.0,
        "border_trim_fraction": 0.03,
        "matte": "inward_feathered_premultiplied",
    }
    assert summary["training_executed"] is False
    assert Path(summary["cutout_contact_sheet"]).is_file()
