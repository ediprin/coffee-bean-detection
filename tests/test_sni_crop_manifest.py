import csv
import io
import json
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw

from coffee_detector.generate_vadcp_dataset import generate_vadcp_dataset
from coffee_detector.sni_crop_manifest import build_sni_crop_calibration
from coffee_detector.vadcp.library import prepare_sni_crop_manifest_library


def _write_crop_package(root: Path) -> None:
    rows = []
    payloads = {}
    sample_id = 0
    for split, count in (("train", 8), ("val", 2), ("test", 2)):
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
                        "generated_split": split,
                        "image_id": str(sample_id),
                        "source_identity": f"{split}-{class_name}-{index}",
                        "canonical_class": class_name,
                        "bbox_width": "36",
                        "bbox_height": "24",
                        "crop_sha256": f"digest-{sample_id}",
                        "crop_path": crop_path,
                    }
                )
    root.mkdir(parents=True)
    (root / "shards").mkdir()
    (root / "complete.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
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
    assert result["source"]["split_counts"] == {
        "test": 4,
        "train": 16,
        "val": 4,
    }


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
