import json
from pathlib import Path

from PIL import Image, ImageDraw

from coffee_detector.prepare_faruq_locked_test import prepare_faruq_locked_test
from coffee_detector.prepare_sni_fullscene import (
    FARUQ_CLASS_MAP,
    SNI21_CLASSES,
    canonical_source_identity,
)


RAW_NAMES = {canonical: raw for raw, canonical in FARUQ_CLASS_MAP.items()}


def _write_raw_test(root: Path) -> tuple[str, str]:
    test = root / "test"
    test.mkdir(parents=True)
    categories = [
        {"id": index + 1, "name": RAW_NAMES[name]}
        for index, name in enumerate(SNI21_CLASSES)
    ]
    images, annotations = [], []

    def add(image_id: int, class_index: int, stem: str) -> Path:
        file_name = f"{stem}_jpg.rf.{image_id:08x}.jpg"
        image_path = test / file_name
        image = Image.new("RGB", (80, 60), "white")
        ImageDraw.Draw(image).rectangle(
            (15, 12, 55, 45),
            fill=(60 + image_id % 120, 30 + image_id % 80, 10 + image_id % 50),
        )
        image.save(image_path, quality=100, subsampling=0)
        images.append({"id": image_id, "file_name": file_name, "width": 80, "height": 60})
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": class_index + 1,
                "bbox": [15, 12, 40, 33],
                "segmentation": [[15, 12, 55, 12, 55, 45, 15, 45]],
            }
        )
        return image_path

    overlap_path = add(1, 0, "overlap")
    for index in range(len(SNI21_CLASSES)):
        add(index + 10, index, f"parent_{index}")
    add(100, 0, "parent_0")  # test-only pseudo-replicate
    payload = {"images": images, "annotations": annotations, "categories": categories}
    (test / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    return canonical_source_identity(overlap_path.name), "unused-hash"


def test_locked_test_filters_development_and_test_pseudoreplication(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    overlap_parent, overlap_hash = _write_raw_test(raw)
    development = tmp_path / "faruq_grouped_manifest.json"
    development.write_text(
        json.dumps(
            [
                {
                    "output_split": "train",
                    "source_parent_id": overlap_parent,
                    "source_sha256": overlap_hash,
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "locked"
    summary = prepare_faruq_locked_test(
        raw,
        development,
        output,
        minimum_images=21,
        minimum_instances_per_class=1,
        minimum_parents_per_class=1,
    )

    assert summary["decision"] == "PASS"
    assert summary["materialized_images"] == 21
    assert summary["materialized_annotations"] == 21
    assert summary["gates"]["zero_development_parent_overlap"]
    assert summary["gates"]["one_image_per_test_parent"]
    assert summary["test_images_accessed"] is True
    assert summary["inference_executed"] is False
    assert len(list((output / "test/images").glob("*.jpg"))) == 21


def test_locked_test_default_gate_rejects_weak_independent_support(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    overlap_parent, overlap_hash = _write_raw_test(raw)
    development = tmp_path / "faruq_grouped_manifest.json"
    development.write_text(
        json.dumps(
            [
                {
                    "output_split": "val",
                    "source_parent_id": overlap_parent,
                    "source_sha256": overlap_hash,
                }
            ]
        ),
        encoding="utf-8",
    )
    summary = prepare_faruq_locked_test(raw, development, tmp_path / "locked")
    assert summary["decision"] == "FAIL"
    assert summary["next_action"] == "STOP_TEST_INFERENCE_USE_GROUPED_CV_OR_EXTERNAL_TEST"
    assert not summary["gates"]["minimum_parents_per_class"]
