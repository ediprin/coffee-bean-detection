import csv
import json
from pathlib import Path

import yaml
from PIL import Image

from coffee_detector.prepare_sni_fullscene import (
    ADRIAN_CLASS_MAP,
    FARUQ_CLASS_MAP,
    SNI21_CLASSES,
    prepare_sni_fullscene,
)
from coffee_detector.run_sni_fullscene_visual_audit import (
    run_sni_fullscene_visual_audit,
)


def _write_coco_split(
    root: Path,
    split: str,
    categories: list[dict],
    images: list[dict],
    annotations: list[dict],
) -> None:
    split_root = root / split
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": item["id"],
                        "file_name": item["file_name"],
                        "width": item["declared_size"][0],
                        "height": item["declared_size"][1],
                    }
                    for item in images
                ],
                "annotations": annotations,
                "categories": categories,
            }
        ),
        encoding="utf-8",
    )
    for item in images:
        Image.new("RGB", item["raw_size"], item["color"]).save(
            split_root / item["file_name"]
        )


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    adrian = root / "adrian"
    faruq = root / "faruq"
    adrian_categories = [
        {"id": 0, "name": "coffee-E6YO"},
        {"id": 1, "name": "Biji tanpa cacat"},
        {"id": 2, "name": "Batu berukuran besar"},
    ]
    faruq_categories = [
        {"id": 0, "name": "robusta-SNI-Dataset"},
        {"id": 1, "name": "biji_normal"},
        {"id": 2, "name": "biji_hitam"},
    ]
    _write_coco_split(
        adrian,
        "train",
        adrian_categories,
        [
            {
                "id": 1,
                "file_name": "shared_jpg.rf.aaa.jpg",
                "declared_size": (20, 10),
                "raw_size": (20, 10),
                "color": (50, 50, 50),
            }
        ],
        [{"id": 11, "image_id": 1, "category_id": 1, "bbox": [2, 2, 8, 5]}],
    )
    _write_coco_split(
        adrian,
        "valid",
        adrian_categories,
        [
            {
                "id": 2,
                "file_name": "shared_jpg.rf.bbb.jpg",
                "declared_size": (20, 10),
                "raw_size": (20, 10),
                "color": (70, 70, 70),
            }
        ],
        [{"id": 12, "image_id": 2, "category_id": 2, "bbox": [4, 1, 7, 7]}],
    )
    _write_coco_split(
        adrian,
        "test",
        adrian_categories,
        [
            {
                "id": 3,
                "file_name": "edge_jpg.rf.ccc.jpg",
                "declared_size": (20, 10),
                "raw_size": (20, 10),
                "color": (90, 90, 90),
            }
        ],
        [{"id": 13, "image_id": 3, "category_id": 1, "bbox": [-0.01, 2, 6, 5]}],
    )
    _write_coco_split(
        faruq,
        "train",
        faruq_categories,
        [
            {
                "id": 4,
                "file_name": "rotated_jpg.rf.ddd.jpg",
                "declared_size": (10, 20),
                "raw_size": (20, 10),
                "color": (110, 80, 50),
            }
        ],
        [
            {
                "id": 14,
                "image_id": 4,
                "category_id": 2,
                "bbox": [1, 4, 6, 8],
                "segmentation": [[1, 4, 7, 4, 7, 12, 1, 12]],
            }
        ],
    )
    _write_coco_split(
        faruq,
        "valid",
        faruq_categories,
        [
            {
                "id": 5,
                "file_name": "conflict_jpg.rf.eee.jpg",
                "declared_size": (20, 10),
                "raw_size": (20, 10),
                "color": (130, 100, 70),
            }
        ],
        [
            {"id": 15, "image_id": 5, "category_id": 1, "bbox": [1, 1, 5, 5]},
            {"id": 16, "image_id": 5, "category_id": 2, "bbox": [8, 1, 5, 5]},
        ],
    )
    _write_coco_split(
        faruq,
        "test",
        faruq_categories,
        [
            {
                "id": 6,
                "file_name": "test_jpg.rf.fff.jpg",
                "declared_size": (20, 10),
                "raw_size": (20, 10),
                "color": (150, 120, 80),
            }
        ],
        [{"id": 17, "image_id": 6, "category_id": 1, "bbox": [2, 2, 8, 5]}],
    )

    rows = [
        ("adrian_detection", "train", "train", "g-shared", "sharedjpg", 1, 11, "biji_normal"),
        (
            "adrian_detection",
            "val",
            "train",
            "g-shared",
            "sharedjpg",
            2,
            12,
            "tanah_batu_ranting_besar",
        ),
        ("adrian_detection", "test", "test", "g-test-a", "edgejpg", 3, 13, "biji_normal"),
        ("faruq_segmentation", "train", "val", "g-val-f", "rotatedjpg", 4, 14, "biji_hitam"),
        # Annotation 16 sengaja tidak dipertahankan; seluruh image 5 harus dikarantina.
        ("faruq_segmentation", "val", "train", "g-conflict", "conflictjpg", 5, 15, "biji_normal"),
        ("faruq_segmentation", "test", "test", "g-test-f", "testjpg", 6, 17, "biji_normal"),
    ]
    manifest = root / "manifest.csv"
    fields = [
        "dataset",
        "archive_split",
        "generated_split",
        "group_id",
        "source_identity",
        "image_id",
        "annotation_id",
        "canonical_class",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(zip(fields, row)))
    return adrian, faruq, manifest


def test_sni_class_maps_cover_exactly_21_canonical_classes() -> None:
    assert set(ADRIAN_CLASS_MAP.values()) == set(SNI21_CLASSES)
    assert set(FARUQ_CLASS_MAP.values()) == set(SNI21_CLASSES)
    assert ADRIAN_CLASS_MAP["Batu berukuran besar"] == ADRIAN_CLASS_MAP[
        "Tanah berukuran besar"
    ]
    assert ADRIAN_CLASS_MAP["Tanah berukuran besar"] == ADRIAN_CLASS_MAP[
        "Ranting berukuran besar"
    ]


def test_materializer_reuses_crop_split_and_repairs_orientation(
    tmp_path: Path,
) -> None:
    adrian, faruq, manifest = _write_fixture(tmp_path)
    output = tmp_path / "prepared"

    result = prepare_sni_fullscene(
        adrian,
        faruq,
        manifest,
        output,
        seed=42,
        link_mode="copy",
    )

    assert result["images_by_split"] == {"train": 2, "test": 2, "val": 1}
    assert result["grouping"]["cross_split_groups"] == 0
    assert result["rotated_clockwise_images"] == 1
    assert result["clamped_boxes"] == 1
    assert result["quarantined_images"] == 1
    assert result["quarantine_reasons"] == {
        "annotation_excluded_by_crop_audit": 1
    }
    assert result["training_executed"] is False
    assert result["training_ready"] is False

    rotated = (
        output
        / "val"
        / "images"
        / "faruq_segmentation__rotated_jpg.rf.ddd.jpg"
    )
    with Image.open(rotated) as image:
        assert image.size == (10, 20)
    label = (
        output
        / "val"
        / "labels"
        / "faruq_segmentation__rotated_jpg.rf.ddd.txt"
    ).read_text(encoding="utf-8")
    assert label.startswith(f"{SNI21_CLASSES.index('biji_hitam')} ")

    train_names = {
        path.name for path in (output / "train" / "images").iterdir()
    }
    assert "adrian_detection__shared_jpg.rf.aaa.jpg" in train_names
    assert "adrian_detection__shared_jpg.rf.bbb.jpg" in train_names

    data_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert list(data_yaml["names"].values()) == list(SNI21_CLASSES)
    quarantine = json.loads(
        (output / "quarantine.json").read_text(encoding="utf-8")
    )
    assert quarantine[0]["image_id"] == "5"

    visual_root = tmp_path / "visual"
    visual = run_sni_fullscene_visual_audit(
        output,
        visual_root,
        dense_samples=2,
        rotated_samples=1,
        seed=42,
    )
    assert visual["splits_rendered"] == ["train", "val"]
    assert visual["test_rendered"] is False
    assert visual["training_executed"] is False
    assert visual["rotated_images_available"] == 1
    assert Path(visual["contact_sheets"]["dense"]).is_file()
    assert Path(visual["contact_sheets"]["rotated_faruq"]).is_file()
