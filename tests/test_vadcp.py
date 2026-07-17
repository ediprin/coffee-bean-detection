from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from coffee_detector.audit_vadcp import audit_vadcp_dataset, decode_uncompressed_rle
from coffee_detector.generate_vadcp_dataset import generate_vadcp_dataset
from coffee_detector.evaluate_visibility import (
    count_metrics,
    evaluate_predictions_by_visibility,
)
from coffee_detector.prepare_object_library import prepare_classification_library
from coffee_detector.run_vadcp_visual_audit import run_vadcp_visual_audit
from coffee_detector.vadcp.compositor import CompositionSpec, compose_scene
from coffee_detector.vadcp.library import load_object_library
from coffee_detector.vadcp.masks import binary_mask_rle


def _write_classification_assets(root: Path) -> None:
    colors = {"normal": (80, 130, 60, 255), "defect": (90, 55, 35, 255)}
    for split in ("train", "test"):
        for class_name, color in colors.items():
            folder = root / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                image = Image.new("RGBA", (48, 36), (0, 0, 0, 0))
                draw = ImageDraw.Draw(image)
                draw.ellipse((5 + index, 4, 42, 31), fill=color)
                image.save(folder / f"{class_name}-{index}.png")


def _write_real_dataset(root: Path) -> None:
    names = {0: "defect", 1: "normal"}
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for split_index, split in enumerate(("train", "val", "test")):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        image = Image.new(
            "RGB", (128, 128), (220 - split_index * 12, 222, 224 + split_index * 4)
        )
        image.save(image_dir / f"{split}.jpg")
        (label_dir / f"{split}.txt").write_text(
            "0 0.30 0.50 0.20 0.25\n1 0.70 0.50 0.20 0.25\n",
            encoding="utf-8",
        )


def test_uncompressed_rle_round_trip() -> None:
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:5, 2:7] = True
    mask[3, 4] = False

    decoded = decode_uncompressed_rle(binary_mask_rle(mask))

    assert np.array_equal(decoded, mask)


def test_object_library_uses_train_and_excludes_test(tmp_path: Path) -> None:
    source = tmp_path / "classification"
    output = tmp_path / "library"
    _write_classification_assets(source)

    result = prepare_classification_library(source, output)
    classes, cutouts, info = load_object_library(output)

    assert set(classes.values()) == {"normal", "defect"}
    assert len(cutouts) == 4
    assert {item.source_split for item in cutouts} == {"train"}
    assert result["source"]["skipped_by_split"] == {"test": 4}
    assert info["rejected_splits"] == {}


def test_visibility_compositor_is_deterministic_and_masks_are_consistent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "classification"
    library = tmp_path / "library"
    _write_classification_assets(source)
    prepare_classification_library(source, library)
    _, cutouts, _ = load_object_library(library)
    spec = CompositionSpec(
        canvas_size=(128, 128),
        object_range=(5, 5),
        object_scale=(0.20, 0.28),
        mode="visibility",
        use_shadows=False,
    )
    background = Image.new("RGB", spec.canvas_size, "white")

    first = compose_scene(background, cutouts, spec, random.Random(19))
    second = compose_scene(background, cutouts, spec, random.Random(19))

    assert first.image.tobytes() == second.image.tobytes()
    assert first.target_visibility_bin == second.target_visibility_bin
    assert [item.visibility_ratio for item in first.instances] == [
        item.visibility_ratio for item in second.instances
    ]
    for item in first.instances:
        assert item.visible_mask is not None
        assert not np.any(item.visible_mask & ~item.full_mask)
        expected = float(item.visible_mask.sum()) / float(item.full_mask.sum())
        assert abs(expected - item.visibility_ratio) < 1e-12


def test_generate_and_audit_vadcp_dataset(tmp_path: Path) -> None:
    source = tmp_path / "classification"
    library = tmp_path / "library"
    real = tmp_path / "real"
    output = tmp_path / "vadcp"
    _write_classification_assets(source)
    _write_real_dataset(real)
    prepare_classification_library(source, library)

    manifest = generate_vadcp_dataset(
        real,
        library,
        output,
        synthetic_images=3,
        seed=7,
        mode="visibility",
        canvas_size=128,
        object_range=(4, 6),
        object_scale=(0.18, 0.28),
        use_shadows=False,
    )
    audit = audit_vadcp_dataset(output)
    visual = run_vadcp_visual_audit(
        output, tmp_path / "visual", samples=2, seed=7
    )

    assert manifest["synthetic_images"] == 3
    assert manifest["real_images"] == {"train": 1, "val": 1, "test": 1}
    assert audit["safe_for_training"], audit["errors"]
    assert audit["synthetic_images"] == 3
    assert visual["samples"] == 2
    assert Path(visual["contact_sheet"]).is_file()
    metadata = json.loads(
        (output / "metadata" / "instances_synthetic_train.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["annotations"]
    assert all(
        row["source_split"] in {"train", "unspecified"}
        for row in metadata["annotations"]
    )


def test_visibility_evaluation_ignores_other_bins_and_counts_duplicates() -> None:
    annotations = [
        {
            "image_id": 1,
            "category_id": 0,
            "bbox": [0, 0, 10, 10],
            "visibility_bin": "clear",
            "ignore": 0,
        },
        {
            "image_id": 1,
            "category_id": 0,
            "bbox": [20, 0, 10, 10],
            "visibility_bin": "severe",
            "ignore": 0,
        },
    ]
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.95},
        {"image_id": 1, "category_id": 0, "bbox": [20, 0, 10, 10], "score": 0.90},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.80},
    ]

    visibility = evaluate_predictions_by_visibility(
        predictions, annotations, {0: "bean"}, iou_thresholds=(0.50,)
    )
    counts = count_metrics(predictions, annotations, [1], confidence=0.25)

    assert visibility["clear"]["ap50"] == 1.0
    clear = visibility["clear"]["per_class"]["bean"]
    assert clear["ignored_detections50"] == 1
    assert clear["duplicate_predictions50"] == 1
    assert counts["mean_count_bias"] == 1.0
    assert counts["duplicate_predictions"] == 1
    assert counts["duplicate_prediction_rate"] == 1 / 3
