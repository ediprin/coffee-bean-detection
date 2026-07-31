import json

import numpy as np
import pytest
import yaml
from PIL import Image

from coffee_detector.run_sni21_local_context_control import (
    classify_background_retention,
    isolated_context_box,
    prepare_local_context_dataset,
)
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def test_context_rejects_edges_and_neighboring_objects() -> None:
    target = [40, 40, 60, 60]
    assert isolated_context_box(target, [target], 0, (100, 100), context_multiplier=3) == [20, 20, 80, 80]
    assert isolated_context_box([0, 0, 20, 20], [[0, 0, 20, 20]], 0, (100, 100), context_multiplier=3) is None
    assert isolated_context_box(target, [target, [70, 45, 80, 55]], 0, (100, 100), context_multiplier=3) is None


@pytest.mark.parametrize(
    ("map_retention", "class_retention", "expected"),
    [
        (0.9, 0.8, "procedural_background_not_primary_cause"),
        (0.7, 0.9, "procedural_background_material_partial_cause"),
        (0.9, 0.4, "procedural_background_dominant_cause"),
    ],
)
def test_background_gate_is_frozen(map_retention, class_retention, expected) -> None:
    assert classify_background_retention(map_retention, class_retention) == expected


def test_prepare_local_context_creates_exact_paired_geometry(tmp_path) -> None:
    real = tmp_path / "real"
    library = tmp_path / "library"
    output = tmp_path / "output"
    (real / "val/images").mkdir(parents=True)
    (real / "val/labels").mkdir(parents=True)
    (real / "train/images").mkdir(parents=True)
    (real / "train/labels").mkdir(parents=True)
    names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    (real / "data.yaml").write_text(yaml.safe_dump({"path": str(real), "train": "train/images", "val": "val/images", "names": names}, sort_keys=False), encoding="utf-8")

    assets = []
    for class_id, class_name in enumerate(SNI21_CLASSES):
        image_name = f"faruq_segmentation__parent{class_id}_jpg.rf.abcdef012345.jpg"
        canvas = np.full((100, 100, 3), 230, dtype=np.uint8)
        canvas[40:60, 40:60] = (80 + class_id, 40, 20)
        Image.fromarray(canvas).save(real / "val/images" / image_name)
        (real / "val/labels" / image_name.replace(".jpg", ".txt")).write_text(f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        asset_path = library / f"assets/{class_name}/asset{class_id}.png"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        rgba[:, :, :3] = (80 + class_id, 40, 20)
        rgba[:, :, 3] = 255
        Image.fromarray(rgba, "RGBA").save(asset_path)
        assets.append({"asset_id": f"asset{class_id}", "class_id": class_id, "class_name": class_name, "image": f"assets/{class_name}/asset{class_id}.png", "intrinsic_aspect_ratio": 1.0, "source_split": "val", "source_dataset": "faruq_segmentation", "source_parent_id": f"parent{class_id}jpg"})
    (library / "object_library.json").write_text(json.dumps({"classes": {str(k): v for k, v in names.items()}, "assets": assets}), encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"format": "coffee_detector.scene_calibration.v7", "scene_counts": [1], "object_long_sides": [0.2], "bbox_width_height_ratios": [1.0], "scene_scale_medians": [0.2], "within_scene_scale_ratios": [1.0], "background_colors": [[235, 234, 230]], "background_gradient_std": [2.0], "background_sensor_std": [1.0], "source_images": 21, "source_boxes": 21, "scene_count_scale_pairs": [], "bbox_width_height_ratios_by_class": {}, "canvas_width_height_ratios": [1.0], "class_probabilities": {}, "split": "val"}), encoding="utf-8")

    report = prepare_local_context_dataset(real, library, profile, output, max_per_class=1, minimum_samples=21, minimum_classes=21)

    assert report["samples"] == 21
    for arm in report["arms"]:
        assert len(list((output / arm / "val/images").glob("*.png"))) == 21
        labels = [path.read_text() for path in (output / arm / "val/labels").glob("*.txt")]
        assert len(labels) == 21
    assert not (output / "test").exists()
