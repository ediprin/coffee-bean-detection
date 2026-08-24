import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from coffee_detector.audit_faruq_mask_geometry import audit_faruq_mask_geometry
from coffee_detector.repair_faruq_mask_geometry import repair_faruq_mask_geometry


def _write_split(root: Path, split: str, image_id: int, stored: Image.Image) -> None:
    split_root = root / split
    split_root.mkdir(parents=True)
    stored.save(split_root / f"bean_{image_id}.jpg", quality=100, subsampling=0)
    payload = {
        "images": [
            {
                "id": image_id,
                "file_name": f"bean_{image_id}.jpg",
                "width": 60,
                "height": 100,
            }
        ],
        "annotations": [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [8, 12, 20, 27],
                "segmentation": [[8, 12, 28, 12, 28, 39, 8, 39]],
            }
        ],
        "categories": [{"id": 1, "name": "biji_normal"}],
    }
    (split_root / "_annotations.coco.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_repair_uses_frozen_audit_and_keeps_test_locked(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    expected = Image.new("RGB", (60, 100), "white")
    ImageDraw.Draw(expected).rectangle((8, 12, 28, 39), fill=(80, 40, 20))
    _write_split(raw, "train", 1, expected.transpose(Image.Transpose.ROTATE_270))
    _write_split(raw, "valid", 2, expected.transpose(Image.Transpose.ROTATE_180))
    (raw / "test").mkdir()
    (raw / "test" / "broken.json").write_text("not json", encoding="utf-8")

    audit_root = tmp_path / "audit"
    audit = audit_faruq_mask_geometry(
        raw, audit_root, score_long_side=100, min_improvement=0.01
    )
    repaired_root = tmp_path / "repaired"
    summary = repair_faruq_mask_geometry(raw, audit["records"], repaired_root)

    assert summary["test_images_accessed"] is False
    assert summary["training_executed"] is False
    assert summary["training_ready"] is False
    assert summary["counters"]["images_written"] == 2
    assert not (repaired_root / "test").exists()
    assert (repaired_root / "train/labels/bean_1.txt").is_file()
    assert (repaired_root / "val/labels/bean_2.txt").is_file()

    with Image.open(repaired_root / "train/images/bean_1.jpg") as image:
        pixels = np.asarray(image)
    assert pixels[20, 15].mean() < 150
    with Image.open(repaired_root / "val/images/bean_2.jpg") as image:
        pixels = np.asarray(image)
    assert pixels[20, 15].mean() < 150

    post = audit_faruq_mask_geometry(
        repaired_root, tmp_path / "post", score_long_side=100, min_improvement=0.01
    )
    assert post["flagged_fraction"] == 0.0
