import json
from pathlib import Path

from PIL import Image

from coffee_detector.group_faruq_development import group_faruq_development
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def _write_repaired_fixture(root: Path) -> None:
    manifest = []
    image_id = 0
    for class_id, _ in enumerate(SNI21_CLASSES):
        for parent_index in range(10):
            original_split = "train" if parent_index % 2 == 0 else "val"
            image_name = f"class{class_id}_parent{parent_index}.jpg"
            label_name = f"class{class_id}_parent{parent_index}.txt"
            image_path = root / original_split / "images" / image_name
            label_path = root / original_split / "labels" / label_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (class_id, parent_index, 80)).save(image_path)
            label_path.write_text(
                f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
            manifest.append(
                {
                    "split": original_split,
                    "image_id": image_id,
                    "output_image": str(image_path),
                    "output_label": str(label_path),
                    "source_parent_id": f"class{class_id}_parent{parent_index}",
                    "source_sha256": f"hash-{class_id}-{parent_index}",
                }
            )
            image_id += 1
    (root / "faruq_geometry_repair_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_grouped_split_removes_parent_leakage_and_keeps_all_classes(tmp_path: Path) -> None:
    repaired = tmp_path / "repaired"
    output = tmp_path / "grouped"
    _write_repaired_fixture(repaired)

    result = group_faruq_development(repaired, output, seed=42, val_fraction=0.15)

    assert result["cross_split_parent_identities"] == 0
    assert result["cross_split_exact_hashes"] == 0
    assert result["missing_classes_by_split"] == {"train": [], "val": []}
    assert result["minimum_val_class_support"] >= 1
    assert result["training_executed"] is False
    assert result["test_images_accessed"] is False
    assert not (output / "test").exists()
