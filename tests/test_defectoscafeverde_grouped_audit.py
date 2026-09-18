import hashlib
import json
from pathlib import Path

import yaml
from PIL import Image

from coffee_detector.analysis.defectoscafeverde_grouped_audit import audit_grouped_dataset
from coffee_detector.data.prepare_defectoscafeverde_grouped import NAMES


def test_audit_fails_a_tiny_but_structurally_valid_fixture(tmp_path: Path) -> None:
    rows = []
    for split_index, split in enumerate(("train", "val", "test")):
        for class_id in range(len(NAMES)):
            number = (split_index * len(NAMES) + class_id) * 2
            source_name = f"X{number}.png"
            image_id = f"id-{split_index}-{class_id}"
            stem = f"X{number}__{image_id}"
            image = tmp_path / split / "images" / f"{stem}.jpg"
            label = tmp_path / split / "labels" / f"{stem}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (class_id, split_index, 0)).save(image)
            label.write_text(f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            rows.append(
                {
                    "image_id": image_id,
                    "source_name": source_name,
                    "source_split": split,
                    "physical_group_id": f"x-{number // 2:05d}",
                    "output_split": split,
                    "image": str(image.relative_to(tmp_path)).replace("\\", "/"),
                    "label": str(label.relative_to(tmp_path)).replace("\\", "/"),
                    "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                }
            )
    (tmp_path / "grouped_manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "grouped_summary.json").write_text(
        json.dumps({"augmentation_applied": False, "test_accessed": False}), encoding="utf-8"
    )
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": NAMES}), encoding="utf-8")

    report = audit_grouped_dataset(tmp_path, tmp_path / "audit.json")

    assert report["decision"] == "FAIL_GROUPED_DATASET_GATE"
    assert report["gates"]["zero_invalid_files_or_labels"] is True
    assert report["gates"]["all_12_classes_present_each_split"] is True
    assert report["gates"]["exactly_4038_original_images"] is False
