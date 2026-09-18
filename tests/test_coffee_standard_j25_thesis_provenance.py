import json
import hashlib
import zipfile
from pathlib import Path

import yaml

from coffee_detector.analysis.coffee_standard_j25_thesis_provenance import (
    audit_j25_thesis_provenance,
)


def _write_archive(path: Path) -> None:
    data = {
        "nc": 25,
        "names": [f"class-{index}" for index in range(25)],
        "roboflow": {
            "workspace": "tes-rcphs",
            "project": "coffee-detection-with-standard",
            "version": 8,
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.yaml", yaml.safe_dump(data))
        archive.writestr(
            "README.roboflow.txt",
            "The dataset includes 993 images. Augmentation was applied to create 3 versions of each source image.",
        )
        split_counts = {"train": (813, 11160), "valid": (113, 1606), "test": (67, 1160)}
        for split, (images, instances) in split_counts.items():
            base, remainder = divmod(instances, images)
            for index in range(images):
                archive.writestr(f"{split}/images/image-{index}.jpg", f"{split}-{index}".encode())
                count = base + int(index < remainder)
                archive.writestr(
                    f"{split}/labels/image-{index}.txt",
                    "\n".join("0 0.5 0.5 0.1 0.1" for _ in range(count)),
                )


def test_thesis_lineage_arithmetic_passes(tmp_path: Path) -> None:
    archive = tmp_path / "data_aug_11.zip"
    _write_archive(archive)
    expected_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = audit_j25_thesis_provenance(
        archive, tmp_path / "summary.json", expected_sha256=expected_sha256
    )

    assert result["decision"] == "PASS_THESIS_LINEAGE_WITH_ONE_ANNOTATION_DISCREPANCY"
    assert result["dataset_role"] == "PRIMARY_DATASET_PROVENANCE_SUPPORTED"
    assert result["observed"]["inferred_raw_images"] == 451
    assert result["observed"]["inferred_train_annotations_before_augmentation"] == 3720
    assert result["test_annotation_delta_export_minus_thesis"] == -1
    assert result["training_authorized"] is False
    assert json.loads((tmp_path / "summary.json").read_text())["decision"].startswith("PASS")
