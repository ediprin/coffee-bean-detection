import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from coffee_detector.analysis.sni21_structured_target_audit import (
    audit_sni21_structured_targets,
)
from coffee_detector.sni21_ontology import SNI21_CLASSES


def _dataset(root: Path, *, include_test: bool = False) -> Path:
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": {i: name for i, name in enumerate(SNI21_CLASSES)}}),
        encoding="utf-8",
    )
    manifest = []
    splits = ["train", "val"] + (["test"] if include_test else [])
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for index, class_name in enumerate(SNI21_CLASSES):
            name = f"{split}_{index}.jpg"
            Image.new("RGB", (16, 16), "white").save(root / split / "images" / name)
            (root / split / "labels" / f"{Path(name).stem}.txt").write_text(
                f"{index} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
            if split != "test":
                manifest.append(
                    {
                        "group_id": f"{split}-group-{index}",
                        "output_split": split,
                        "output_image": f"/old/path/{name}",
                        "output_label": f"/old/path/{Path(name).stem}.txt",
                        "class_counts": [int(value == index) for value in range(21)],
                    }
                )
    (root / "faruq_grouped_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def test_structured_target_audit_counts_support_and_locks_training(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "data")
    output = tmp_path / "report.json"
    report = audit_sni21_structured_targets(data, output)

    assert report["decision"] == "AUDIT_COMPLETE_NO_TRAINING_AUTHORIZATION"
    assert report["training_executed"] is False
    assert report["inference_executed"] is False
    assert report["test_images_accessed"] is False
    assert report["test_locked"] is True
    assert report["instances_by_split"] == {"train": 21, "val": 21}
    assert "physical_size_mm" in report["blocked_tasks"]
    assert "relative_completeness" in report["domain_expert_review_tasks"]
    assert report["statistically_ready"] is False
    assert output.is_file()


def test_structured_target_audit_rejects_available_test(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "data", include_test=True)
    with pytest.raises(RuntimeError, match="tidak boleh menyediakan test"):
        audit_sni21_structured_targets(data, tmp_path / "report.json")
