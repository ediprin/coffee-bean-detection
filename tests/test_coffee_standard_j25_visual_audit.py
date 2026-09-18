import json
from pathlib import Path

import yaml
from PIL import Image

from coffee_detector.analysis.coffee_standard_j25_visual_audit import (
    audit_coffee_standard_j25_visuals,
)
from coffee_detector.data.prepare_coffee_standard_primary import (
    J25_CLASSES,
    prepare_coffee_standard_primary,
)


def _source(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": {index: name for index, name in enumerate(J25_CLASSES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    for parent in range(20):
        stem = f"bean-{parent}.rf.{parent:032x}"
        image = root / "train" / "images" / f"{stem}.jpg"
        Image.new(
            "RGB", (80, 60), ((17 * parent) % 255, (43 * parent) % 255, (79 * parent) % 255)
        ).save(image)
        rows = []
        for class_id in range(len(J25_CLASSES)):
            width = 0.05 + class_id * 0.002
            rows.append(f"{class_id} 0.5 0.5 {width:.4f} 0.08")
        (root / "train" / "labels" / f"{stem}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
    # A generated sibling with one missing class creates a reviewable count
    # disagreement while remaining within the same identity component.
    stem = "bean-0.rf.ffffffffffffffffffffffffffffffff"
    Image.new("RGB", (80, 60), (81, 101, 121)).save(root / "valid" / "images" / f"{stem}.jpg")
    (root / "valid" / "labels" / f"{stem}.txt").write_text(
        "\n".join(f"{class_id} 0.5 0.5 0.08 0.08" for class_id in range(24)) + "\n",
        encoding="utf-8",
    )


def test_j25_visual_audit_writes_review_evidence_and_never_authorizes_training(
    tmp_path: Path,
) -> None:
    source = tmp_path / "archive" / "nested"
    _source(source)
    grouped = tmp_path / "grouped"
    prepare_coffee_standard_primary(source, grouped, seed=42, link_mode="copy")

    result = audit_coffee_standard_j25_visuals(
        grouped,
        tmp_path / "archive",
        tmp_path / "visual",
        samples_per_class=1,
        flagged_limit=8,
    )

    assert result["decision"] == "PENDING_HUMAN_VISUAL_REVIEW"
    assert result["training_authorized"] is False
    assert result["training_executed"] is False
    assert result["model_inference_executed"] is False
    assert result["test_model_evaluation_executed"] is False
    assert set(result["class_review_sheets"]) == {"train", "val", "test"}
    assert all(Path(path).is_file() for path in result["class_review_sheets"].values())
    assert Path(result["geometry_flag_sheet"]).is_file()
    assert Path(result["review_form"]).is_file()
    assert result["sibling_consistency"]["parents_with_class_count_disagreement"] == 1
    payload = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    assert payload["next_action"] == "complete_human_review_form_and_freeze_label_decision"
