import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from coffee_detector.data.prepare_defectoscafeverde_grouped import (
    NAMES,
    SourceRecord,
    _box_rows,
    assign_grouped_splits,
    infer_physical_group,
)


def test_even_odd_source_views_share_physical_group() -> None:
    assert infer_physical_group("A0.png") == infer_physical_group("A1.png")
    assert infer_physical_group("Ca42.png") == infer_physical_group("Ca43.png")
    assert infer_physical_group("Ca43.png") != infer_physical_group("Ca44.png")


def test_source_name_contract_rejects_generated_roboflow_name() -> None:
    try:
        infer_physical_group("A0_png.rf.abc123.jpg")
    except ValueError as error:
        assert "pola" in str(error)
    else:
        raise AssertionError("generated filename must be rejected")


def test_polygon_box_is_converted_to_yolo_detection_row() -> None:
    rows, counts = _box_rows(
        {
            "width": 640,
            "height": 480,
            "boxes": [
                {
                    "type": "polygon",
                    "label": "agrio",
                    "x": "320",
                    "y": "240",
                    "width": "64",
                    "height": "48",
                    "points": [[288, 216], [352, 264]],
                }
            ],
        }
    )
    assert rows == ["0 0.5000000000 0.5000000000 0.1000000000 0.1000000000"]
    assert counts == Counter({0: 1})


def _record(tmp_path: Path, group: int, side: int, class_id: int) -> SourceRecord:
    image = tmp_path / f"X{group * 2 + side}.jpg"
    label = tmp_path / f"X{group * 2 + side}.txt"
    Image.new("RGB", (8, 8), (group % 255, class_id, side)).save(image)
    label.write_text(f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return SourceRecord(
        image_id=f"id-{group}-{side}",
        source_name=f"X{group * 2 + side}.png",
        source_split="train",
        group_id=f"x-{group:05d}",
        image_path=image,
        label_path=label,
        class_counts=Counter({class_id: 1}),
        sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
    )


def test_assignment_keeps_views_together_and_all_classes_present(tmp_path: Path) -> None:
    records = [
        _record(tmp_path, group, side, group % len(NAMES))
        for group in range(120)
        for side in (0, 1)
    ]
    assignments = assign_grouped_splits(records, seed=42)
    owner = {}
    for split, rows in assignments.items():
        assert {class_id for row in rows for class_id in row.class_counts} == set(range(len(NAMES)))
        for row in rows:
            assert owner.setdefault(row.group_id, split) == split
    assert sum(map(len, assignments.values())) == len(records)
    assert abs(len(assignments["train"]) / len(records) - 0.70) < 0.03
    assert abs(len(assignments["val"]) / len(records) - 0.20) < 0.03
    assert abs(len(assignments["test"]) / len(records) - 0.10) < 0.03


def test_no_api_key_is_part_of_source_record_schema() -> None:
    fields = set(SourceRecord.__dataclass_fields__)
    assert "api_key" not in fields
    assert "token" not in fields
