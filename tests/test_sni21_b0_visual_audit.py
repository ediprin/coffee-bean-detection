import json

from coffee_detector.run_sni21_b0_visual_audit import (
    _select_rows,
    _validate_record_alignment,
)


def test_record_alignment_accepts_yolo_rounding() -> None:
    scene = {
        "annotations": [
            {"category_id": 2, "bbox": [10, 20, 30, 40], "z_order": 0}
        ]
    }
    record = {
        "ground_truth": [
            {"class_id": 2, "xyxy": [10.001, 19.999, 40.001, 60.0]}
        ],
        "ground_truth_diagnosis": [
            {"ground_truth_index": 0, "category": "proposal_miss"}
        ],
    }

    _validate_record_alignment(scene, record)


def test_row_selection_round_robins_transitions() -> None:
    rows = [
        {"transition": "a", "class_id": index % 2, "id": index}
        for index in range(8)
    ] + [
        {"transition": "b", "class_id": index % 2, "id": 100 + index}
        for index in range(2)
    ]

    selected = _select_rows(rows, samples=4, seed=42)

    assert len(selected) == 4
    assert {row["transition"] for row in selected} == {"a", "b"}
