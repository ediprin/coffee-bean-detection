from coffee_detector.run_visual_audit import select_audit_rows


def test_select_audit_rows_prioritizes_count_errors() -> None:
    rows = [
        {"image": "easy", "absolute_count_error": 0, "minimum_confidence": 0.2},
        {"image": "hard", "absolute_count_error": 3, "minimum_confidence": 0.9},
        {"image": "medium", "absolute_count_error": 1, "minimum_confidence": 0.1},
    ]

    selected = select_audit_rows(rows, samples=2, seed=42)

    assert [row["image"] for row in selected] == ["hard", "medium"]

