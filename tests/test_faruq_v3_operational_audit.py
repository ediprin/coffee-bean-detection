from pathlib import Path

import pytest
import torch

from coffee_detector.analysis.faruq_v3_operational_audit import (
    _select,
    _select_operating_point,
    audit_faruq_v3_operating_points,
)


def test_class_agnostic_nms_suppresses_cross_class_duplicate() -> None:
    final = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
            [0.0, 0.0, 10.0, 10.0, 0.8, 1.0],
            [20.0, 20.0, 30.0, 30.0, 0.7, 1.0],
        ]
    )
    native = _select(final, 0.25, "native")
    suppressed = _select(final, 0.25, "class_agnostic_nms")
    assert len(native) == 3
    assert len(suppressed) == 2
    assert suppressed[:, 4].tolist() == pytest.approx([0.9, 0.7])


def test_operating_point_maximizes_correct_recall_then_accuracy() -> None:
    rows = [
        {"threshold": 0.1, "correct_decision_recall": 0.5, "conditional_top1_accuracy": 0.7},
        {"threshold": 0.2, "correct_decision_recall": 0.6, "conditional_top1_accuracy": 0.6},
        {"threshold": 0.3, "correct_decision_recall": 0.6, "conditional_top1_accuracy": 0.8},
    ]
    assert _select_operating_point(rows)["threshold"] == 0.3


def test_operational_audit_locks_test_before_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="dikunci pada validation"):
        audit_faruq_v3_operating_points(
            tmp_path / "missing.pt",
            tmp_path / "missing-data",
            tmp_path / "output.json",
            split="test",
        )
