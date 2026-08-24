from pathlib import Path

import pytest
import torch

from coffee_detector.analysis.faruq_v3_operational_audit import (
    _select,
    _select_operating_point,
    audit_faruq_v3_operating_points,
    correct_existing_operational_payload,
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


def test_operating_point_maximizes_correct_decision_f1() -> None:
    rows = [
        {
            "threshold": 0.01,
            "correct_decision_f1": 0.40,
            "correct_decision_precision": 0.25,
            "correct_decision_recall": 0.80,
            "conditional_top1_accuracy": 0.80,
            "mean_predictions_per_image": 8.0,
        },
        {
            "threshold": 0.05,
            "correct_decision_f1": 0.60,
            "correct_decision_precision": 0.62,
            "correct_decision_recall": 0.58,
            "conditional_top1_accuracy": 0.70,
            "mean_predictions_per_image": 1.8,
        },
    ]
    assert _select_operating_point(rows)["threshold"] == 0.05


def test_existing_report_is_corrected_without_inference() -> None:
    payload = {
        "protocol": "faruq-v3-operational-audit-v1",
        "evaluation_split": "val",
        "test_images_accessed": False,
        "rows": [
            {
                "policy": "native",
                "threshold": 0.25,
                "correct": 40,
                "predictions": 80,
                "targets": 100,
                "proposal_accessibility": 0.70,
                "conditional_top1_accuracy": 0.60,
                "mean_predictions_per_image": 0.80,
            },
            {
                "policy": "native",
                "threshold": 0.01,
                "correct": 56,
                "predictions": 200,
                "targets": 100,
                "proposal_accessibility": 0.99,
                "conditional_top1_accuracy": 0.57,
                "mean_predictions_per_image": 2.00,
            },
            {
                "policy": "class_agnostic_nms",
                "threshold": 0.05,
                "correct": 55,
                "predictions": 96,
                "targets": 100,
                "proposal_accessibility": 0.96,
                "conditional_top1_accuracy": 0.57,
                "mean_predictions_per_image": 0.96,
            },
        ],
        "selected": {"policy": "native", "threshold": 0.01},
        "selected_per_class": {"example": {"targets": 1}},
    }

    corrected = correct_existing_operational_payload(payload)

    assert corrected["protocol"] == "faruq-v3-operational-audit-v2"
    assert corrected["selected"]["policy"] == "class_agnostic_nms"
    assert corrected["selected"]["threshold"] == 0.05
    assert corrected["comparison"]["postprocessing_improves_operating_point"]
    assert corrected["comparison"]["classification_refinement_still_justified"]
    assert "selected_per_class" not in corrected
    assert "legacy_selected_per_class" in corrected


def test_operational_audit_locks_test_before_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="dikunci pada validation"):
        audit_faruq_v3_operating_points(
            tmp_path / "missing.pt",
            tmp_path / "missing-data",
            tmp_path / "output.json",
            split="test",
        )
