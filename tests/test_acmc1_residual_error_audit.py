import numpy as np

from coffee_detector.analysis.acmc1_residual_error_audit import (
    IOU_THRESHOLDS,
    _aggregate,
    _attribution_label,
    _evidence_flags,
    _profile_from_ap_matrix,
)


def test_profile_extracts_ap50_ap75_ap95_and_mean():
    names = {0: "a", 1: "b"}
    matrix = np.array([
        [0.95, 0.94, 0.93, 0.92, 0.91, 0.80, 0.78, 0.75, 0.70, 0.60],
        [0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81],
    ])
    result = _profile_from_ap_matrix(matrix, np.array([0, 1]), names)
    assert len(IOU_THRESHOLDS) == 10
    assert result["a"]["ap50"] == 0.95
    assert result["a"]["ap75"] == 0.80
    assert result["a"]["ap95"] == 0.60
    assert np.isclose(result["a"]["map50_95"], matrix[0].mean())


def test_evidence_flags_and_attribution_are_frozen():
    row = {
        "ap50": 0.86,
        "ap50_to_ap75_drop": 0.04,
        "class_accuracy_given_iou50_match": 0.78,
        "proposal_accessibility_iou50": 0.97,
    }
    flags = _evidence_flags(row)
    assert flags["low_ap50"] is True
    assert flags["classification_headroom_material"] is True
    assert flags["high_iou_localization_drop"] is False
    assert _attribution_label(row) == "classification_or_ranking_limited"


def test_localization_attribution():
    row = {
        "ap50": 0.96,
        "ap50_to_ap75_drop": 0.14,
        "class_accuracy_given_iou50_match": 0.95,
        "proposal_accessibility_iou50": 0.97,
    }
    assert _attribution_label(row) == "high_iou_localization_limited"


def test_aggregate_uses_three_seeds_and_keeps_seed_agreement():
    base = {
        "ap50": 0.95,
        "ap75": 0.90,
        "ap95": 0.70,
        "map50_95": 0.88,
        "ap50_to_ap75_drop": 0.05,
        "ap75_to_ap95_drop": 0.20,
        "proposal_accessibility_iou50": 0.98,
        "matched_recall_iou50": 0.94,
        "class_accuracy_given_iou50_match": 0.80,
        "classification_headroom_iou50": 0.20,
        "attribution": "classification_or_ranking_limited",
    }
    per_seed = {
        "42": {"per_class": {"x": dict(base)}},
        "123": {"per_class": {"x": dict(base)}},
        "2026": {"per_class": {"x": dict(base)}},
    }
    rows = _aggregate(per_seed)
    assert len(rows) == 1
    assert rows[0]["class_name"] == "x"
    assert rows[0]["attribution"] == "classification_or_ranking_limited"
    assert rows[0]["attribution_seed_agreement"] == 3
