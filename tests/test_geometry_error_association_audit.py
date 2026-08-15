from coffee_detector.analysis.geometry_error_association_audit import (
    _geometry_prediction,
    _geometry_state,
    _safe_rate,
)


def test_geometry_prediction_threshold():
    assert _geometry_prediction(0.1, 0.2, "small", "large") == "small"
    assert _geometry_prediction(0.3, 0.2, "small", "large") == "large"


def test_geometry_state_both_support_gt():
    pair = {
        "low_class": "small",
        "high_class": "large",
        "features": {
            "long_side_norm": {"best_threshold": 0.2},
            "area_norm": {"best_threshold": 0.05},
        },
    }
    row = {"gt_class_name": "small", "long_side_norm": 0.1, "area_norm": 0.03}
    assert _geometry_state(row, pair)["state"] == "both_support_gt"


def test_geometry_state_mixed():
    pair = {
        "low_class": "small",
        "high_class": "large",
        "features": {
            "long_side_norm": {"best_threshold": 0.2},
            "area_norm": {"best_threshold": 0.05},
        },
    }
    row = {"gt_class_name": "small", "long_side_norm": 0.1, "area_norm": 0.08}
    assert _geometry_state(row, pair)["state"] == "mixed_geometry"


def test_safe_rate_zero_denominator():
    assert _safe_rate(1, 0) == 0.0
    assert _safe_rate(1, 2) == 0.5
