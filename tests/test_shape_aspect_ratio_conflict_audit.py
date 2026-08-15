from coffee_detector.analysis.shape_aspect_ratio_conflict_audit import (
    _auc_greater,
    _geometry_state,
    _shape_extremeness,
)


def test_shape_extremeness():
    assert _shape_extremeness(2.0, 1.0, 0.5) == 2.0
    assert _shape_extremeness(1.25, 1.0, 0.5) == 0.5


def test_auc_greater():
    assert _auc_greater([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert _auc_greater([1.0, 2.0], [3.0, 4.0]) == 0.0


def test_geometry_state():
    pair = {
        "low_class": "small",
        "high_class": "large",
        "features": {
            "long_side_norm": {"best_threshold": 0.5},
            "area_norm": {"best_threshold": 0.2},
        },
    }
    assert _geometry_state(
        {"gt_class_name": "small", "long_side_norm": 0.4, "area_norm": 0.1}, pair
    ) == "both_support_gt"
    assert _geometry_state(
        {"gt_class_name": "small", "long_side_norm": 0.6, "area_norm": 0.1}, pair
    ) == "mixed_geometry"
    assert _geometry_state(
        {"gt_class_name": "small", "long_side_norm": 0.6, "area_norm": 0.3}, pair
    ) == "both_support_other"
