import numpy as np
import pytest

from coffee_detector.analysis.faruq_v3_predicted_roi_transfer import (
    decide_predicted_roi_transfer,
    fit_pca_projection,
)


def _result(macro: float, bottom3: float) -> dict:
    return {"validation": {"macro_f1": macro, "bottom3_f1": bottom3}}


def test_pca_projection_uses_fixed_capacity_for_both_splits() -> None:
    generator = np.random.default_rng(42)
    train = generator.normal(size=(40, 10)).astype(np.float32)
    validation = generator.normal(size=(12, 10)).astype(np.float32)
    projected_train, projected_val, metadata = fit_pca_projection(
        train, validation, components=4
    )
    assert projected_train.shape == (40, 4)
    assert projected_val.shape == (12, 4)
    assert metadata["fit_split"] == "train"
    assert 0.0 < metadata["explained_variance_fraction"] <= 1.0


def test_pca_projection_rejects_impossible_rank() -> None:
    with pytest.raises(ValueError, match="rank maksimum"):
        fit_pca_projection(
            np.ones((3, 5), dtype=np.float32),
            np.ones((2, 5), dtype=np.float32),
            components=4,
        )


def test_predicted_roi_gate_passes_only_when_raw_and_capacity_control_pass() -> None:
    results = {
        "P5_RAW": _result(0.72, 0.52),
        "P3+P4+P5_RAW": _result(0.78, 0.60),
        "P5_CM128": _result(0.73, 0.53),
        "P3+P4+P5_CM128": _result(0.77, 0.58),
    }
    decision = decide_predicted_roi_transfer(
        results, {"train": 0.98, "val": 0.99}, gt_fusion_macro_f1=0.79
    )
    assert decision["decision"] == "PASS"
    assert decision["next_action"] == "AUTHORIZE_MULTILEVEL_HEAD_STATIC_AUDIT"
    assert decision["detector_training_authorized"] is False


def test_predicted_roi_gate_attributes_capacity_only_gain() -> None:
    results = {
        "P5_RAW": _result(0.70, 0.50),
        "P3+P4+P5_RAW": _result(0.76, 0.56),
        "P5_CM128": _result(0.74, 0.55),
        "P3+P4+P5_CM128": _result(0.75, 0.56),
    }
    decision = decide_predicted_roi_transfer(
        results, {"train": 0.98, "val": 0.99}, gt_fusion_macro_f1=0.79
    )
    assert decision["decision"] == "FAIL"
    assert decision["next_action"] == "STOP_FUSION_GAIN_EXPLAINED_BY_CAPACITY"


def test_predicted_roi_gate_stops_low_coverage_first() -> None:
    results = {
        "P5_RAW": _result(0.70, 0.50),
        "P3+P4+P5_RAW": _result(0.76, 0.56),
        "P5_CM128": _result(0.72, 0.52),
        "P3+P4+P5_CM128": _result(0.76, 0.56),
    }
    decision = decide_predicted_roi_transfer(
        results, {"train": 0.89, "val": 0.99}, gt_fusion_macro_f1=0.79
    )
    assert decision["next_action"] == "STOP_PREDICTED_ROI_COVERAGE"
