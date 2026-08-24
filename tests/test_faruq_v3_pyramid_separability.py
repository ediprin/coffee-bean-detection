import numpy as np
import pytest

from coffee_detector.analysis.faruq_v3_pyramid_separability import (
    classification_metrics,
    decide_pyramid_route,
    fit_balanced_ridge_probe,
)


def _result(macro: float, bottom3: float, worst: float = 0.4) -> dict:
    return {
        "validation": {
            "macro_f1": macro,
            "bottom3_f1": bottom3,
            "worst_class_f1": worst,
        }
    }


def test_balanced_ridge_probe_separates_fixed_clusters() -> None:
    train_features = np.asarray(
        [
            [3.0, 0.0],
            [2.8, 0.1],
            [0.0, 3.0],
            [0.1, 2.8],
            [-3.0, -3.0],
            [-2.8, -3.1],
        ],
        dtype=np.float32,
    )
    train_labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    validation_features = np.asarray(
        [[2.9, 0.0], [0.0, 2.9], [-2.9, -3.0]], dtype=np.float32
    )
    _, logits = fit_balanced_ridge_probe(
        train_features,
        train_labels,
        validation_features,
        num_classes=3,
    )
    assert logits.argmax(axis=1).tolist() == [0, 1, 2]


def test_classification_metrics_reports_lower_tail_and_top3() -> None:
    logits = np.asarray(
        [
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
            [0.0, 3.0, 2.0],
            [0.0, 3.0, 2.0],
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
        ]
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    result = classification_metrics(logits, labels, num_classes=3)
    assert result["accuracy"] == pytest.approx(5 / 6)
    assert result["top3_accuracy"] == 1.0
    assert result["worst_class_f1"] < 1.0
    assert len(result["per_class"]) == 3


def test_route_gate_authorizes_fusion_only_for_material_lower_tail_safe_gain() -> None:
    results = {
        "P3": _result(0.78, 0.53),
        "P4": _result(0.76, 0.52),
        "P5": _result(0.72, 0.50),
        "P3+P4": _result(0.79, 0.54),
        "P3+P4+P5": _result(0.81, 0.55),
    }
    decision = decide_pyramid_route(results)
    assert decision["decision"] == "PASS"
    assert decision["next_action"] == "AUTHORIZE_MULTILEVEL_CLASSIFICATION_PROTOCOL"
    assert decision["detector_training_authorized"] is False


def test_route_gate_stops_when_every_representation_is_weak() -> None:
    results = {
        "P3": _result(0.60, 0.30),
        "P4": _result(0.62, 0.31),
        "P5": _result(0.58, 0.25),
        "P3+P4": _result(0.64, 0.33),
        "P3+P4+P5": _result(0.65, 0.35),
    }
    decision = decide_pyramid_route(results)
    assert decision["decision"] == "FAIL"
    assert (
        decision["next_action"]
        == "STOP_HEAD_ONLY_SEARCH_REPRESENTATION_OR_LABEL_LIMITED"
    )


def test_route_gate_can_select_high_resolution_without_fusion_gain() -> None:
    results = {
        "P3": _result(0.80, 0.55),
        "P4": _result(0.77, 0.54),
        "P5": _result(0.72, 0.50),
        "P3+P4": _result(0.79, 0.55),
        "P3+P4+P5": _result(0.79, 0.54),
    }
    decision = decide_pyramid_route(results)
    assert decision["decision"] == "PASS"
    assert decision["next_action"] == "AUTHORIZE_HIGH_RES_CLASSIFICATION_PROTOCOL"
