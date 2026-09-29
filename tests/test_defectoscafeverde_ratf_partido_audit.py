import pytest

from coffee_detector.analysis.defectoscafeverde_ratf_partido_audit import (
    MODEL_ORDER,
    TARGET_CLASS,
    TARGET_CLASS_ID,
    summarize,
)


def _stage(*, accessible=True, matched=True, correct=False, predicted=8):
    return {
        "accessible": accessible,
        "matched": matched,
        "correct": correct,
        "predicted_class_id": TARGET_CLASS_ID if correct else predicted,
        "confidence": 0.5,
        "iou": 0.8,
    }


def _endpoint(*, correct, margin, best_iou=0.8):
    return {
        "raw_top500": _stage(correct=correct),
        "final_conf0001": _stage(correct=correct),
        "raw_best_iou": best_iou,
        "localized_true_score": 0.6 + margin,
        "localized_best_rival_score": 0.6,
        "localized_margin": margin,
        "localized_true_class_rank": 1 if margin >= 0 else 2,
        "localized_best_rival_class_id": 8,
    }


def _records(*, d0=True, active=False, zero=False, zero_iou=0.8):
    return [
        {
            "image": "image.jpg",
            "target_index": 0,
            "class_id": TARGET_CLASS_ID,
            "class_name": TARGET_CLASS,
            "models": {
                "D0DIRECT": _endpoint(correct=d0, margin=0.1),
                "RATF1_ACTIVE": _endpoint(correct=active, margin=-0.1),
                "RATF1_ZERO_RESIDUAL": _endpoint(
                    correct=zero, margin=0.05 if zero else -0.05, best_iou=zero_iou
                ),
            },
        }
    ]


def test_target_and_endpoint_order_are_frozen():
    assert TARGET_CLASS == "partido"
    assert TARGET_CLASS_ID == 9
    assert MODEL_ORDER == ("D0DIRECT", "RATF1_ACTIVE", "RATF1_ZERO_RESIDUAL")


def test_zero_residual_recovery_attributes_inference_residual():
    result = summarize(_records(d0=True, active=False, zero=True))
    assert result["attribution"] == "INFERENCE_TEXTURE_RESIDUAL_CONTRIBUTES_TO_CLASS_REGRESSION"
    assert result["raw_localization_active_zero_identical"] is True
    assert result["stages"]["raw_top500"]["RATF1_ZERO_RESIDUAL"][
        "correct_decision_recall"
    ] == 1.0


def test_zero_residual_still_wrong_attributes_training_path():
    result = summarize(_records(d0=True, active=False, zero=False))
    assert result["attribution"] == "TRAINING_PATH_DOMINANT_ZERO_RESIDUAL_STILL_BELOW_D0"
    assert result["localized_score_diagnostic"]["RATF1_ACTIVE"][
        "localized_top1_rate"
    ] == 0.0


def test_endpoint_order_and_localization_intervention_are_guarded():
    records = _records()
    records[0]["models"] = dict(reversed(list(records[0]["models"].items())))
    with pytest.raises(RuntimeError, match="tidak lengkap"):
        summarize(records)
    result = summarize(_records(zero_iou=0.7))
    assert result["raw_localization_active_zero_identical"] is False
