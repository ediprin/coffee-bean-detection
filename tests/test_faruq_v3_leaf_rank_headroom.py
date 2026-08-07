import pytest

from coffee_detector.analysis.faruq_v3_leaf_rank_headroom import (
    decide_leaf_rank_headroom,
    summarize_true_class_ranks,
)


def test_rank_summary_reports_recoverable_top3_headroom() -> None:
    result = summarize_true_class_ranks([1, 1, 2, 3, 4])
    assert result["conditional_top1_accuracy"] == pytest.approx(0.4)
    assert result["conditional_top3_accuracy"] == pytest.approx(0.8)
    assert result["top3_recovery_over_top1"] == pytest.approx(0.4)
    assert result["rank_distribution"] == {"1": 2, "2": 1, "3": 1, "4": 1}


def test_leaf_rank_gate_routes_without_authorizing_training() -> None:
    passed = decide_leaf_rank_headroom(
        {
            "conditional_top1_accuracy": 0.60,
            "conditional_top3_accuracy": 0.84,
            "top3_recovery_over_top1": 0.24,
        }
    )
    assert passed["decision"] == "PASS"
    assert passed["next_action"] == "AUTHORIZE_LEAF_RERANKING_PROTOCOL"
    assert passed["training_authorized"] is False

    limited = decide_leaf_rank_headroom(
        {
            "conditional_top1_accuracy": 0.60,
            "conditional_top3_accuracy": 0.75,
            "top3_recovery_over_top1": 0.15,
        }
    )
    assert limited["next_action"] == "STOP_RERANKING_REPRESENTATION_LIMITED"

