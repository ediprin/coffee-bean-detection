from __future__ import annotations

from coffee_detector.experiments.run_faruq_v3_acmc2_paired_confirmation import (
    ALL_SEEDS,
    CONFIRMATION_SEEDS,
    METRICS,
    paired_confirmation_decision,
)


def _record(d0: float, d0ft: float, acmc1: float, acmc2: float) -> dict:
    return {
        "results": {
            arm: {metric: value for metric in METRICS}
            for arm, value in (
                ("D0", d0),
                ("D0FT", d0ft),
                ("ACMC1", acmc1),
                ("ACMC2", acmc2),
            )
        }
    }


def test_confirmation_seed_lock_is_42_123_2026() -> None:
    assert ALL_SEEDS == (42, 123, 2026)
    assert CONFIRMATION_SEEDS == (123, 2026)


def test_paired_confirmation_passes_consistent_acmc2_improvement() -> None:
    per_seed = {
        "42": _record(0.80, 0.86, 0.875, 0.881),
        "123": _record(0.81, 0.865, 0.878, 0.884),
        "2026": _record(0.79, 0.858, 0.872, 0.879),
    }
    aggregate, criteria, decision = paired_confirmation_decision(per_seed)
    assert decision == "PASS"
    assert all(criteria.values())
    assert aggregate["macro_map50_95"]["acmc2_vs_acmc1_improved_seeds"] == 3
    assert aggregate["bottom3_class_map50_95"]["acmc2_vs_d0ft_mean"] > 0.005


def test_paired_confirmation_rejects_acmc2_that_loses_to_acmc1() -> None:
    per_seed = {
        "42": _record(0.80, 0.86, 0.875, 0.876),
        "123": _record(0.81, 0.865, 0.878, 0.876),
        "2026": _record(0.79, 0.858, 0.872, 0.870),
    }
    _, criteria, decision = paired_confirmation_decision(per_seed)
    assert decision == "FAIL"
    assert criteria["macro_mean_not_lower_than_acmc1"] is False
    assert criteria["macro_improved_over_acmc1_at_least_2_of_3"] is False


def test_paired_confirmation_rejects_tail_collapse() -> None:
    per_seed = {}
    for seed in ALL_SEEDS:
        results = {
            "D0": {metric: 0.80 for metric in METRICS},
            "D0FT": {metric: 0.86 for metric in METRICS},
            "ACMC1": {metric: 0.875 for metric in METRICS},
            "ACMC2": {
                "macro_map50_95": 0.881,
                "bottom3_class_map50_95": 0.879,
                "worst_class_map50_95": 0.850,
            },
        }
        per_seed[str(seed)] = {"results": results}
    _, criteria, decision = paired_confirmation_decision(per_seed)
    assert decision == "FAIL"
    assert criteria["neither_tail_mean_drops_more_than_1_point_vs_acmc1"] is False
