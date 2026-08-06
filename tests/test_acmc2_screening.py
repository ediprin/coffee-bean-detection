from coffee_detector.experiments.run_faruq_v3_acmc2_screening import screening_decision


D0FT = {
    "macro_map50_95": 0.8662,
    "bottom3_class_map50_95": 0.7658,
    "worst_class_map50_95": 0.7305,
}
ACMC1 = {
    "macro_map50_95": 0.8762,
    "bottom3_class_map50_95": 0.7913,
    "worst_class_map50_95": 0.7630,
}


def test_acmc2_screening_passes_when_control_is_retained_and_tail_improves() -> None:
    acmc2 = {
        "macro_map50_95": 0.8780,
        "bottom3_class_map50_95": 0.7950,
        "worst_class_map50_95": 0.7620,
    }
    vs_d0ft, vs_acmc1, criteria, decision = screening_decision(D0FT, ACMC1, acmc2)
    assert decision == "PASS"
    assert all(criteria.values())
    assert vs_d0ft["macro_map50_95"] >= 0.005
    assert vs_acmc1["bottom3_class_map50_95"] > 0.0


def test_acmc2_screening_rejects_macro_regression_vs_acmc1() -> None:
    acmc2 = {
        "macro_map50_95": 0.8750,
        "bottom3_class_map50_95": 0.8000,
        "worst_class_map50_95": 0.7700,
    }
    _, _, criteria, decision = screening_decision(D0FT, ACMC1, acmc2)
    assert decision == "FAIL"
    assert criteria["macro_not_lower_than_acmc1"] is False


def test_acmc2_screening_requires_tail_improvement_vs_acmc1() -> None:
    acmc2 = {
        "macro_map50_95": 0.8800,
        "bottom3_class_map50_95": 0.7913,
        "worst_class_map50_95": 0.7630,
    }
    _, _, criteria, decision = screening_decision(D0FT, ACMC1, acmc2)
    assert decision == "FAIL"
    assert criteria["at_least_one_tail_metric_improves_over_acmc1"] is False
