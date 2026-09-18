from pathlib import Path

import yaml

from coffee_detector.experiments.run_defectoscafeverde_af2_direct import (
    AF2_CONFIG,
    EXPECTED_AF2,
    NATIVE_CONFIG,
    NC,
    screen_decision,
)


def test_configs_are_matched_except_af2_frontend() -> None:
    native = yaml.safe_load(Path(NATIVE_CONFIG).read_text(encoding="utf-8"))
    candidate = yaml.safe_load(Path(AF2_CONFIG).read_text(encoding="utf-8"))
    assert native["model"] == candidate["model"]
    assert native["train"] == candidate["train"]
    assert candidate["afab"] == EXPECTED_AF2
    assert NC == 12


def test_overall_route_is_frozen() -> None:
    result = screen_decision(
        {
            "macro_map50_95": 0.005,
            "bottom3_class_map50_95": 0.0,
            "worst_class_map50_95": -0.01,
        }
    )
    assert result["overall_route"] is True
    assert result["decision"] == "PROMOTE_TO_PAIRED_3_SEED"


def test_tail_route_is_frozen() -> None:
    result = screen_decision(
        {
            "macro_map50_95": -0.002,
            "bottom3_class_map50_95": 0.01,
            "worst_class_map50_95": 0.01,
        }
    )
    assert result["lower_tail_route"] is True


def test_no_signal_stops() -> None:
    result = screen_decision(
        {
            "macro_map50_95": 0.0,
            "bottom3_class_map50_95": 0.0,
            "worst_class_map50_95": 0.0,
        }
    )
    assert result["decision"] == "STOP_AFTER_SEED42"
