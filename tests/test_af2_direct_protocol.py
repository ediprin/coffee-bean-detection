from pathlib import Path

import yaml

from coffee_detector.experiments.run_faruq_v3_af2_direct import (
    AF2_CONFIG,
    EXPECTED_AF2,
    MIN_RAW_PROPOSAL_DELTA,
    NATIVE_CONFIG,
    _screen_decision,
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_direct_configs_are_schedule_and_model_matched():
    native = _load(NATIVE_CONFIG)
    af2 = _load(AF2_CONFIG)
    assert native["model"] == af2["model"]
    assert native["train"] == af2["train"]
    assert native["train"] == {
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "workers": 2,
        "patience": 15,
        "optimizer": "auto",
        "pretrained": True,
        "cache": False,
        "close_mosaic": 10,
        "max_det": 500,
    }


def test_direct_af2_mapping_is_frozen():
    af2 = _load(AF2_CONFIG)
    assert af2["afab"] == EXPECTED_AF2


def test_route_a_promotes_only_with_localization_safety():
    good = {
        "macro_map50_95": 0.005,
        "bottom3_class_map50_95": -0.005,
        "worst_class_map50_95": -0.005,
        "raw_top500_proposal_accessibility": MIN_RAW_PROPOSAL_DELTA,
    }
    result = _screen_decision(good)
    assert result["route_a_direct_overall_gain"]
    assert result["decision"] == "PROMOTE_TO_3_SEED"

    bad_localization = dict(good)
    bad_localization["raw_top500_proposal_accessibility"] = MIN_RAW_PROPOSAL_DELTA - 1e-6
    result = _screen_decision(bad_localization)
    assert not result["localization_safe"]
    assert result["decision"] == "DO_NOT_PROMOTE"


def test_route_b_accepts_tail_pareto_signal():
    values = {
        "macro_map50_95": -0.002,
        "bottom3_class_map50_95": 0.010,
        "worst_class_map50_95": 0.010,
        "raw_top500_proposal_accessibility": 0.0,
    }
    result = _screen_decision(values)
    assert not result["route_a_direct_overall_gain"]
    assert result["route_b_lower_tail_pareto"]
    assert result["decision"] == "PROMOTE_TO_3_SEED"


def test_no_signal_does_not_promote():
    values = {
        "macro_map50_95": 0.0,
        "bottom3_class_map50_95": 0.0,
        "worst_class_map50_95": 0.0,
        "raw_top500_proposal_accessibility": 0.0,
    }
    result = _screen_decision(values)
    assert not result["route_a_direct_overall_gain"]
    assert not result["route_b_lower_tail_pareto"]
    assert result["decision"] == "DO_NOT_PROMOTE"
