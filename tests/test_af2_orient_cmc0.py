from pathlib import Path

import yaml

from coffee_detector.af2_iso import AF2IsolatedConfig, frozen_arm_config
from coffee_detector.experiments.run_af2_orient_cmc0 import _decision
from coffee_detector.stb import STBConfig


ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / "configs/af2_orient_cmc0/AF2_ORIENT_CMC0_yolo26n.yaml"
PARENT = ROOT / "configs/af2_iso/AF2_ORIENT_yolo26n.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_combined_config_uses_exact_frozen_af2_orient_operator():
    payload = _load(COMBINED)
    config = AF2IsolatedConfig.from_mapping(payload["af2_iso"])
    assert config == frozen_arm_config("AF2_ORIENT")
    assert config.angular_bins == 180
    assert config.orientation_period == 180.0
    assert config.radial_bands == 1
    assert config.stride == 16


def test_combined_config_uses_frozen_cmc0_and_parent_schedule():
    combined = _load(COMBINED)
    parent = _load(PARENT)
    assert STBConfig.from_mapping(combined["stb"]) == STBConfig()
    assert combined["model"] == parent["model"]
    assert combined["train"] == parent["train"]
    assert combined["train"]["pretrained"] is False
    assert combined["train"]["save_period"] == 1


def test_decision_accepts_superiority_route():
    parent = {
        "macro_map50_95": 0.88,
        "bottom3_class_map50_95": 0.80,
        "worst_class_map50_95": 0.78,
    }
    candidate = {
        "macro_map50_95": 0.883,
        "bottom3_class_map50_95": 0.799,
        "worst_class_map50_95": 0.779,
    }
    result = _decision(candidate, parent)
    assert result["decision"] == "PASS"
    assert result["route"] == "SUPERIORITY"


def test_decision_accepts_tail_pareto_route():
    parent = {
        "macro_map50_95": 0.88,
        "bottom3_class_map50_95": 0.80,
        "worst_class_map50_95": 0.78,
    }
    candidate = {
        "macro_map50_95": 0.8795,
        "bottom3_class_map50_95": 0.806,
        "worst_class_map50_95": 0.791,
    }
    result = _decision(candidate, parent)
    assert result["decision"] == "PASS"
    assert result["route"] == "TAIL_PARETO"


def test_decision_rejects_tradeoff_outside_both_routes():
    parent = {
        "macro_map50_95": 0.88,
        "bottom3_class_map50_95": 0.80,
        "worst_class_map50_95": 0.78,
    }
    candidate = {
        "macro_map50_95": 0.877,
        "bottom3_class_map50_95": 0.81,
        "worst_class_map50_95": 0.79,
    }
    result = _decision(candidate, parent)
    assert result["decision"] == "FAIL"
    assert result["route"] == "NONE"
