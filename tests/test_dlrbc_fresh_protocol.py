import json
from pathlib import Path

import pytest
import yaml

from coffee_detector.experiments.decide_faruq_v3_dlrbc_fresh import (
    build_fresh_dlrbc_decision,
)
from coffee_detector.experiments.run_faruq_v3_dlrbc_fresh_arm import (
    ARMS,
    CONFIGS,
    EXPECTED_COMMON_DLRBC,
    EXPECTED_TRAIN,
)


def test_all_fresh_configs_are_schedule_and_model_matched():
    payloads = {
        arm: yaml.safe_load(path.read_text(encoding="utf-8")) for arm, path in CONFIGS.items()
    }
    assert tuple(payloads) == ARMS
    assert all(payload["train"] == EXPECTED_TRAIN for payload in payloads.values())
    assert len({payload["model"] for payload in payloads.values()}) == 1
    assert all(payload["weights"] == "official_yolo26n_sha_locked" for payload in payloads.values())


def test_linear_and_quadratic_configs_differ_only_by_mode():
    linear = yaml.safe_load(CONFIGS["LRLIN_FRESH"].read_text(encoding="utf-8"))["dlrbc"]
    quadratic = yaml.safe_load(CONFIGS["DLRBC_FRESH"].read_text(encoding="utf-8"))["dlrbc"]
    assert linear.pop("mode") == "linear"
    assert quadratic.pop("mode") == "quadratic"
    assert linear == quadratic == EXPECTED_COMMON_DLRBC


def _result(arm: str, values: tuple[float, float, float]) -> dict:
    return {
        "arm": arm,
        "seed": 42,
        "metrics": dict(
            zip(
                (
                    "macro_map50_95",
                    "bottom3_class_map50_95",
                    "worst_class_map50_95",
                ),
                values,
            )
        ),
        "fresh_optimizer": True,
        "coffee_parent_checkpoint": None,
        "test_images_accessed": False,
    }


def test_decision_promotes_quadratic_pareto_signal(tmp_path: Path):
    values = {
        "B0_FRESH": (0.80, 0.70, 0.65),
        "LRLIN_FRESH": (0.81, 0.71, 0.66),
        "DLRBC_FRESH": (0.812, 0.72, 0.675),
    }
    paths = []
    for arm in ARMS:
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(_result(arm, values[arm])), encoding="utf-8")
        paths.append(path)
    decision = build_fresh_dlrbc_decision(paths, tmp_path / "decision.json")
    assert decision["decision"] == "PROMOTE_TO_FRESH_3_SEED"
    assert decision["next"] == "RUN_FRESH_SEEDS_123_2026_FROM_OFFICIAL_YOLO26N"
    assert decision["test_images_accessed"] is False


def test_decision_stops_when_quadratic_only_adds_capacity(tmp_path: Path):
    values = {
        "B0_FRESH": (0.80, 0.70, 0.65),
        "LRLIN_FRESH": (0.82, 0.73, 0.69),
        "DLRBC_FRESH": (0.81, 0.72, 0.68),
    }
    paths = []
    for arm in ARMS:
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(_result(arm, values[arm])), encoding="utf-8")
        paths.append(path)
    decision = build_fresh_dlrbc_decision(paths, tmp_path / "decision.json")
    assert decision["decision"] == "STOP_AFTER_SEED42"


def test_decision_rejects_nonfresh_result(tmp_path: Path):
    paths = []
    for arm in ARMS:
        payload = _result(arm, (0.8, 0.7, 0.6))
        if arm == "DLRBC_FRESH":
            payload["coffee_parent_checkpoint"] = "D0FT.pt"
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    with pytest.raises(RuntimeError, match="bukan fresh protocol"):
        build_fresh_dlrbc_decision(paths, tmp_path / "decision.json")
