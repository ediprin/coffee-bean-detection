from __future__ import annotations

import json
from pathlib import Path

from coffee_detector.experiments.run_faruq_v3_af2_ffa_from_start_decision import (
    run_faruq_v3_af2_ffa_from_start_decision,
)


def _result(arm: str, seed: int, macro: float, bottom3: float, worst: float) -> dict:
    return {
        "format": "coffee_detector.af2_ffa.from_start_arm_result.v1",
        "arm": arm,
        "seed": seed,
        "initial_d0_checkpoint_sha256": f"d0-{seed}",
        "test_images_accessed": False,
        "metrics": {
            "macro_map50_95": macro,
            "bottom3_class_map50_95": bottom3,
            "worst_class_map50_95": worst,
        },
    }


def test_from_start_decision_requires_all_three_metrics(tmp_path: Path) -> None:
    af2_paths = []
    ffab2_paths = []
    for seed in (42, 123, 2026):
        left = tmp_path / f"af2_{seed}.json"
        right = tmp_path / f"ffab2_{seed}.json"
        left.write_text(json.dumps(_result("AF2FS", seed, 0.88, 0.79, 0.77)))
        right.write_text(json.dumps(_result("AF2FFAB2FS", seed, 0.888, 0.800, 0.775)))
        af2_paths.append(left)
        ffab2_paths.append(right)
    output = tmp_path / "decision.json"
    decision = run_faruq_v3_af2_ffa_from_start_decision(af2_paths, ffab2_paths, output)
    assert decision["decision"] == "PASS"
    assert decision["next"] == "AUTHORIZE_DCT_EFFICIENCY_STAGE"
    assert decision["test_opened"] is False


def test_from_start_decision_rejects_worst_regression(tmp_path: Path) -> None:
    af2_paths = []
    ffab2_paths = []
    for seed in (42, 123, 2026):
        left = tmp_path / f"af2_{seed}.json"
        right = tmp_path / f"ffab2_{seed}.json"
        left.write_text(json.dumps(_result("AF2FS", seed, 0.88, 0.79, 0.77)))
        right.write_text(json.dumps(_result("AF2FFAB2FS", seed, 0.89, 0.81, 0.76)))
        af2_paths.append(left)
        ffab2_paths.append(right)
    decision = run_faruq_v3_af2_ffa_from_start_decision(
        af2_paths, ffab2_paths, tmp_path / "decision.json"
    )
    assert decision["decision"] == "REJECT"
    assert decision["next"] == "STOP_FFAB2_UPGRADE_CLAIM"
