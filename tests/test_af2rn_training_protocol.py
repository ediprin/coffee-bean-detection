from __future__ import annotations

import json

from coffee_detector.experiments.run_faruq_v3_af2rn import (
    run_af2rn_seed42_decision,
)


def _candidate(tmp_path, metrics):
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "format": "coffee_detector.af2rn.arm_result.v1",
                "seed": 42,
                "metrics": {**metrics, "classes_without_ground_truth": []},
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def _baseline(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "candidate": {
                    "AF2": {
                        "macro_map50_95": 0.88,
                        "bottom3_class_map50_95": 0.80,
                        "worst_class_map50_95": 0.79,
                    }
                },
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_af2rn_seed42_decision_passes_frozen_gate(tmp_path) -> None:
    candidate = _candidate(
        tmp_path,
        {
            "macro_map50_95": 0.886,
            "bottom3_class_map50_95": 0.801,
            "worst_class_map50_95": 0.781,
        },
    )
    result = run_af2rn_seed42_decision(
        candidate, _baseline(tmp_path), tmp_path / "decision.json"
    )
    assert result["decision"] == "PASS"
    assert result["next"] == "FREEZE_PAIRED_THREE_SEED_CONFIRMATION"
    assert result["test_opened"] is False


def test_af2rn_seed42_decision_rejects_lower_bottom3(tmp_path) -> None:
    candidate = _candidate(
        tmp_path,
        {
            "macro_map50_95": 0.90,
            "bottom3_class_map50_95": 0.799,
            "worst_class_map50_95": 0.80,
        },
    )
    result = run_af2rn_seed42_decision(
        candidate, _baseline(tmp_path), tmp_path / "decision.json"
    )
    assert result["decision"] == "FAIL"
    assert result["criteria"]["bottom3_not_lower"] is False
    assert result["next"] == "RETAIN_ORIGINAL_AF2_AND_STOP_AF2RN"
