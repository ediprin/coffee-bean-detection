from __future__ import annotations

import json
from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_wav_l1_paired_decision import run_decision
from coffee_detector.experiments.run_faruq_v3_wav_l1_confirmation_seed import ALLOWED_SEEDS


ROOT = Path(__file__).resolve().parents[1]


def _candidate(seed: int, metrics: dict) -> dict:
    return {
        "format": "coffee_detector.wav_l1_confirmation.seed_result.v1",
        "arm": "WAV_L1",
        "seed": seed,
        "metrics": metrics,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }


def test_confirmation_scope_is_frozen_to_123_and_2026():
    assert ALLOWED_SEEDS == (123, 2026)
    protocol = (ROOT / "docs/FARUQ_V3_WAV_L1_PAIRED_CONFIRMATION_PROTOCOL_2026-08-19.md").read_text(
        encoding="utf-8"
    )
    assert "frozen before seed-123/2026 WAV_L1 training" in protocol
    assert "Locked test" in protocol
    assert "Mean paired Macro gain >= **+0.5 percentage point**" in protocol


def test_seed42_machine_reference_matches_stage1_selection():
    payload = json.loads(
        (ROOT / "docs/evidence/FARUQ_V3_WAV_L1_SEED42_RESULT_2026-08-19.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["arm"] == "WAV_L1" and payload["seed"] == 42
    assert payload["metrics"]["macro_map50_95"] == pytest.approx(0.885720537714217)
    assert payload["metrics"]["bottom3_class_map50_95"] == pytest.approx(0.8399334705085897)
    assert payload["metrics"]["worst_class_map50_95"] == pytest.approx(0.8209474694929713)
    assert payload["test_images_accessed"] is False


def test_paired_decision_accepts_upstream_acmc_d0ft_reference(tmp_path: Path):
    seed42 = ROOT / "docs/evidence/FARUQ_V3_WAV_L1_SEED42_RESULT_2026-08-19.json"
    d0 = {
        42: {"macro_map50_95": 0.8668870418312263, "bottom3_class_map50_95": 0.7498085237045921, "worst_class_map50_95": 0.7202242739437643},
        123: {"macro_map50_95": 0.8635, "bottom3_class_map50_95": 0.7475, "worst_class_map50_95": 0.6783},
        2026: {"macro_map50_95": 0.8681, "bottom3_class_map50_95": 0.7999, "worst_class_map50_95": 0.7931},
    }
    c123 = {"macro_map50_95": 0.8740, "bottom3_class_map50_95": 0.7800, "worst_class_map50_95": 0.7200}
    c2026 = {"macro_map50_95": 0.8690, "bottom3_class_map50_95": 0.8050, "worst_class_map50_95": 0.7900}
    p123, p2026 = tmp_path / "123.json", tmp_path / "2026.json"
    p123.write_text(json.dumps(_candidate(123, c123)), encoding="utf-8")
    p2026.write_text(json.dumps(_candidate(2026, c2026)), encoding="utf-8")
    reference = {
        "protocol": "faruq-v3-acmc-paired-optimization-confirmation-v1",
        "seeds": [42, 123, 2026],
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "decision": "PASS",
        "per_seed": {
            str(seed): {"results": {"D0FT": metrics}} for seed, metrics in d0.items()
        },
    }
    ref = tmp_path / "d0ft.json"
    ref.write_text(json.dumps(reference), encoding="utf-8")
    output = tmp_path / "decision.json"
    result = run_decision(seed42, p123, p2026, ref, output)
    assert output.is_file()
    assert result["evaluation_split"] == "val"
    assert result["test_opened"] is False
    assert result["test_images_accessed"] is False
    assert set(result["criteria"]) == {
        "macro_gain_at_least_0_5_point",
        "macro_improved_at_least_2_of_3",
        "bottom3_mean_not_lower",
        "bottom3_improved_at_least_2_of_3",
        "worst_mean_drop_no_more_than_1_point",
    }
