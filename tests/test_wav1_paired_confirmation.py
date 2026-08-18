from __future__ import annotations

import json
from pathlib import Path

from coffee_detector.experiments.run_faruq_v3_wav1_paired_confirmation import (
    ALL_SEEDS,
    CONFIRMATION_SEEDS,
    EXPECTED_SEED42,
    METRICS,
    REFERENCE_MEANS,
    _aggregate,
    _primary_decision,
    _validate_seed42,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/FARUQ_V3_WAV1_SEED42_RESULT_2026-08-19.json"


def _row(macro: float, bottom3: float, worst: float) -> dict[str, float]:
    return {
        "macro_map50_95": macro,
        "bottom3_class_map50_95": bottom3,
        "worst_class_map50_95": worst,
    }


def test_protocol_freezes_only_two_new_training_seeds_and_test_lock():
    protocol = (ROOT / "docs/FARUQ_V3_WAV1_PAIRED_CONFIRMATION_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert "frozen before seed-123/2026 WAV1 training" in protocol
    assert "Only WAV1 seeds 123 and 2026 are newly trained" in protocol
    assert "a Kaggle Saved Version of seed42 is not required" in protocol
    assert "Faruq locked test is not restored, read, or reopened" in protocol
    assert CONFIRMATION_SEEDS == (123, 2026)
    assert ALL_SEEDS == (42, 123, 2026)


def test_seed42_contract_matches_completed_wav1_result_and_repository_evidence():
    assert EXPECTED_SEED42 == {
        "macro_map50_95": 0.8841052369918866,
        "bottom3_class_map50_95": 0.8327607439278027,
        "worst_class_map50_95": 0.8203489485589485,
    }
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert payload["checkpoint_sha256"] == "ff8d06f2f9b98ae005c1b60d67613e3397eb541dc6420a3d2b069f8cd56ac426"
    assert payload["initial_d0_checkpoint_sha256"] == "0c458841b84bedce4e0ddada6a5773f6a5ac8a91dad084a4a5f24e89f04e6367"
    values, validated = _validate_seed42(EVIDENCE)
    assert values == EXPECTED_SEED42
    assert validated["evaluation_split"] == "val"
    assert validated["test_images_accessed"] is False


def test_primary_gate_matches_af2_igem_confirmation_rules():
    per_seed = {
        "42": {"D0FT": _row(0.86, 0.75, 0.72), "WAV1": _row(0.88, 0.83, 0.82)},
        "123": {"D0FT": _row(0.86, 0.75, 0.68), "WAV1": _row(0.87, 0.78, 0.72)},
        "2026": {"D0FT": _row(0.87, 0.80, 0.79), "WAV1": _row(0.875, 0.80, 0.785)},
    }
    aggregate = _aggregate(per_seed)
    criteria, decision = _primary_decision(aggregate)
    assert decision == "PASS"
    assert criteria == {
        "macro_gain_at_least_0_5_point": True,
        "macro_improved_at_least_2_of_3": True,
        "bottom3_mean_not_lower": True,
        "bottom3_improved_at_least_2_of_3": True,
        "worst_mean_drop_no_more_than_1_point": True,
    }


def test_macro_gate_cannot_be_relaxed_post_hoc():
    per_seed = {
        "42": {"D0FT": _row(0.86, 0.75, 0.72), "WAV1": _row(0.864, 0.80, 0.80)},
        "123": {"D0FT": _row(0.86, 0.75, 0.68), "WAV1": _row(0.864, 0.80, 0.80)},
        "2026": {"D0FT": _row(0.86, 0.75, 0.79), "WAV1": _row(0.864, 0.80, 0.80)},
    }
    aggregate = _aggregate(per_seed)
    criteria, decision = _primary_decision(aggregate)
    assert aggregate["macro_map50_95"]["paired_delta_mean"] < 0.005
    assert criteria["macro_gain_at_least_0_5_point"] is False
    assert decision == "FAIL"


def test_reference_means_are_frozen_for_descriptive_comparison_only():
    assert set(REFERENCE_MEANS) == {"AF2", "IGEM1", "STB1", "ACMC1"}
    assert REFERENCE_MEANS["AF2"]["macro_map50_95"] == 0.8793765273831853
    assert REFERENCE_MEANS["IGEM1"]["worst_class_map50_95"] == 0.7773973469475308
    assert REFERENCE_MEANS["STB1"]["bottom3_class_map50_95"] == 0.8049539441492847


def test_kaggle_notebook_compiles_and_uses_repository_seed42_without_retraining_it():
    path = ROOT / "notebooks/Faruq_V3_WAV1_Paired_Multiseed_Kaggle.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(path), "exec")
    assert "FARUQ_V3_WAV1_SEED42_RESULT_2026-08-19.json" in source
    assert "Attach Saved Version output WAV1 seed42" not in source
    assert "D0_seed123_best.pt" in source
    assert "D0_seed2026_best.pt" in source
    assert "--authorize-training" in source
    assert "test_images_accessed" in source
    assert "for seed in (123,2026)" in source
    assert "D0_seed42_best.pt" not in source


def test_metric_names_remain_ap50_95_statistics():
    assert METRICS == (
        "macro_map50_95",
        "bottom3_class_map50_95",
        "worst_class_map50_95",
    )
