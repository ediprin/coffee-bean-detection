import json
from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_stb_paired_confirmation import (
    CONFIRMATION_SEEDS,
    _aggregate,
    _decision,
    _validate_seed42_result,
    run_faruq_v3_stb_paired_confirmation,
)


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _row(macro: float, bottom: float, worst: float) -> dict:
    return dict(zip(METRICS, (macro, bottom, worst)))


def _seed42_payload() -> dict:
    return {
        "protocol": "faruq-v3-stb-capacity-causal-control-v1",
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "decision": "PASS",
        "models": {
            "CMC0": _row(0.87, 0.81, 0.80),
            "STB1": _row(0.89, 0.84, 0.80),
        },
    }


def test_seed42_result_must_be_compatible_pass(tmp_path: Path) -> None:
    path = tmp_path / "seed42.json"
    path.write_text(json.dumps(_seed42_payload()), encoding="utf-8")
    assert _validate_seed42_result(path)["decision"] == "PASS"
    payload = _seed42_payload()
    payload["test_opened"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="bukan PASS"):
        _validate_seed42_result(path)


def test_aggregate_and_frozen_gate_use_paired_deltas() -> None:
    per_seed = {
        "42": {"CMC0": _row(0.87, 0.81, 0.81), "STB1": _row(0.89, 0.84, 0.80)},
        "123": {"CMC0": _row(0.86, 0.80, 0.77), "STB1": _row(0.88, 0.82, 0.78)},
        "2026": {"CMC0": _row(0.88, 0.82, 0.79), "STB1": _row(0.89, 0.83, 0.79)},
    }
    aggregate = _aggregate(per_seed)
    criteria, decision = _decision(aggregate)
    assert decision == "PASS"
    assert all(criteria.values())
    assert aggregate["macro_map50_95"]["spatial_improved_seeds"] == 3
    assert set(aggregate["macro_map50_95"]["deltas"]) == {"42", "123", "2026"}


def test_confirmation_rejects_wrong_seed_set_and_missing_authorization(tmp_path: Path) -> None:
    args = (
        tmp_path / "data",
        tmp_path / "grouped.json",
        tmp_path / "seed42.json",
        (tmp_path / "d0-123.pt", tmp_path / "d0-2026.pt"),
        tmp_path / "output",
    )
    with pytest.raises(ValueError, match="dikunci"):
        run_faruq_v3_stb_paired_confirmation(
            *args, seeds=(42, *CONFIRMATION_SEEDS), authorize_training=True
        )
    with pytest.raises(RuntimeError, match="belum diotorisasi"):
        run_faruq_v3_stb_paired_confirmation(*args)


def test_notebook_is_resumable_shared_drive_and_validation_only() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/Faruq_V3_STB_Paired_Confirmation_Colab.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "agent/stb-capacity-causal-control" in source
    assert "resolve_drive_project_root(required_relative_paths=REQUIRED)" in source
    assert "D0_seed123/weights/best.pt" in source
    assert "D0_seed2026/weights/best.pt" in source
    assert "--seeds', '123', '2026'" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
    assert "test tidak" in source.lower()
