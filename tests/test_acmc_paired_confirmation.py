import json
from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_acmc_paired_confirmation import (
    CONFIRMATION_SEEDS,
    _validate_seed42_control,
    run_faruq_v3_acmc_paired_confirmation,
)


def _control_payload() -> dict:
    metrics = {
        "macro_map50_95": 0.8,
        "bottom3_class_map50_95": 0.7,
        "worst_class_map50_95": 0.6,
    }
    return {
        "protocol": "faruq-v3-acmc-optimization-control-v1",
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "decision": "PASS",
        "results": {"D0": metrics, "D0FT": metrics, "ACMC1": metrics},
        "control_deltas_d0ft_vs_d0": metrics,
        "head_deltas_acmc1_vs_d0ft": metrics,
    }


def test_seed42_control_requires_valid_paired_pass(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(json.dumps(_control_payload()), encoding="utf-8")
    assert _validate_seed42_control(path)["seed"] == 42
    payload = _control_payload()
    payload["decision"] = "FAIL"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="bukan PASS"):
        _validate_seed42_control(path)


def test_paired_confirmation_is_locked_to_two_new_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dikunci"):
        run_faruq_v3_acmc_paired_confirmation(
            tmp_path / "data",
            tmp_path / "grouped",
            tmp_path / "control.json",
            tmp_path / "output",
            seeds=(42, *CONFIRMATION_SEEDS),
            authorize_training=True,
        )
    with pytest.raises(RuntimeError, match="belum diotorisasi"):
        run_faruq_v3_acmc_paired_confirmation(
            tmp_path / "data",
            tmp_path / "grouped",
            tmp_path / "control.json",
            tmp_path / "output",
        )
