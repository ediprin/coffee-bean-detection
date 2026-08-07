from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_acmc_finetune_control import (
    METRICS,
    _metrics,
    run_faruq_v3_acmc_finetune_control,
)
from coffee_detector.train import load_experiment


ROOT = Path(__file__).parents[1]


def test_d0ft_control_is_seed_locked_and_uses_matching_schedule(tmp_path: Path) -> None:
    config = load_experiment(ROOT / "configs/fine_tune_control/D0FT_yolo26n_p3.yaml")
    assert config["variant"] == "baseline"
    assert config["train"]["epochs"] == 50
    with pytest.raises(RuntimeError, match="belum diotorisasi"):
        run_faruq_v3_acmc_finetune_control(
            tmp_path / "data", tmp_path / "grouped", tmp_path / "d0", tmp_path / "d0.pt",
            tmp_path / "acmc", tmp_path / "output",
        )
    with pytest.raises(ValueError, match="seed 42"):
        run_faruq_v3_acmc_finetune_control(
            tmp_path / "data", tmp_path / "grouped", tmp_path / "d0", tmp_path / "d0.pt",
            tmp_path / "acmc", tmp_path / "output", seed=123, authorize_training=True,
        )


def test_metrics_accepts_raw_report_and_named_screening_arm() -> None:
    values = dict(zip(METRICS, (0.8, 0.7, 0.6)))
    assert _metrics({"metrics": values}) == values
    assert _metrics({"results": {"ACMC1": values}}, "ACMC1") == values
    with pytest.raises(KeyError, match="result_key"):
        _metrics({"results": {"ACMC1": values}})
