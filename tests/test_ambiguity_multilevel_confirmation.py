from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_acmc_confirmation import (
    DEFAULT_SEEDS,
    run_faruq_v3_acmc_confirmation,
)


def test_acmc_confirmation_is_authorized_only_for_frozen_paired_seeds(tmp_path: Path) -> None:
    assert DEFAULT_SEEDS == (42, 123, 2026)
    with pytest.raises(RuntimeError, match="belum diotorisasi"):
        run_faruq_v3_acmc_confirmation(
            tmp_path / "data", tmp_path / "grouped.json", tmp_path / "baseline", tmp_path / "output"
        )
    with pytest.raises(ValueError, match="dikunci"):
        run_faruq_v3_acmc_confirmation(
            tmp_path / "data",
            tmp_path / "grouped.json",
            tmp_path / "baseline",
            tmp_path / "output",
            seeds=(42, 123),
            authorize_training=True,
        )
