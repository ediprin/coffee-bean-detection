from pathlib import Path

import pytest

import coffee_detector.run_baseline as baseline_module


def test_baseline_stops_before_training_when_audit_is_unsafe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        baseline_module,
        "audit_dataset",
        lambda data_root, output: {"safe_for_training": False},
    )

    with pytest.raises(RuntimeError, match="belum aman"):
        baseline_module.run_baseline(
            tmp_path / "raw",
            tmp_path / "results",
            tmp_path / "config.yaml",
        )

