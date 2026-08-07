from pathlib import Path

import pytest

from coffee_detector.analysis.faruq_v3_diagnostics import (
    _top_confusions,
    run_faruq_v3_diagnostics,
)


def test_top_confusions_excludes_correct_diagonal() -> None:
    result = _top_confusions(
        {
            "partial_black": {"partial_black": 10, "black": 4},
            "black": {"partial_black": 2, "black": 8},
        }
    )
    assert result == [
        {"expected": "partial_black", "predicted": "black", "count": 4},
        {"expected": "black", "predicted": "partial_black", "count": 2},
    ]


def test_diagnostic_rejects_test_before_dataset_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="dikunci pada validation"):
        run_faruq_v3_diagnostics(
            tmp_path / "missing.pt",
            tmp_path / "missing-data",
            tmp_path / "output.json",
            split="test",
        )
