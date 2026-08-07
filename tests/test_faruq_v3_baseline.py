import json
from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)


def _write_summary(path: Path, data_root: Path, **overrides) -> None:
    payload = {
        "format": "coffee_detector.faruq_grouped_development.v1",
        "output_root": str(data_root.resolve()),
        "training_ready": True,
        "cross_split_parent_identities": 0,
        "cross_split_exact_hashes": 0,
        "test_images_accessed": False,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_grouped_summary_gate_accepts_matching_clean_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    summary = tmp_path / "summary.json"
    _write_summary(summary, data_root)
    payload = load_faruq_grouped_summary(summary, data_root)
    assert payload["training_ready"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"training_ready": False},
        {"cross_split_parent_identities": 1},
        {"cross_split_exact_hashes": 1},
        {"test_images_accessed": True},
    ],
)
def test_grouped_summary_gate_rejects_unsafe_dataset(
    tmp_path: Path, overrides: dict
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    summary = tmp_path / "summary.json"
    _write_summary(summary, data_root, **overrides)
    with pytest.raises(RuntimeError):
        load_faruq_grouped_summary(summary, data_root)
