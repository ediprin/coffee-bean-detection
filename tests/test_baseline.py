from pathlib import Path
import json

import pytest

import coffee_detector.run_baseline as baseline_module


def test_baseline_stops_before_training_when_audit_is_unsafe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        baseline_module,
        "audit_dataset",
        lambda data_root, output, near_threshold: {"safe_for_training": False},
    )

    with pytest.raises(RuntimeError, match="belum aman"):
        baseline_module.run_baseline(
            tmp_path / "raw",
            tmp_path / "results",
            tmp_path / "config.yaml",
        )


def test_verified_audit_must_match_dataset_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "safe_for_training": True,
                "dataset_root": str(tmp_path / "different-data"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="dataset berbeda"):
        baseline_module.load_verified_audit(audit_path, data_root)


def test_partial_checkpoint_is_not_a_completion_marker(tmp_path: Path) -> None:
    run_dir = tmp_path / "D0_seed42"
    best = run_dir / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"partial checkpoint")

    assert not baseline_module.is_training_complete(run_dir)

    (run_dir / "experiment_manifest.json").write_text("{}", encoding="utf-8")
    assert baseline_module.is_training_complete(run_dir)
