import json
from pathlib import Path

import pytest

from coffee_detector.experiments.prepare_faruq_v3_kaggle import (
    ARCHIVE_SHA256,
    EXPECTED_ANNOTATIONS,
    EXPECTED_IMAGES,
)
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


def _write_kaggle_contract(data_root: Path) -> None:
    contract = {
        "format": "coffee_detector.faruq_v3_kaggle_input_contract.v1",
        "archive_sha256": ARCHIVE_SHA256,
        "data_root": str(data_root.resolve()),
        "splits": {
            split: {
                "images": EXPECTED_IMAGES[split],
                "labels": EXPECTED_IMAGES[split],
                "annotations": EXPECTED_ANNOTATIONS[split],
                "classes": list(range(21)),
            }
            for split in ("train", "val")
        },
        "test_images_accessed": False,
        "decision": "PASS",
    }
    (data_root / "kaggle_input_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )


def test_grouped_summary_gate_accepts_matching_clean_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    summary = tmp_path / "summary.json"
    _write_summary(summary, data_root)
    payload = load_faruq_grouped_summary(summary, data_root)
    assert payload["training_ready"] is True


def test_grouped_summary_gate_accepts_verified_kaggle_relocation(tmp_path: Path) -> None:
    original_root = tmp_path / "content" / "faruq-development-v3-grouped"
    original_root.mkdir(parents=True)
    relocated_root = tmp_path / "kaggle" / "faruq-development-v3-grouped"
    relocated_root.mkdir(parents=True)
    summary = relocated_root / "faruq_grouped_summary.json"
    _write_summary(
        summary,
        original_root,
        images_by_split=EXPECTED_IMAGES,
        annotations_by_split=EXPECTED_ANNOTATIONS,
    )
    _write_kaggle_contract(relocated_root)

    payload = load_faruq_grouped_summary(summary, relocated_root)

    assert payload["output_root"] == str(original_root.resolve())
    assert payload["training_ready"] is True


def test_grouped_summary_gate_rejects_unverified_relocation(tmp_path: Path) -> None:
    original_root = tmp_path / "content" / "faruq-development-v3-grouped"
    original_root.mkdir(parents=True)
    relocated_root = tmp_path / "kaggle" / "faruq-development-v3-grouped"
    relocated_root.mkdir(parents=True)
    summary = relocated_root / "faruq_grouped_summary.json"
    _write_summary(
        summary,
        original_root,
        images_by_split=EXPECTED_IMAGES,
        annotations_by_split=EXPECTED_ANNOTATIONS,
    )

    with pytest.raises(RuntimeError, match="tidak ada kontrak relokasi Kaggle terverifikasi"):
        load_faruq_grouped_summary(summary, relocated_root)


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
