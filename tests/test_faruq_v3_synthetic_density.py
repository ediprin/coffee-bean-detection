import json
from pathlib import Path

from coffee_detector.experiments import run_faruq_v3_synthetic_density_screening as screen
from coffee_detector.experiments.run_faruq_v3_synthetic_density_setup import (
    _audit_library,
)
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def test_validation_library_audit_accepts_group_safe_parents(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    library_root = tmp_path / "library"
    data_root.mkdir()
    library_root.mkdir()
    parents = [f"parent-{index}" for index in range(len(SNI21_CLASSES))]
    manifest = [
        {"source_parent_id": "train-parent", "output_split": "train"},
        *[
            {"source_parent_id": parent, "output_split": "val"}
            for parent in parents
        ],
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    library = {
        "source": {"root": str(data_root), "source_split": "val"},
        "classes": {str(index): name for index, name in enumerate(SNI21_CLASSES)},
        "assets": [
            {"asset_id": str(index), "source_parent_id": parents[index]}
            for index in range(len(SNI21_CLASSES))
        ],
    }
    (library_root / "object_library.json").write_text(
        json.dumps(library), encoding="utf-8"
    )
    result = _audit_library(data_root, manifest_path, library_root)
    assert result["safe_for_development_diagnostic"] is True
    assert result["train_parent_overlap"] == []


def test_validation_library_audit_rejects_train_parent_overlap(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    library_root = tmp_path / "library"
    data_root.mkdir()
    library_root.mkdir()
    parents = [f"parent-{index}" for index in range(len(SNI21_CLASSES))]
    manifest = [
        {"source_parent_id": parents[0], "output_split": "train"},
        *[
            {"source_parent_id": parent, "output_split": "val"}
            for parent in parents[1:]
        ],
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    library = {
        "source": {"root": str(data_root), "source_split": "val"},
        "classes": {str(index): name for index, name in enumerate(SNI21_CLASSES)},
        "assets": [
            {"asset_id": str(index), "source_parent_id": parents[index]}
            for index in range(len(SNI21_CLASSES))
        ],
    }
    (library_root / "object_library.json").write_text(
        json.dumps(library), encoding="utf-8"
    )
    result = _audit_library(data_root, manifest_path, library_root)
    assert result["safe_for_development_diagnostic"] is False
    assert result["train_parent_overlap"] == [parents[0]]


def test_seed42_screening_uses_frozen_checkpoints_without_test(
    tmp_path: Path, monkeypatch
) -> None:
    arms = {}
    for index, arm in enumerate(screen.ARM_ORDER):
        root = tmp_path / arm
        root.mkdir()
        arms[arm] = {"root": str(root), "density": [index + 1, index + 2]}
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps(
            {
                "format": "coffee_detector.faruq_v3_synthetic_density_setup.v1",
                "ready_for_frozen_screening": True,
                "training_executed": False,
                "test_images_accessed": False,
                "source_split": "faruq_v3_validation",
                "arms": arms,
            }
        ),
        encoding="utf-8",
    )
    d0ft, acmc = tmp_path / "d0ft.pt", tmp_path / "acmc.pt"
    d0ft.write_bytes(b"d0ft")
    acmc.write_bytes(b"acmc")

    def fake_evaluate(checkpoint, checkpoint_hash, arm, arm_root, output, **kwargs):
        gain = 0.02 if checkpoint.name == "acmc.pt" else 0.0
        return {
            "metrics": {
                "macro_map50_95": 0.7 + gain,
                "bottom3_map50_95": 0.6 + gain,
                "worst_map50_95": 0.5 + gain,
            }
        }

    monkeypatch.setattr(screen, "_evaluate", fake_evaluate)
    result = screen.run_faruq_v3_synthetic_density_screening(
        setup, d0ft, acmc, tmp_path / "output", device=None
    )
    assert result["summary"]["macro_improved_conditions"] == 4
    assert result["training_executed"] is False
    assert result["test_images_accessed"] is False
    assert result["locked_test_conclusion_changed"] is False
