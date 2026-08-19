from __future__ import annotations

import hashlib
import json
from pathlib import Path

from coffee_detector.experiments.prepare_wav1_factorization_kaggle import (
    ADDON_MANIFEST_FORMAT,
    build_wav1_factorization_kaggle_addon,
    restore_wav1_factorization_kaggle_run,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "macro_map50_95": 0.8841052369918866,
    "bottom3_class_map50_95": 0.8327607439278027,
    "worst_class_map50_95": 0.8203489485589485,
}


def _wav1_result(path: Path) -> Path:
    payload = {
        "format": "coffee_detector.af2_spectral.arm_result.v1",
        "arm": "WAV1",
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "metrics": EXPECTED,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_addon_builder_copies_only_frozen_wav1_reference(tmp_path):
    project = tmp_path / "project"
    source = _wav1_result(project / "prior" / "WAV1_seed42_result.json")
    output = tmp_path / "bundle"
    manifest = build_wav1_factorization_kaggle_addon(
        project, output, wav1_seed42_result=source
    )
    assert manifest["format"] == ADDON_MANIFEST_FORMAT
    assert manifest["test_images_included"] is False
    copied = output / "WAV1_seed42_result.json"
    assert copied.read_bytes() == source.read_bytes()
    assert manifest["artifacts"][copied.name]["sha256"] == _sha(copied)
    assert not any(path.suffix == ".pt" for path in output.iterdir())


def test_restore_requires_exact_factorization_contract(tmp_path):
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    prior = input_root / "prior-arm"
    prior.mkdir(parents=True)
    checkpoint = tmp_path / "D0_seed42_best.pt"
    config = tmp_path / "HP1_yolo26n.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("code: HP1\n", encoding="utf-8")
    contract = {
        "format": "coffee_detector.wav1_factorization.run_contract.v1",
        "arm": "HP1",
        "seed": 42,
        "config_sha256": _sha(config),
        "d0_checkpoint_sha256": _sha(checkpoint),
        "epochs": 50,
        "test_images_accessed": False,
    }
    (prior / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (prior / "weights.txt").write_text("resume payload", encoding="utf-8")

    restored = restore_wav1_factorization_kaggle_run(
        input_root,
        output_root,
        arm="HP1",
        seed=42,
        d0_checkpoint=checkpoint,
        config=config,
    )
    assert restored == output_root / "HP1/HP1_seed42"
    assert (restored / "weights.txt").is_file()
    assert (
        restore_wav1_factorization_kaggle_run(
            input_root,
            output_root,
            arm="WAV_L1",
            seed=42,
            d0_checkpoint=checkpoint,
            config=config,
        )
        is None
    )


def test_factorization_notebooks_are_compilable_and_keep_test_closed():
    paths = (
        ROOT / "notebooks/Faruq_V3_WAV1_Factorization_Kaggle_Addon_Colab.ipynb",
        ROOT / "notebooks/Faruq_V3_WAV1_Factorization_Stage1_Sequential_Kaggle.ipynb",
    )
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        for cell in payload["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), str(path), "exec")
        assert "agent/wav1-mechanism-factorization" in source
        assert "test" not in source.lower() or "locked test" in source.lower()

    kaggle = json.loads(paths[1].read_text(encoding="utf-8"))
    kaggle_source = "\n".join(
        "".join(cell.get("source", [])) for cell in kaggle["cells"]
    )
    assert "for arm in ARMS: run_arm(arm)" in kaggle_source
    assert "HP1" in kaggle_source and "WAV_RAWFUSE" in kaggle_source
    assert "run_factorization_report" in kaggle_source
    assert "STOP HERE" in kaggle_source
    assert "seed 123/2026" in kaggle_source
