import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

import coffee_detector.experiments.run_faruq_v3_acmc_adrian_external as runner
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_combined_validation(root: Path) -> None:
    for kind in ("images", "labels"):
        (root / "val" / kind).mkdir(parents=True, exist_ok=True)
    rows = (
        ("adrian_detection__bean_a_jpg.rf.aaa111.jpg", 0),
        ("adrian_detection__bean_b_jpg.rf.bbb222.jpg", 1),
        ("faruq_segmentation__other_jpg.rf.ccc333.jpg", 2),
    )
    for ordinal, (name, class_id) in enumerate(rows):
        Image.new("RGB", (24, 24), (ordinal * 50, 30, 90)).save(
            root / "val/images" / name
        )
        (root / "val/labels" / Path(name).with_suffix(".txt")).write_text(
            f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "val": "val/images",
                "names": {
                    index: name for index, name in enumerate(SNI21_CLASSES)
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "validation_restore.json").write_text(
        json.dumps(
            {"test_files_extracted": 0, "test_images_accessed": False}
        ),
        encoding="utf-8",
    )


def _write_faruq_manifest(path: Path, *, parent: str = "faruqparent") -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "source_parent_id": parent,
                    "source_sha256": "f" * 64,
                    "output_split": "train",
                    "input_image": "faruq_jpg.rf.ddd444.jpg",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_prepare_adrian_validation_is_source_only_and_test_locked(
    tmp_path: Path,
) -> None:
    combined = tmp_path / "combined"
    _write_combined_validation(combined)
    faruq_manifest = tmp_path / "faruq.json"
    _write_faruq_manifest(faruq_manifest)

    result = runner.prepare_adrian_external_validation(
        combined, faruq_manifest, tmp_path / "adrian"
    )

    assert result["status"] == "complete"
    assert result["images"] == 2
    assert result["boxes"] == 2
    assert result["independent_parent_ids"] == 2
    assert all(result["gates"].values())
    assert result["test_images_accessed"] is False
    assert not (tmp_path / "adrian/test").exists()
    assert len(list((tmp_path / "adrian/val/images").iterdir())) == 2
    assert len(result["classes_without_ground_truth"]) == 19


def test_prepare_adrian_validation_rejects_faruq_parent_overlap(
    tmp_path: Path,
) -> None:
    combined = tmp_path / "combined"
    _write_combined_validation(combined)
    faruq_manifest = tmp_path / "faruq.json"
    _write_faruq_manifest(faruq_manifest, parent="beanajpg")

    with pytest.raises(RuntimeError, match="Audit external Adrian gagal"):
        runner.prepare_adrian_external_validation(
            combined, faruq_manifest, tmp_path / "adrian"
        )


def test_external_runner_requires_locked_hashes_and_aggregates_three_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    combined = tmp_path / "combined"
    _write_combined_validation(combined)
    faruq_manifest = tmp_path / "faruq.json"
    _write_faruq_manifest(faruq_manifest)
    checkpoints = {arm: [] for arm in runner.ARMS}
    hashes = {arm: [] for arm in runner.ARMS}
    for arm in runner.ARMS:
        for seed in runner.FROZEN_SEEDS:
            path = tmp_path / f"{arm}_{seed}.pt"
            path.write_bytes(f"{arm}:{seed}".encode())
            checkpoints[arm].append(path)
            hashes[arm].append(_sha256(path))
    locked = tmp_path / "locked.json"
    locked.write_text(
        json.dumps(
            {
                "status": "complete",
                "seeds": list(runner.FROZEN_SEEDS),
                "checkpoint_hashes": hashes,
                "training_executed": False,
                "test_opened": True,
                "further_tuning_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def fake_evaluate(checkpoint, checkpoint_hash, data_root, dataset_hash, output, *, device):
        calls.append((Path(checkpoint).name, device))
        is_acmc = Path(checkpoint).name.startswith("ACMC1")
        value = 0.6 if not is_acmc else 0.62
        return {
            "metrics": {
                "macro_map50_95": value,
                "bottom3_class_map50_95": value - 0.2,
                "worst_class_map50_95": value - 0.3,
                "classes_without_ground_truth": list(SNI21_CLASSES[2:]),
            }
        }

    monkeypatch.setattr(runner, "_evaluate_checkpoint", fake_evaluate)
    result = runner.run_faruq_v3_acmc_adrian_external(
        combined,
        faruq_manifest,
        locked,
        tmp_path / "adrian",
        tmp_path / "reports",
        tuple(checkpoints["D0FT"]),
        tuple(checkpoints["ACMC1"]),
        device="cpu",
    )

    assert len(calls) == 6
    assert result["directional_status"] == "SUPPORTS_EXTERNAL_DIRECTION"
    assert result["aggregate"]["macro_map50_95"]["head_improved_seeds"] == 3
    assert result["training_executed"] is False
    assert result["test_images_accessed"] is False
    assert result["further_tuning_authorized"] is False


def test_external_runner_rejects_checkpoint_not_in_locked_summary(
    tmp_path: Path,
) -> None:
    checkpoints = {arm: [] for arm in runner.ARMS}
    for arm in runner.ARMS:
        for seed in runner.FROZEN_SEEDS:
            path = tmp_path / f"{arm}_{seed}.pt"
            path.write_bytes(f"{arm}:{seed}".encode())
            checkpoints[arm].append(path)
    locked = tmp_path / "locked.json"
    locked.write_text(
        json.dumps(
            {
                "status": "complete",
                "seeds": list(runner.FROZEN_SEEDS),
                "checkpoint_hashes": {
                    arm: ["wrong"] * 3 for arm in runner.ARMS
                },
                "training_executed": False,
                "test_opened": True,
                "further_tuning_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    combined = tmp_path / "combined"
    _write_combined_validation(combined)
    faruq_manifest = tmp_path / "faruq.json"
    _write_faruq_manifest(faruq_manifest)

    with pytest.raises(RuntimeError, match="Hash checkpoint D0FT berbeda"):
        runner.run_faruq_v3_acmc_adrian_external(
            combined,
            faruq_manifest,
            locked,
            tmp_path / "adrian",
            tmp_path / "reports",
            tuple(checkpoints["D0FT"]),
            tuple(checkpoints["ACMC1"]),
            device="cpu",
        )
