import json
from pathlib import Path

from coffee_detector.archive_sni21_pilot import (
    pack_real_a0,
    pack_sni21_pilot_bundle,
    restore_real_a0_development,
    restore_real_a0_validation,
    restore_sni21_pilot_bundle,
)


def _write_dataset(root: Path, *, synthetic: bool) -> None:
    root.mkdir(parents=True)
    (root / "data.yaml").write_text("names: [bean]\n", encoding="utf-8")
    if synthetic:
        (root / "train/images").mkdir(parents=True)
        (root / "train/labels").mkdir(parents=True)
        (root / "metadata").mkdir()
        (root / "train/images/one.jpg").write_bytes(b"image")
        (root / "train/labels/one.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        (root / "metadata/generation_manifest.json").write_text(
            "{}", encoding="utf-8"
        )
        (root / "metadata/vadcp_audit.json").write_text(
            json.dumps(
                {"dataset_root": str(root.resolve()), "safe_for_training": True}
            ),
            encoding="utf-8",
        )
        return
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        (root / split / "images/one.jpg").write_bytes(split.encode())
        (root / split / "labels/one.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    (root / "audit.json").write_text("{}", encoding="utf-8")
    (root / "post_materialization_audit.json").write_text(
        json.dumps(
            {"dataset_root": str(root.resolve()), "safe_for_training": True}
        ),
        encoding="utf-8",
    )


def test_pack_and_restore_sni21_pilot_bundle(tmp_path: Path) -> None:
    source_work = tmp_path / "source"
    a0 = source_work / "sni21-fullscene-v1"
    setup = source_work / "sni21-vadcp-pilot"
    _write_dataset(a0, synthetic=False)
    _write_dataset(setup / "A1", synthetic=True)
    _write_dataset(setup / "A2", synthetic=True)
    (setup / "setup_summary.json").write_text(
        json.dumps(
            {
                "synthetic_images_per_arm": 200,
                "training_ready": True,
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    bundle = pack_sni21_pilot_bundle(a0, setup, tmp_path / "bundle")
    assert bundle["training_ready"] is True
    assert bundle["test_opened"] is False

    restored = restore_sni21_pilot_bundle(
        tmp_path / "bundle", tmp_path / "restored"
    )
    assert restored["training_ready"] is True
    assert restored["test_accessed"] is False
    assert Path(restored["a0_root"], "val/images/one.jpg").is_file()
    assert Path(restored["arms"]["A1"]["root"], "train/images/one.jpg").is_file()
    assert Path(restored["arms"]["A2"]["root"], "train/images/one.jpg").is_file()


def test_restore_real_a0_validation_does_not_extract_test(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_dataset(source, synthetic=False)
    archive = pack_real_a0(source, tmp_path / "A0_real.tar")

    restored = restore_real_a0_validation(
        archive, tmp_path / "validation-only"
    )

    assert (restored / "val/images/one.jpg").is_file()
    assert not (restored / "test").exists()
    assert not any((restored / "train/images").iterdir())
    payload = json.loads(
        (restored / "validation_restore.json").read_text(encoding="utf-8")
    )
    assert payload["test_files_extracted"] == 0
    assert payload["test_images_accessed"] is False


def test_restore_real_a0_development_extracts_train_val_not_test(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_dataset(source, synthetic=False)
    archive = pack_real_a0(source, tmp_path / "A0_real.tar")

    restored = restore_real_a0_development(
        archive, tmp_path / "development-only"
    )

    assert (restored / "train/images/one.jpg").is_file()
    assert (restored / "val/images/one.jpg").is_file()
    assert not (restored / "test").exists()
    payload = json.loads(
        (restored / "development_restore.json").read_text(encoding="utf-8")
    )
    assert payload["test_files_extracted"] == 0
    assert payload["test_images_accessed"] is False
