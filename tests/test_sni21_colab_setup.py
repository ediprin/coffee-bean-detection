import io
import json
import tarfile
from pathlib import Path

import pytest

import coffee_detector.run_sni21_colab_setup as runner_module


def test_extract_archive_writes_marker_and_reuses_output(tmp_path: Path) -> None:
    archive = tmp_path / "data.tar"
    content = b"dataset"
    info = tarfile.TarInfo("nested/file.txt")
    info.size = len(content)
    with tarfile.open(archive, "w") as bundle:
        bundle.addfile(info, io.BytesIO(content))

    target = runner_module.extract_archive(
        archive, tmp_path / "extract", progress_every=1
    )
    assert (target / "nested" / "file.txt").read_bytes() == content
    assert (target / ".extract_complete").is_file()

    reused = runner_module.extract_archive(
        archive, target, progress_every=1
    )
    assert reused == target


def test_prepare_or_reuse_a0_does_not_repeat_completed_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "a0"
    output.mkdir()
    (output / "audit.json").write_text(
        json.dumps({"training_ready": True, "test_locked": True}),
        encoding="utf-8",
    )
    post = {
        "safe_for_training": True,
        "cross_split_duplicate_components": 0,
        "images_by_split": {"train": 8, "val": 2, "test": 2},
    }
    (output / "post_materialization_audit.json").write_text(
        json.dumps(post), encoding="utf-8"
    )

    monkeypatch.setattr(
        runner_module,
        "prepare_sni_fullscene",
        lambda *_a, **_k: pytest.fail("materialisasi tidak boleh diulang"),
    )
    monkeypatch.setattr(
        runner_module,
        "audit_dataset",
        lambda *_a, **_k: pytest.fail("post-audit tidak boleh diulang"),
    )

    root, reused = runner_module.prepare_or_reuse_a0(
        tmp_path / "adrian",
        tmp_path / "faruq",
        tmp_path / "manifest.csv",
        output,
    )
    assert root == output.resolve()
    assert reused == post


def test_colab_smoke_orchestrator_never_trains_or_opens_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop = tmp_path / "crop"
    crop.mkdir()
    for name in ("manifest.csv", "audit.json", "complete.json"):
        (crop / name).write_text("{}", encoding="utf-8")
    (crop / "shards").mkdir()
    adrian_archive = tmp_path / "adrian.tar"
    faruq_archive = tmp_path / "faruq.tar"
    adrian_archive.write_bytes(b"placeholder")
    faruq_archive.write_bytes(b"placeholder")
    calls = []

    def fake_extract(_archive, target, **_kwargs):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def fake_prepare(_adrian, _faruq, _manifest, output, **_kwargs):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        return output, {
            "safe_for_training": True,
            "cross_split_duplicate_components": 0,
            "images_by_split": {"train": 8, "val": 2, "test": 2},
        }

    def fake_setup(_a0, _crop, output, **kwargs):
        calls.append(kwargs)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        return {
            "training_ready": True,
            "training_executed": False,
            "test_accessed": False,
            "arms": {
                arm: {
                    "root": str(output / arm),
                    "raw_contact_sheet": str(output / f"{arm}_raw.jpg"),
                    "overlay_contact_sheet": str(
                        output / f"{arm}_overlay.jpg"
                    ),
                }
                for arm in ("A1", "A2")
            },
        }

    monkeypatch.setattr(runner_module, "extract_archive", fake_extract)
    monkeypatch.setattr(
        runner_module, "find_coco_root", lambda path: Path(path)
    )
    monkeypatch.setattr(runner_module, "prepare_or_reuse_a0", fake_prepare)
    monkeypatch.setattr(runner_module, "run_sni21_vadcp_setup", fake_setup)

    result = runner_module.run_sni21_colab_setup(
        adrian_archive,
        faruq_archive,
        crop,
        tmp_path / "work",
        profile="smoke",
    )

    assert calls[0]["synthetic_images"] == 2
    assert calls[0]["objects_min"] == 220
    assert calls[0]["objects_max"] == 300
    assert result["training_ready"] is True
    assert result["training_executed"] is False
    assert result["test_accessed"] is False


def test_pilot_profile_is_fixed_to_200_synthetic_scenes() -> None:
    assert runner_module.PROFILES["pilot"] == {
        "synthetic_images": 200,
        "visual_samples": 12,
    }
