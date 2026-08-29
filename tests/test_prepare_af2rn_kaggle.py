from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import yaml

from coffee_detector.experiments import prepare_af2rn_kaggle as kaggle


def test_prepare_af2rn_kaggle_validates_only_train(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source" / kaggle.DATASET_DIRNAME
    images = source / "train/images"
    labels = source / "train/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(2):
        (images / f"{index}.jpg").write_bytes(b"image")
        classes = range(index, 21, 2)
        (labels / f"{index}.txt").write_text(
            "\n".join(f"{class_id} 0.5 0.5 0.2 0.2" for class_id in classes)
            + "\n",
            encoding="utf-8",
        )
    (source / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": "/content/faruq-development-v3-grouped",
                "train": "train/images",
                "val": "val/images",
                "names": {index: f"class_{index}" for index in range(21)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (source / "faruq_grouped_summary.json").write_text(
        json.dumps(
            {
                "format": "coffee_detector.faruq_grouped_development.v1",
                "images_by_split": {"train": 2, "val": 2},
                "annotations_by_split": {"train": 21, "val": 21},
                "training_ready": True,
                "test_locked": True,
                "gates": {"safe": True},
            }
        ),
        encoding="utf-8",
    )

    input_root = tmp_path / "input"
    input_root.mkdir()
    archive = input_root / kaggle.ARCHIVE_NAME
    with tarfile.open(archive, "w") as stream:
        stream.add(source, arcname=kaggle.DATASET_DIRNAME)
    d0 = input_root / kaggle.D0_NAME
    d0.write_bytes(b"checkpoint")
    af2_result = input_root / kaggle.AF2_RESULT_NAME
    af2_result.write_text(
        json.dumps(
            {
                "candidate": {
                    "AF2": {
                        "macro_map50_95": 0.88,
                        "bottom3_class_map50_95": 0.80,
                        "worst_class_map50_95": 0.79,
                    }
                },
                "test_images_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    d0_sha = hashlib.sha256(d0.read_bytes()).hexdigest()
    af2_result_sha = hashlib.sha256(af2_result.read_bytes()).hexdigest()
    manifest = {
        "format": kaggle.MANIFEST_FORMAT,
        "artifacts": {
            kaggle.ARCHIVE_NAME: {
                "bytes": archive.stat().st_size,
                "sha256": archive_sha,
            },
            kaggle.D0_NAME: {"bytes": d0.stat().st_size, "sha256": d0_sha},
            kaggle.AF2_RESULT_NAME: {
                "bytes": af2_result.stat().st_size,
                "sha256": af2_result_sha,
            },
        },
        "checkpoint_validation": {
            kaggle.D0_NAME: {
                "loadable_by_ultralytics": True,
                "nc": 21,
                "bytes": d0.stat().st_size,
                "sha256": d0_sha,
            }
        },
        "test_images_included": False,
    }
    (input_root / kaggle.MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    monkeypatch.setattr(kaggle, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(kaggle, "ARCHIVE_SHA256", archive_sha)
    monkeypatch.setattr(kaggle, "EXPECTED_IMAGES", {"train": 2, "val": 2})
    monkeypatch.setattr(kaggle, "EXPECTED_ANNOTATIONS", {"train": 21, "val": 21})

    work_root = tmp_path / "work"
    work_root.mkdir()
    data_root, resolved_d0, contract = kaggle.prepare_af2rn_kaggle_input(
        input_root, work_root
    )

    assert resolved_d0 == d0.resolve()
    assert contract["decision"] == "PASS"
    assert Path(contract["af2_result"]) == af2_result.resolve()
    assert contract["validation_files_read"] is False
    assert contract["splits"]["train"]["annotations"] == 21
    assert contract["splits"]["val"]["source"] == (
        "immutable_archive_sha_and_grouped_summary"
    )
    assert not (data_root / "val").exists()
    assert not (data_root / "test").exists()
