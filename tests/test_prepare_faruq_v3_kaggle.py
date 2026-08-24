import hashlib
import json
import tarfile
from pathlib import Path

import yaml

from coffee_detector.experiments import prepare_faruq_v3_kaggle as kaggle_input


def _build_archive(tmp_path: Path) -> Path:
    source = tmp_path / "source" / kaggle_input.DATASET_DIRNAME
    for split in ("train", "val"):
        images = source / split / "images"
        labels = source / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for index in range(2):
            (images / f"{index}.jpg").write_bytes(b"image")
            class_ids = range(index, 21, 2)
            rows = [f"{class_id} 0.5 0.5 0.2 0.2" for class_id in class_ids]
            (labels / f"{index}.txt").write_text("\n".join(rows) + "\n")
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
                "gates": {"all_classes_present": True},
            }
        ),
        encoding="utf-8",
    )
    input_root = tmp_path / "input"
    input_root.mkdir()
    archive = input_root / kaggle_input.ARCHIVE_NAME
    with tarfile.open(archive, "w") as handle:
        handle.add(source, arcname=kaggle_input.DATASET_DIRNAME)
    return archive


def test_prepare_kaggle_input_extracts_validates_and_rewrites_yaml(
    tmp_path, monkeypatch
):
    archive = _build_archive(tmp_path)
    monkeypatch.setattr(kaggle_input, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        kaggle_input,
        "ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(kaggle_input, "EXPECTED_IMAGES", {"train": 2, "val": 2})
    monkeypatch.setattr(
        kaggle_input, "EXPECTED_ANNOTATIONS", {"train": 21, "val": 21}
    )

    work_root = tmp_path / "work"
    work_root.mkdir()
    data_root, contract = kaggle_input.prepare_faruq_v3_kaggle_input(
        archive.parent, work_root
    )

    resolved_yaml = yaml.safe_load(
        (data_root / "data.yaml").read_text(encoding="utf-8")
    )
    assert resolved_yaml["path"] == str(data_root)
    assert contract["decision"] == "PASS"
    assert contract["splits"]["train"]["annotations"] == 21
    assert contract["splits"]["val"]["classes"] == list(range(21))
    assert not (data_root / "test").exists()


def test_kaggle_notebook_installs_ultralytics_before_importing_it():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks/Faruq_V3_AF2R_Kaggle.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][1]["source"])
    install = "'ultralytics==8.4.96'"
    first_import = "import ultralytics"

    assert install in setup
    assert setup.index(install) < setup.index(first_import)
    assert "TORCH_VERSION_BEFORE" in setup
    assert "TORCH_VERSION_AFTER" in setup
