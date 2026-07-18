import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from coffee_detector.run_vadcp_ablation import build_combined_training_view


def _write_arm(root: Path, *, synthetic: bool, names: dict[int, str] | None = None) -> None:
    names = names or {0: "bean", 1: "defect"}
    for split in ("train", "val", "test"):
        image_root = root / split / "images"
        label_root = root / split / "labels"
        image_root.mkdir(parents=True, exist_ok=True)
        label_root.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (40, 50, 60)).save(
            image_root / f"{split}.png"
        )
        (label_root / f"{split}.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if synthetic:
        metadata = root / "metadata"
        metadata.mkdir(parents=True)
        (metadata / "generation_manifest.json").write_text(
            json.dumps({"include_real_train": False}), encoding="utf-8"
        )


def test_combined_training_view_uses_path_list_without_copy(tmp_path: Path) -> None:
    real = tmp_path / "real"
    synthetic = tmp_path / "synthetic"
    output = tmp_path / "results"
    _write_arm(real, synthetic=False)
    _write_arm(synthetic, synthetic=True)

    view = build_combined_training_view("A2", real, synthetic, output)
    data = yaml.safe_load((view / "data.yaml").read_text(encoding="utf-8"))
    manifest = json.loads(
        (view / "combined_view_manifest.json").read_text(encoding="utf-8")
    )

    assert data["train"] == [
        str((real / "train" / "images").resolve()),
        str((synthetic / "train" / "images").resolve()),
    ]
    assert data["val"] == str((real / "val" / "images").resolve())
    assert data["test"] == str((real / "test" / "images").resolve())
    assert manifest["real_train_images"] == 1
    assert manifest["synthetic_train_images"] == 1
    assert manifest["files_copied"] == 0
    assert not list((view / "train" / "images").iterdir())


def test_combined_training_view_rejects_synthetic_with_real_train(tmp_path: Path) -> None:
    real = tmp_path / "real"
    synthetic = tmp_path / "synthetic"
    _write_arm(real, synthetic=False)
    _write_arm(synthetic, synthetic=True)
    manifest = synthetic / "metadata" / "generation_manifest.json"
    manifest.write_text(
        json.dumps({"include_real_train": True}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="menduplikasi A0"):
        build_combined_training_view("A1", real, synthetic, tmp_path / "results")
