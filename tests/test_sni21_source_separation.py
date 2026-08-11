from pathlib import Path

import yaml
from PIL import Image

from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES
from coffee_detector.separate_sni21_sources import separate_sni21_sources


def _write_combined(root: Path) -> None:
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for ordinal, source in enumerate(("adrian_detection", "faruq_segmentation")):
            name = f"{source}__{split}_{ordinal}.jpg"
            Image.new("RGB", (16, 16), (80 + ordinal, 40, 20)).save(
                root / split / "images" / name
            )
            (root / split / "labels" / Path(name).with_suffix(".txt")).write_text(
                f"{ordinal} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {index: name for index, name in enumerate(SNI21_CLASSES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_separation_keeps_sources_independent_and_test_locked(tmp_path: Path) -> None:
    combined = tmp_path / "combined"
    output = tmp_path / "separated"
    _write_combined(combined)

    summary = separate_sni21_sources(combined, output, link_mode="copy")

    assert summary["status"] == "complete"
    assert summary["test_locked"] is True
    assert summary["test_images_accessed"] is False
    assert summary["training_executed"] is False
    assert not (output / "adrian_detection" / "test").exists()
    assert not (output / "faruq_segmentation" / "test").exists()
    for source in ("adrian_detection", "faruq_segmentation"):
        train_names = {
            path.name for path in (output / source / "train" / "images").iterdir()
        }
        val_names = {
            path.name for path in (output / source / "val" / "images").iterdir()
        }
        assert train_names == {f"{source}__train_{int(source == 'faruq_segmentation')}.jpg"}
        assert val_names == {f"{source}__val_{int(source == 'faruq_segmentation')}.jpg"}
        data = yaml.safe_load((output / source / "data.yaml").read_text())
        assert list(data["names"].values()) == list(SNI21_CLASSES)
        assert "test" not in data


def test_separation_reuses_only_complete_matching_output(tmp_path: Path) -> None:
    combined = tmp_path / "combined"
    output = tmp_path / "separated"
    _write_combined(combined)
    first = separate_sni21_sources(combined, output, link_mode="copy")
    second = separate_sni21_sources(combined, output, link_mode="copy")
    assert second == first
