from pathlib import Path

import yaml
from PIL import Image

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.dataset import discover_layout
from coffee_detector.prepare_dataset import prepare_dataset


def _write_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    names = {0: "normal", 1: "defect"}
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": names,
            }
        ),
        encoding="utf-8",
    )
    colors = {
        "train": [(10, 10, 10), (30, 30, 30), (50, 50, 50)],
        "valid": [(80, 80, 80), (100, 100, 100)],
        "test": [(130, 130, 130), (160, 160, 160), (190, 190, 190)],
    }
    for split, split_colors in colors.items():
        for folder in ("images", "labels"):
            (root / split / folder).mkdir(parents=True, exist_ok=True)
        for index, color in enumerate(split_colors):
            stem = f"{split}-{index}"
            Image.new("RGB", (24, 24), color).save(root / split / "images" / f"{stem}.png")
            class_id = index % 2
            (root / split / "labels" / f"{stem}.txt").write_text(
                f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )


def test_audit_detects_cross_split_exact_duplicate(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_dataset(raw)
    source = raw / "train" / "images" / "train-0.png"
    duplicate = raw / "test" / "images" / "duplicate.png"
    duplicate.write_bytes(source.read_bytes())
    (raw / "test" / "labels" / "duplicate.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )

    audit = audit_dataset(raw, tmp_path / "audit.json")

    assert audit["cross_split_duplicate_components"] >= 1
    assert not audit["safe_for_training"]


def test_prepare_creates_non_overlapping_grouped_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_dataset(raw)
    output = tmp_path / "prepared"

    result = prepare_dataset(raw, output, seed=7, near_threshold=0)
    layout = discover_layout(output)
    audit = audit_dataset(output, tmp_path / "prepared-audit.json", near_threshold=0)

    assert layout.names == {0: "normal", 1: "defect"}
    assert sum(result["images"].values()) == 8
    assert set(result["images"]) == {"train", "val", "test"}
    assert audit["cross_split_duplicate_components"] == 0
    assert not audit["errors"]
