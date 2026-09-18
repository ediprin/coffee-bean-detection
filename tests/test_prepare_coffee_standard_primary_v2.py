from pathlib import Path

import yaml
import pytest
from PIL import Image

from coffee_detector.data.prepare_coffee_standard_primary import J25_CLASSES
from coffee_detector.data.prepare_coffee_standard_primary_v2 import (
    _build_identity_components,
    prepare_coffee_standard_primary_v2,
)
from coffee_detector.dataset import collect_records, discover_layout


def _write_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": {index: name for index, name in enumerate(J25_CLASSES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)

    rows = "\n".join(f"{class_id} 0.5 0.5 0.08 0.08" for class_id in range(25)) + "\n"
    for parent in range(22):
        sibling_count = 3 if parent in {0, 1} else 1
        for sibling in range(sibling_count):
            stem = f"bean-{parent}.rf.{parent:04x}{sibling:028x}"
            if parent == 0:
                colour = (50 + 5 * sibling, 70, 90)
            elif parent == 1:
                colour = (100 + 5 * sibling, 120, 140)
            else:
                colour = ((17 * parent) % 255, (43 * parent) % 255, (79 * parent) % 255)
            Image.new("RGB", (48, 48), colour).save(
                root / "train" / "images" / f"{stem}.jpg"
            )
            label = rows
            if parent == 0 and sibling == 2:
                label += "0 0.25 0.25 0.05 0.05\n"
            (root / "train" / "labels" / f"{stem}.txt").write_text(label, encoding="utf-8")


def test_v2_quarantines_disagreement_and_never_authorizes_training(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(source)
    with pytest.raises(RuntimeError, match="RETRACTED"):
        prepare_coffee_standard_primary_v2(
            source, tmp_path / "grouped", seed=42, link_mode="copy"
        )


def test_v2_visual_medoid_is_not_maximum_box_selector(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dataset(source)
    records, errors = collect_records(discover_layout(source), compute_visual_features=True)
    assert not errors

    components, representatives, quarantined = _build_identity_components(records, seed=42)

    assert len(quarantined) == 1
    assert all("bean-0" not in row.image_path.name for row in representatives.values())
    selected = next(row for row in representatives.values() if row.image_path.name.startswith("bean-1."))
    with Image.open(selected.image_path) as image:
        assert abs(image.getpixel((0, 0))[0] - 105) <= 2
    assert all(row["selection_policy"].startswith("visual_medoid") for row in components)
