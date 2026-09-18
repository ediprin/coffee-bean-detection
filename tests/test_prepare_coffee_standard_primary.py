from pathlib import Path

import yaml
from PIL import Image

from coffee_detector.data.prepare_coffee_standard_primary import (
    J25_CLASSES,
    prepare_coffee_standard_primary,
)


def _write_source(root: Path) -> None:
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
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
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    # Each parent carries all classes so every split can satisfy ontology
    # coverage in this compact fixture. Parent 0 deliberately crosses splits.
    for parent_index in range(20):
        for sibling_index, split in enumerate(("train", "valid") if parent_index == 0 else ("train",)):
            stem = f"parent-{parent_index}.rf.{sibling_index:032x}"
            image_root = root / split / "images"
            label_root = root / split / "labels"
            image_root.mkdir(parents=True, exist_ok=True)
            label_root.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (parent_index * 13, sibling_index * 31, 70)).save(
                image_root / f"{stem}.jpg"
            )
            rows = [
                f"{class_id} 0.5 0.5 0.1 0.1" for class_id in range(len(J25_CLASSES))
            ]
            (label_root / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    # Empty official test directories keep the source layout explicit.
    (root / "test" / "images").mkdir(parents=True)
    (root / "test" / "labels").mkdir(parents=True)


def test_primary_candidate_is_grouped_deterministic_and_training_locked(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)
    first = prepare_coffee_standard_primary(source, tmp_path / "first", seed=42, link_mode="copy")
    second = prepare_coffee_standard_primary(source, tmp_path / "second", seed=42, link_mode="copy")

    assert first["source_derivative_images"] == 21
    assert first["identity_components"] == 20
    assert first["selected_representatives"] == 20
    assert first["discarded_generated_siblings"] == 1
    assert first["cross_split_parent_ids"] == 0
    assert first["cross_split_exact_hashes"] == 0
    assert first["technical_split_ready"] is True
    assert first["training_authorized"] is False
    assert first["test_access_authorized"] is False
    assert first["images_by_split"] == second["images_by_split"]
    assert first["class_distribution"] == second["class_distribution"]

    output_yaml = yaml.safe_load((tmp_path / "first" / "data.yaml").read_text(encoding="utf-8"))
    assert list(output_yaml["names"].values()) == list(J25_CLASSES)
    assert output_yaml["train"] == "train/images"
    assert output_yaml["val"] == "val/images"
    assert output_yaml["test"] == "test/images"


def test_primary_candidate_rejects_ontology_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)
    payload = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    payload["names"][0] = "renamed"
    (source / "data.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )

    try:
        prepare_coffee_standard_primary(source, tmp_path / "output", link_mode="copy")
    except ValueError as error:
        assert "Ontologi export" in str(error)
    else:
        raise AssertionError("Ontology drift harus ditolak")


def test_primary_candidate_discovers_one_nested_yolo_root(tmp_path: Path) -> None:
    source = tmp_path / "archive" / "nested" / "dataset"
    _write_source(source)

    result = prepare_coffee_standard_primary(
        tmp_path / "archive", tmp_path / "output", seed=42, link_mode="copy"
    )

    assert Path(result["source_root"]) == source.resolve()
