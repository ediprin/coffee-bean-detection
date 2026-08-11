import json
from pathlib import Path

from PIL import Image, ImageDraw

from coffee_detector.experiments import run_faruq_v3_synthetic_density_screening as screen
from coffee_detector.experiments.run_faruq_v3_synthetic_density_setup import (
    _audit_library,
    _prepare_faruq_polygon_library,
)
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def test_polygon_library_preserves_all_canonical_classes(tmp_path: Path) -> None:
    polygon_root = tmp_path / "polygon"
    images = []
    annotations = []
    manifest = []
    for class_id, _ in enumerate(SNI21_CLASSES):
        file_name = f"images/parent_{class_id:02d}_jpg.rf.abcdef.jpg"
        image_path = polygon_root / "train" / file_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (32, 32), "white")
        ImageDraw.Draw(image).rectangle(
            (8, 8, 23, 23), fill=(class_id * 9 % 255, 80, 180)
        )
        image.save(image_path)
        images.append(
            {"id": class_id, "file_name": file_name, "width": 32, "height": 32}
        )
        annotations.append(
            {
                "id": class_id,
                "image_id": class_id,
                "category_id": class_id,
                "segmentation": [[8, 8, 23, 8, 23, 23, 8, 23]],
            }
        )
        manifest.append(
            {
                "source_parent_id": f"parent{class_id:02d}jpg",
                "output_split": "val",
            }
        )
    categories = [
        {"id": class_id, "name": name}
        for class_id, name in enumerate(SNI21_CLASSES)
    ]
    for split in ("train", "val"):
        root = polygon_root / split
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "images": images if split == "train" else [],
            "annotations": annotations if split == "train" else [],
            "categories": categories,
        }
        (root / "_annotations.coco.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = _prepare_faruq_polygon_library(
        polygon_root, manifest_path, tmp_path / "library"
    )
    assert len(result["assets"]) == len(SNI21_CLASSES)
    assert result["classes"] == {
        str(class_id): name for class_id, name in enumerate(SNI21_CLASSES)
    }
    assert result["audit"]["failures"] == 0


def test_validation_library_audit_accepts_group_safe_parents(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    library_root = tmp_path / "library"
    data_root.mkdir()
    library_root.mkdir()
    parents = [f"parent-{index}" for index in range(len(SNI21_CLASSES))]
    manifest = [
        {"source_parent_id": "train-parent", "output_split": "train"},
        *[
            {"source_parent_id": parent, "output_split": "val"}
            for parent in parents
        ],
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    library = {
        "source": {
            "root": str(data_root),
            "source_split": "val",
            "source_group_split": "faruq_v3_validation",
        },
        "classes": {str(index): name for index, name in enumerate(SNI21_CLASSES)},
        "assets": [
            {"asset_id": str(index), "source_parent_id": parents[index]}
            for index in range(len(SNI21_CLASSES))
        ],
    }
    (library_root / "object_library.json").write_text(
        json.dumps(library), encoding="utf-8"
    )
    result = _audit_library(data_root, manifest_path, library_root)
    assert result["safe_for_development_diagnostic"] is True
    assert result["train_parent_overlap"] == []


def test_validation_library_audit_rejects_train_parent_overlap(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    library_root = tmp_path / "library"
    data_root.mkdir()
    library_root.mkdir()
    parents = [f"parent-{index}" for index in range(len(SNI21_CLASSES))]
    manifest = [
        {"source_parent_id": parents[0], "output_split": "train"},
        *[
            {"source_parent_id": parent, "output_split": "val"}
            for parent in parents[1:]
        ],
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    library = {
        "source": {
            "root": str(data_root),
            "source_split": "val",
            "source_group_split": "faruq_v3_validation",
        },
        "classes": {str(index): name for index, name in enumerate(SNI21_CLASSES)},
        "assets": [
            {"asset_id": str(index), "source_parent_id": parents[index]}
            for index in range(len(SNI21_CLASSES))
        ],
    }
    (library_root / "object_library.json").write_text(
        json.dumps(library), encoding="utf-8"
    )
    result = _audit_library(data_root, manifest_path, library_root)
    assert result["safe_for_development_diagnostic"] is False
    assert result["train_parent_overlap"] == [parents[0]]


def test_seed42_screening_uses_frozen_checkpoints_without_test(
    tmp_path: Path, monkeypatch
) -> None:
    arms = {}
    for index, arm in enumerate(screen.ARM_ORDER):
        root = tmp_path / arm
        root.mkdir()
        arms[arm] = {"root": str(root), "density": [index + 1, index + 2]}
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps(
            {
                "format": "coffee_detector.faruq_v3_synthetic_density_setup.v1",
                "ready_for_frozen_screening": True,
                "training_executed": False,
                "test_images_accessed": False,
                "source_split": "faruq_v3_validation",
                "arms": arms,
            }
        ),
        encoding="utf-8",
    )
    d0ft, acmc = tmp_path / "d0ft.pt", tmp_path / "acmc.pt"
    d0ft.write_bytes(b"d0ft")
    acmc.write_bytes(b"acmc")

    def fake_evaluate(checkpoint, checkpoint_hash, arm, arm_root, output, **kwargs):
        gain = 0.02 if checkpoint.name == "acmc.pt" else 0.0
        return {
            "metrics": {
                "macro_map50_95": 0.7 + gain,
                "bottom3_map50_95": 0.6 + gain,
                "worst_map50_95": 0.5 + gain,
            }
        }

    monkeypatch.setattr(screen, "_evaluate", fake_evaluate)
    result = screen.run_faruq_v3_synthetic_density_screening(
        setup, d0ft, acmc, tmp_path / "output", device=None
    )
    assert result["summary"]["macro_improved_conditions"] == 4
    assert result["training_executed"] is False
    assert result["test_images_accessed"] is False
    assert result["locked_test_conclusion_changed"] is False
