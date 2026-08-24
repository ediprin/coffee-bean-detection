import json
from pathlib import Path

import pytest
import yaml

import coffee_detector.run_sni21_vadcp_setup as runner_module
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def _write_verified_real(root: Path, *, test_locked: bool = True) -> None:
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {
                    index: name
                    for index, name in enumerate(SNI21_CLASSES)
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "audit.json").write_text(
        json.dumps(
            {
                "training_ready": True,
                "test_locked": test_locked,
            }
        ),
        encoding="utf-8",
    )
    (root / "post_materialization_audit.json").write_text(
        json.dumps(
            {
                "safe_for_training": True,
                "cross_split_duplicate_components": 0,
            }
        ),
        encoding="utf-8",
    )


def test_sni21_setup_rejects_real_dataset_without_locked_test(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    _write_verified_real(real, test_locked=False)

    with pytest.raises(RuntimeError, match="test_locked"):
        runner_module.run_sni21_vadcp_setup(
            real,
            tmp_path / "crop",
            tmp_path / "output",
            synthetic_images=1,
        )


def test_sni21_setup_generates_paired_arms_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    crop = tmp_path / "crop"
    output = tmp_path / "output"
    _write_verified_real(real)
    crop.mkdir()
    names = {
        index: name for index, name in enumerate(SNI21_CLASSES)
    }
    generated = []

    def fake_prepare(_crop, library_root, **kwargs):
        library_root.mkdir(parents=True)
        payload = {
            "source": {
                "type": "sni_crop_manifest",
                "source_split": "train",
                "seed": kwargs["seed"],
                "max_normal_assets": kwargs["max_normal_assets"],
                "max_defect_assets_per_class": (
                    kwargs["max_defect_assets_per_class"]
                ),
            },
            "classes": {str(key): value for key, value in names.items()},
            "audit": {
                "assets": 21,
                "assets_by_source_split": {"train": 21},
            },
        }
        (library_root / "object_library.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    def fake_calibration(*_args, **_kwargs):
        return names, object(), {"policy": "source_empirical"}

    def fake_save(_calibration, path):
        path.write_text("{}", encoding="utf-8")
        return path

    def fake_generate(_real, _library, arm_root, **kwargs):
        generated.append(kwargs)
        metadata = arm_root / "metadata"
        metadata.mkdir(parents=True)
        manifest = {
            "mode": kwargs["mode"],
            "preset": kwargs["preset"],
            "seed": kwargs["seed"],
            "synthetic_images": kwargs["synthetic_images"],
            "include_real_train": kwargs["include_real_train"],
            "materialize_real_splits": kwargs["materialize_real_splits"],
            "classes": kwargs["target_names"],
        }
        (metadata / "generation_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return manifest

    def fake_audit(*_args, **_kwargs):
        return {
            "safe_for_training": True,
            "geometry_ready": True,
            "synthetic_images": 3,
            "synthetic_annotations": 9,
            "labeled_instances_by_visibility": {},
            "focus_target_hit_rate": {},
            "scene_density": {},
            "labeled_scene_density": {},
            "scene_modes": {},
            "repeated_assets": 0,
            "geometry_target_hit_rate": 1.0,
            "geometry_fallbacks": 0,
            "geometry_fallback_rate": 0.0,
            "warnings": [],
            "errors": [],
            "error_count": 0,
        }

    def fake_visual(data_root, visual_root, **_kwargs):
        visual_root.mkdir(parents=True)
        return {
            "raw_contact_sheet": str(visual_root / "raw.jpg"),
            "contact_sheet": str(visual_root / "overlay.jpg"),
        }

    monkeypatch.setattr(
        runner_module, "prepare_sni_crop_manifest_library", fake_prepare
    )
    monkeypatch.setattr(
        runner_module, "build_sni_crop_calibration", fake_calibration
    )
    monkeypatch.setattr(
        runner_module, "save_scene_calibration", fake_save
    )
    monkeypatch.setattr(
        runner_module, "generate_vadcp_dataset", fake_generate
    )
    monkeypatch.setattr(runner_module, "audit_vadcp_dataset", fake_audit)
    monkeypatch.setattr(runner_module, "print_audit_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner_module, "run_vadcp_visual_audit", fake_visual
    )

    result = runner_module.run_sni21_vadcp_setup(
        real,
        crop,
        output,
        synthetic_images=3,
        objects_min=4,
        objects_max=5,
        object_library_root=tmp_path / "shared-library",
        visual_samples=1,
    )

    assert [item["mode"] for item in generated] == ["naive", "visibility"]
    assert all(item["seed"] == 42 for item in generated)
    assert all(item["target_names"] == names for item in generated)
    assert result["training_ready"] is True
    assert result["training_executed"] is False
    assert result["test_accessed"] is False
    assert result["object_library"].endswith("shared-library")
