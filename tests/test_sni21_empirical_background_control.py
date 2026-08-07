import numpy as np
import json
import yaml
from PIL import Image

from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES
from coffee_detector.run_sni21_empirical_background_control import (
    _donor_pairs,
    _inpaint_object,
    _paired_bootstrap,
    prepare_empirical_background_dataset,
)


def test_donor_pairing_is_within_canvas_derangement() -> None:
    rows = [
        {
            "sample_id": f"sample_{index}",
            "source_asset_id": f"asset_{index}",
            "canvas_size": [100, 120],
        }
        for index in range(5)
    ]

    pairs = _donor_pairs(rows, seed=42)

    assert len(pairs) == 5
    assert {donor["sample_id"] for _, donor in pairs} == {
        row["sample_id"] for row in rows
    }
    assert all(target["sample_id"] != donor["sample_id"] for target, donor in pairs)
    assert all(target["canvas_size"] == donor["canvas_size"] for target, donor in pairs)


def test_inpainting_preserves_canvas_and_changes_masked_region() -> None:
    array = np.full((80, 100, 3), 220, dtype=np.uint8)
    array[30:50, 40:60] = (20, 30, 40)
    image = Image.fromarray(array)

    result = _inpaint_object(image, [40, 30, 60, 50])

    assert result.size == image.size
    restored = np.asarray(result)
    assert not np.array_equal(restored[35:45, 45:55], array[35:45, 45:55])


def test_paired_bootstrap_detects_recovery() -> None:
    rows = []
    for class_id in range(3):
        for index in range(10):
            rows.append(
                {
                    "class_id": class_id,
                    "before_correct": int(index == 0),
                    "after_correct": 1,
                }
            )

    result = _paired_bootstrap(rows, iterations=500, seed=42)

    assert np.isclose(result["point"], 0.9)
    assert result["ci95"][0] > 0
    assert result["probability_above_zero"] > 0.99


def test_prepare_empirical_background_dataset_is_paired_and_test_locked(tmp_path) -> None:
    real = tmp_path / "real"
    library = tmp_path / "library"
    fullframe = tmp_path / "fullframe"
    output = tmp_path / "output"
    for split in ("train", "val"):
        (real / split / "images").mkdir(parents=True)
        (real / split / "labels").mkdir(parents=True)
    names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    (real / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(real), "train": "train/images", "val": "val/images", "names": names},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    manifest = []
    assets = []
    for index in range(4):
        source_name = f"source_{index}.jpg"
        canvas = np.full((100, 100, 3), 210 - index * 10, dtype=np.uint8)
        canvas[40:60, 40:60] = (20 + index, 30, 40)
        Image.fromarray(canvas).save(real / "val/images" / source_name)
        asset_path = library / f"assets/asset_{index}.png"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        rgba[:, :, :3] = (20 + index, 30, 40)
        rgba[:, :, 3] = 255
        Image.fromarray(rgba, "RGBA").save(asset_path)
        manifest.append(
            {
                "sample_id": f"fc_{index:04d}_asset_{index}",
                "class_id": index,
                "source_image": source_name,
                "source_asset_id": f"asset_{index}",
                "canvas_size": [100, 100],
                "object_xyxy": [40, 40, 60, 60],
            }
        )
        assets.append({"asset_id": f"asset_{index}", "image": f"assets/asset_{index}.png"})
    fullframe.mkdir(parents=True)
    (fullframe / "local_context_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (library / "object_library.json").write_text(
        json.dumps({"assets": assets}), encoding="utf-8"
    )

    report = prepare_empirical_background_dataset(
        real, library, fullframe, output, seed=42
    )

    assert report["samples"] == 4
    assert report["classes"] == 4
    assert len(list((output / "val/images").glob("*.jpg"))) == 4
    rows = json.loads((output / "empirical_background_manifest.json").read_text())
    assert len({row["donor_asset_id"] for row in rows}) == 4
    assert all(row["target_asset_id"] != row["donor_asset_id"] for row in rows)
    assert not (output / "test").exists()
