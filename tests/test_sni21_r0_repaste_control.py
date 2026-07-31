import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from coffee_detector.run_sni21_r0_repaste_control import (
    classify_repaste_retention,
    match_assets_to_boxes,
    prepare_r0_repaste_dataset,
)


def test_matching_never_crosses_classes_and_orders_aspect_ratios() -> None:
    boxes = [
        {"index": 0, "class_id": 1, "aspect_ratio": 2.1},
        {"index": 1, "class_id": 1, "aspect_ratio": 1.0},
        {"index": 2, "class_id": 2, "aspect_ratio": 4.0},
    ]
    assets = [
        {"asset_id": "wide", "class_id": 1, "intrinsic_aspect_ratio": 2.0},
        {"asset_id": "square", "class_id": 1, "intrinsic_aspect_ratio": 1.1},
        {"asset_id": "wrong-class", "class_id": 3, "intrinsic_aspect_ratio": 4.0},
    ]

    pairs = match_assets_to_boxes(boxes, assets)

    assert [(box["index"], asset["asset_id"]) for box, asset in pairs] == [
        (0, "wide"),
        (1, "square"),
    ]


def _write_tiny_real_dataset(root: Path) -> None:
    (root / "val/images").mkdir(parents=True)
    (root / "val/labels").mkdir(parents=True)
    (root / "train/images").mkdir(parents=True)
    (root / "train/labels").mkdir(parents=True)
    image = np.full((40, 60, 3), 240, dtype=np.uint8)
    Image.fromarray(image).save(
        root / "val/images/faruq_segmentation__bean_jpg.rf.abc.jpg"
    )
    (root / "val/labels/faruq_segmentation__bean_jpg.rf.abc.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )
    names = {index: name for index, name in enumerate(
        (
            "biji_berkulit_tanduk", "biji_berlubang_lebih_satu",
            "biji_berlubang_satu", "biji_bertutul_tutul", "biji_coklat",
            "biji_hitam", "biji_hitam_pecah", "biji_hitam_sebagian",
            "biji_muda", "biji_normal", "biji_pecah", "kopi_gelondong",
            "kulit_kopi_ukuran_besar", "kulit_kopi_ukuran_kecil",
            "kulit_kopi_ukuran_sedang", "kulit_tanduk_ukuran_besar",
            "kulit_tanduk_ukuran_kecil", "kulit_tanduk_ukuran_sedang",
            "tanah_batu_ranting_besar", "tanah_batu_ranting_kecil",
            "tanah_batu_ranting_sedang",
        )
    )}
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(root), "train": "train/images", "val": "val/images", "names": names},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_prepare_repaste_changes_only_target_box(tmp_path) -> None:
    real = tmp_path / "real"
    library = tmp_path / "library"
    output = tmp_path / "repaste"
    _write_tiny_real_dataset(real)
    asset_path = library / "assets/biji_berkulit_tanduk/asset.png"
    asset_path.parent.mkdir(parents=True)
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    rgba[:, :, :3] = (120, 60, 20)
    rgba[:, :, 3] = 255
    Image.fromarray(rgba, "RGBA").save(asset_path)
    classes = {
        str(index): name
        for index, name in enumerate(yaml.safe_load((real / "data.yaml").read_text())["names"].values())
    }
    (library / "object_library.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "assets": [
                    {
                        "asset_id": "asset",
                        "class_id": 0,
                        "class_name": "biji_berkulit_tanduk",
                        "image": "assets/biji_berkulit_tanduk/asset.png",
                        "intrinsic_aspect_ratio": 1.5,
                        "source_split": "val",
                        "source_dataset": "faruq_segmentation",
                        "source_parent_id": "beanjpg",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = prepare_r0_repaste_dataset(real, library, output, minimum_coverage=1.0)

    assert report["coverage"] == 1.0
    rendered = np.asarray(
        Image.open(output / "val/images/faruq_segmentation__bean_jpg.rf.abc.png")
    )
    assert np.all(rendered[0, 0] == 240)
    assert tuple(rendered[20, 30]) == (120, 60, 20)
    assert not (output / "test").exists()


@pytest.mark.parametrize(
    ("map_retention", "class_retention", "expected"),
    [
        (0.9, 0.8, "cutout_repaste_not_primary_cause"),
        (0.9, 0.6, "cutout_repaste_material_partial_cause"),
        (0.4, 0.9, "cutout_repaste_dominant_cause"),
    ],
)
def test_retention_interpretation_is_frozen(
    map_retention, class_retention, expected
) -> None:
    assert classify_repaste_retention(map_retention, class_retention) == expected
