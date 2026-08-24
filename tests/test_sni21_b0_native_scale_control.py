import json

import pytest

from coffee_detector.run_sni21_b0_native_scale_control import (
    audit_paired_scene_draws,
    classify_scale_recovery,
    derive_native_scale,
)


def test_derive_native_scale_uses_box_and_image_long_sides(tmp_path) -> None:
    records = tmp_path / "prediction_records.jsonl"
    rows = [
        {
            "width": 200,
            "height": 100,
            "ground_truth": [{"class_id": 0, "xyxy": [0, 0, 20, 10]}],
        },
        {
            "width": 100,
            "height": 400,
            "ground_truth": [{"class_id": 1, "xyxy": [0, 0, 40, 80]}],
        },
    ]
    records.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    report = derive_native_scale(
        records, lower_quantile=0.0, upper_quantile=1.0
    )

    assert report["images"] == 2
    assert report["boxes"] == 2
    assert report["selected_interval"] == pytest.approx([0.1, 0.2])


def _write_scene_metadata(path, asset: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "generation_seed": 42}],
                "annotations": [
                    {
                        "image_id": 1,
                        "category_id": 3,
                        "source_asset_id": asset,
                        "source_parent_id": "parent",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_pairing_audit_requires_exact_asset_draw(tmp_path) -> None:
    original = tmp_path / "original.json"
    same = tmp_path / "same.json"
    different = tmp_path / "different.json"
    _write_scene_metadata(original, "asset-a")
    _write_scene_metadata(same, "asset-a")
    _write_scene_metadata(different, "asset-b")

    assert audit_paired_scene_draws(original, same)["exact_draw_match"] is True
    mismatch = audit_paired_scene_draws(original, different)
    assert mismatch["exact_draw_match"] is False
    assert mismatch["mismatch_count"] == 1


@pytest.mark.parametrize(
    ("recovery", "expected"),
    [
        (0.50, "scale_explains_majority"),
        (0.20, "scale_is_material_partial_cause"),
        (0.19, "scale_alone_does_not_explain_collapse"),
    ],
)
def test_recovery_interpretation_is_frozen(recovery, expected) -> None:
    assert classify_scale_recovery(recovery) == expected
