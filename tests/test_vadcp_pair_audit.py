from __future__ import annotations

import json
from pathlib import Path

import pytest

from coffee_detector.audit_vadcp_pair import audit_vadcp_pair


def _write_arm(
    root: Path,
    *,
    mode: str,
    visibility: tuple[float, float],
    ignored: tuple[int, int] = (0, 0),
    second_asset: str = "asset-b",
) -> None:
    metadata = {
        "info": {"format": "coffee_detector.vadcp.v2", "mode": mode},
        "categories": [
            {"id": 0, "name": "biji_muda"},
            {"id": 1, "name": "biji_bertutul_tutul"},
        ],
        "images": [
            {
                "id": 1,
                "file_name": "train/images/scene.jpg",
                "width": 100,
                "height": 100,
                "generation_seed": 1234,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [10, 10, 20, 20],
                "full_bbox": [10, 10, 20, 20],
                "full_area": 300,
                "ignore": ignored[0],
                "source_asset_id": "asset-a",
                "z_order": 0,
                "visibility_ratio": visibility[0],
                "target_bbox_ratio": 1.4,
                "achieved_bbox_ratio": 1.39,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 50, 20, 10],
                "full_bbox": [50, 50, 20, 10],
                "full_area": 150,
                "ignore": ignored[1],
                "source_asset_id": second_asset,
                "z_order": 1,
                "visibility_ratio": visibility[1],
                "target_bbox_ratio": 2.0,
                "achieved_bbox_ratio": 1.98,
            },
        ],
    }
    manifest = {"mode": mode, "preset": "sni_spread"}
    metadata_root = root / "metadata"
    metadata_root.mkdir(parents=True)
    (metadata_root / "instances_synthetic_train.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (metadata_root / "generation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_pair_audit_validates_plan_and_flags_visibility_risk(tmp_path: Path) -> None:
    a1, a2 = tmp_path / "A1", tmp_path / "A2"
    _write_arm(a1, mode="naive", visibility=(1.0, 0.9))
    _write_arm(a2, mode="visibility", visibility=(0.8, 0.4))

    report = audit_vadcp_pair(
        a1,
        a2,
        tmp_path / "report.json",
        real_train_boxes=10,
        real_median_bbox_area=0.08,
    )

    assert report["status"] == "REVIEW_LABEL_RISK"
    assert report["paired_contract"]["selection_and_geometry_plan_valid"]
    assert report["class_prior_comparison"]["total_variation_distance"] == 0
    assert report["semantic_label_risk_proxy"][
        "sensitive_instances_below_severe_visibility"
    ] == 1
    assert report["mixed_training_dominance"][
        "synthetic_share_of_mixed_train_boxes"
    ] == pytest.approx(2 / 12)
    assert report["real_to_synthetic_scale_shift"]["a2_to_real_ratio"] == pytest.approx(
        0.375
    )
    assert (tmp_path / "report.json").is_file()


def test_pair_audit_rejects_unpaired_asset_plan(tmp_path: Path) -> None:
    a1, a2 = tmp_path / "A1", tmp_path / "A2"
    _write_arm(a1, mode="naive", visibility=(1.0, 1.0))
    _write_arm(
        a2,
        mode="visibility",
        visibility=(1.0, 1.0),
        second_asset="different-asset",
    )

    report = audit_vadcp_pair(a1, a2, tmp_path / "report.json")

    assert report["status"] == "INVALID_PAIRING"
    assert not report["paired_contract"]["selection_and_geometry_plan_valid"]
    assert report["paired_contract"]["mismatch_examples"]


def test_pair_audit_rejects_invalid_visibility_thresholds(tmp_path: Path) -> None:
    a1, a2 = tmp_path / "A1", tmp_path / "A2"
    _write_arm(a1, mode="naive", visibility=(1.0, 1.0))
    _write_arm(a2, mode="visibility", visibility=(1.0, 1.0))

    with pytest.raises(ValueError, match="Threshold"):
        audit_vadcp_pair(
            a1,
            a2,
            tmp_path / "report.json",
            moderate_visibility=0.5,
            severe_visibility=0.75,
        )
