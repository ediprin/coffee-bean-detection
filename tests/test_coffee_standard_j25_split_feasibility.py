import json
from pathlib import Path

from coffee_detector.analysis.coffee_standard_j25_split_feasibility import (
    audit_j25_split_feasibility,
)
from coffee_detector.data.prepare_coffee_standard_primary import J25_CLASSES
from coffee_detector.data.prepare_coffee_standard_primary_v2 import FORMAT


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_identity_ceiling_rejects_box_rich_but_parent_poor_dataset(tmp_path: Path) -> None:
    components = []
    manifest = []
    for index in range(10):
        counts = [1] * len(J25_CLASSES)
        # Class zero has hundreds of boxes but only four independent identities.
        counts[0] = 50 if index < 4 else 0
        component_id = f"parent-{index}"
        components.append({"component_id": component_id, "class_counts": counts})
        manifest.append(
            {
                "component_id": component_id,
                "output_split": ("train", "val", "test")[index % 3],
            }
        )
    summary = {"format": FORMAT, "training_authorized": False}
    result = audit_j25_split_feasibility(
        _write(tmp_path / "components.json", components),
        _write(tmp_path / "manifest.json", manifest),
        _write(tmp_path / "summary.json", summary),
        tmp_path / "audit.json",
    )

    assert result["global_theoretical_maximum_common_heldout_identities"] == 1
    assert result["decision"] == "FAIL_FIXED_HOLDOUT_PRIMARY_BENCHMARK"
    assert result["training_authorized"] is False
    assert result["test_images_accessed_by_model"] is False


def test_rejects_degenerate_one_identity_threshold(tmp_path: Path) -> None:
    try:
        audit_j25_split_feasibility("a", "b", "c", "d", minimum_heldout_identities=1)
    except ValueError as error:
        assert "minimal 2" in str(error)
    else:
        raise AssertionError("Expected ValueError")
