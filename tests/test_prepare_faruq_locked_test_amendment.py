import json
from pathlib import Path

import pytest

from coffee_detector.prepare_faruq_locked_test_amendment import (
    prepare_faruq_locked_test_amendment,
)
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def _eligibility(path: Path, *, unsafe: bool = False) -> Path:
    gates = {
        "zero_development_parent_overlap": not unsafe,
        "zero_development_hash_overlap": True,
        "one_image_per_test_parent": True,
        "minimum_independent_images": True,
        "all_21_classes_present": True,
        "minimum_instances_per_class": False,
        "minimum_parents_per_class": False,
        "zero_quarantined_selected_images": True,
    }
    payload = {
        "format": "coffee_detector.faruq_locked_test_eligibility.v1",
        "decision": "FAIL",
        "training_executed": False,
        "inference_executed": False,
        "materialized_images": 129,
        "instances_by_class": {name: 5 for name in SNI21_CLASSES},
        "parents_by_class": {name: 4 for name in SNI21_CLASSES},
        "gates": gates,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_support_only_amendment_passes_before_inference(tmp_path: Path) -> None:
    result = prepare_faruq_locked_test_amendment(
        _eligibility(tmp_path / "eligibility.json"), tmp_path / "amendment.json"
    )
    assert result["decision"] == "PASS"
    assert result["source_v1_decision"] == "FAIL"
    assert result["primary_endpoint"] == "paired_three_seed_macro_map50_95_delta"
    assert result["further_tuning_authorized"] is False


def test_amendment_rejects_identity_gate_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="bukan hanya masalah support"):
        prepare_faruq_locked_test_amendment(
            _eligibility(tmp_path / "eligibility.json", unsafe=True),
            tmp_path / "amendment.json",
        )
