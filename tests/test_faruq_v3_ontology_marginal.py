from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_ontology_marginal import (
    _cached_report,
    _persist_provenance,
    _validate_prerequisites,
    compare_screening,
)


def _metrics(**updates) -> dict:
    payload = {
        "macro_map50_95": 0.80,
        "bottom3_class_map50_95": 0.68,
        "worst_class_map50_95": 0.65,
        "proposal_accessibility": 0.95,
        "conditional_top1_accuracy": 0.75,
    }
    payload.update(updates)
    return payload


def test_screening_requires_semantic_gain_over_baseline_and_control() -> None:
    candidate = _metrics(
        macro_map50_95=0.81,
        bottom3_class_map50_95=0.69,
        worst_class_map50_95=0.66,
        proposal_accessibility=0.95,
        conditional_top1_accuracy=0.78,
    )
    assert compare_screening(candidate, _metrics(), semantic_control=False)["decision"] == "PASS"
    control = _metrics(macro_map50_95=0.804, conditional_top1_accuracy=0.77)
    assert compare_screening(candidate, control, semantic_control=True)["decision"] == "PASS"
    candidate["macro_map50_95"] = 0.805
    assert compare_screening(candidate, control, semantic_control=True)["decision"] == "FAIL"


def test_prerequisites_reject_test_before_reading_reports(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "test").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="tidak boleh menyediakan test"):
        _validate_prerequisites(
            data,
            tmp_path / "missing-grouped.json",
            tmp_path / "missing-support.json",
            tmp_path / "missing-static.json",
        )


def test_cached_report_uses_checkpoint_fingerprint_not_drive_path(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    payload = {"test_images_accessed": False, "metrics": {}}
    _persist_provenance(path, payload, "abc", "evaluation")
    assert _cached_report(path, "abc", "evaluation") is not None
    assert _cached_report(path, "different", "evaluation") is None
    assert _cached_report(path, "abc", "diagnostic") is None
