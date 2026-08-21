import json
from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_af2_mechanism_diagnostic import (
    build_af2_mechanism_summary,
)


def _report(
    path: Path,
    *,
    accessibility: float,
    conditional: float,
    correct: int,
    wrong: int,
) -> Path:
    matched = correct + wrong
    targets = 100
    payload = {
        "protocol": "faruq-v3-yolo26n-diagnostic-v1",
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "global": {
            "targets": targets,
            "accessible": int(accessibility * targets),
            "matched": matched,
            "correct_class": correct,
            "wrong_class": wrong,
            "missed": targets - matched,
            "proposal_accessibility": accessibility,
            "matched_recall": matched / targets,
            "localization_conditioned_class_accuracy": conditional,
            "oracle_class_accuracy_headroom": 1.0 - conditional,
        },
        "per_class": [
            {
                "class_name": "example",
                "targets": targets,
                "accessible": int(accessibility * targets),
                "matched": matched,
                "correct_class": correct,
                "proposal_accessibility": accessibility,
                "matched_recall": matched / targets,
                "localization_conditioned_class_accuracy": conditional,
            }
        ],
        "directional_confusions": {},
        "top_directional_confusions": [],
        "raw_candidate_sensitivity": {
            "500": {
                "proposal_accessibility": accessibility,
                "matched_recall": matched / targets,
                "localization_conditioned_class_accuracy": conditional,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_summary_attributes_classification_dominant(tmp_path: Path) -> None:
    controls = []
    candidates = []
    for seed in (42, 123, 2026):
        controls.append(
            _report(
                tmp_path / f"control-{seed}.json",
                accessibility=0.90,
                conditional=0.70,
                correct=63,
                wrong=27,
            )
        )
        candidates.append(
            _report(
                tmp_path / f"candidate-{seed}.json",
                accessibility=0.901,
                conditional=0.72,
                correct=65,
                wrong=25,
            )
        )

    result = build_af2_mechanism_summary(
        controls, candidates, tmp_path / "summary.json"
    )

    assert result["attribution"] == "CLASSIFICATION_DOMINANT"
    assert not result["localization_supported"]
    assert result["classification_supported"]
    assert result["aggregate"]["conditional_top1_accuracy"]["delta_mean"] == pytest.approx(0.02)
    assert result["test_images_accessed"] is False


def test_report_with_test_access_is_rejected(tmp_path: Path) -> None:
    controls = []
    candidates = []
    for seed in (42, 123, 2026):
        control = _report(
            tmp_path / f"control-{seed}.json",
            accessibility=0.90,
            conditional=0.70,
            correct=63,
            wrong=27,
        )
        candidate = _report(
            tmp_path / f"candidate-{seed}.json",
            accessibility=0.91,
            conditional=0.71,
            correct=64,
            wrong=26,
        )
        controls.append(control)
        candidates.append(candidate)
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    payload["test_images_accessed"] = True
    candidates[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="test terkunci"):
        build_af2_mechanism_summary(
            controls, candidates, tmp_path / "summary.json"
        )


def test_requires_exactly_three_paired_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tiga report"):
        build_af2_mechanism_summary([], [], tmp_path / "summary.json")
