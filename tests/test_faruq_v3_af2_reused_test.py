import json
from pathlib import Path

import pytest

from coffee_detector.experiments import run_faruq_v3_af2_reused_test as module
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _observation(correct: bool) -> list[dict]:
    target = {"class_id": 0, "xyxy": [0.0, 0.0, 10.0, 10.0]}
    prediction = {
        "class_id": 0,
        "score": 0.9,
        "xyxy": [0.0, 0.0, 10.0, 10.0]
        if correct
        else [20.0, 20.0, 30.0, 30.0],
    }
    return [{"image_name": "parent.jpg", "targets": [target], "predictions": [prediction]}]


def _report(manifest_hash: str, value: float, *, protocol: str) -> dict:
    return {
        "protocol": protocol,
        "split": "test",
        "test_manifest_sha256": manifest_hash,
        "training_executed": False,
        "test_images_accessed": True,
        "metrics": {
            "macro_map50_95": value,
            "bottom3_class_map50_95": value - 0.1,
            "worst_class_map50_95": value - 0.15,
            "map50_95_by_class": {name: value for name in SNI21_CLASSES},
        },
        "prediction_observations": _observation(value > 0.7),
    }


def test_build_summary_marks_positive_direction_without_confirmatory_claim(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "faruq_locked_test_manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    manifest_hash = module._sha256(manifest)
    controls, candidates = [], []
    for seed in module.SEEDS:
        controls.append(
            _write_json(
                tmp_path / f"D0FT_seed{seed}.json",
                _report(manifest_hash, 0.70, protocol="historical-locked-test"),
            )
        )
        candidates.append(
            _write_json(
                tmp_path / f"AF2_seed{seed}.json",
                _report(
                    manifest_hash,
                    0.72,
                    protocol="faruq-v3-af2-reused-test-posthoc-report-v1",
                ),
            )
        )

    result = module.build_af2_reused_test_summary(
        controls,
        candidates,
        manifest,
        tmp_path / "summary.json",
        bootstrap_iterations=10,
    )

    assert result["status"] == "POSTHOC_DIRECTION_POSITIVE"
    assert result["scientific_status"] == "REUSED_TEST_POSTHOC_NOT_LOCKED_CONFIRMATION"
    assert result["aggregate"]["macro_map50_95"]["delta_mean"] == pytest.approx(0.02)
    assert result["aggregate"]["macro_map50_95"]["improved_seeds"] == 3
    assert result["paired_parent_bootstrap"]["candidate_arm"] == "AF2"
    assert result["training_executed"] is False
    assert result["further_tuning_authorized"] is False


def test_build_summary_rejects_manifest_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "faruq_locked_test_manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    wrong_hash = "0" * 64
    controls, candidates = [], []
    for seed in module.SEEDS:
        controls.append(
            _write_json(
                tmp_path / f"D0FT_seed{seed}.json",
                _report(wrong_hash, 0.70, protocol="historical-locked-test"),
            )
        )
        candidates.append(
            _write_json(
                tmp_path / f"AF2_seed{seed}.json",
                _report(
                    wrong_hash,
                    0.72,
                    protocol="faruq-v3-af2-reused-test-posthoc-report-v1",
                ),
            )
        )

    with pytest.raises(RuntimeError, match="manifest test berbeda"):
        module.build_af2_reused_test_summary(
            controls,
            candidates,
            manifest,
            tmp_path / "summary.json",
            bootstrap_iterations=10,
        )


def test_reused_test_requires_three_paired_reports(tmp_path: Path) -> None:
    manifest = tmp_path / "faruq_locked_test_manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="masing-masing memerlukan tiga report"):
        module.build_af2_reused_test_summary([], [], manifest, tmp_path / "summary.json")
