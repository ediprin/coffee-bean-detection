import json
from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_af2_igem_paired_confirmation import (
    ARMS,
    CONFIRMATION_SEEDS,
    _aggregate,
    _decision,
    _validate_d0ft_confirmation,
    _validate_seed42_results,
    run_faruq_v3_af2_igem_paired_confirmation,
)


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _row(macro: float, bottom: float, worst: float) -> dict:
    return dict(zip(METRICS, (macro, bottom, worst)))


def _af2_payload() -> dict:
    return {
        "protocol": "faruq-v3-lfdet-afab-breadth-screening-v1",
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "controls": {"D0FT": _row(0.86, 0.75, 0.72)},
        "candidate": {"AF2": _row(0.88, 0.80, 0.79)},
        "decisions": {"AF2": {"decision": "RETAIN"}},
    }


def _igem_payload() -> dict:
    return {
        "protocol": "faruq-v3-igem-classification-guidance-screening-v1",
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "controls": {"D0FT": _row(0.86, 0.75, 0.72)},
        "candidate": {"IGEM1": _row(0.88, 0.82, 0.82)},
        "decision": "RETAIN",
    }


def _d0ft_payload() -> dict:
    return {
        "protocol": "faruq-v3-acmc-paired-optimization-confirmation-v1",
        "seeds": [42, 123, 2026],
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "decision": "PASS",
        "per_seed": {
            "42": {"results": {"D0FT": _row(0.86, 0.75, 0.72)}},
            "123": {"results": {"D0FT": _row(0.87, 0.76, 0.73)}},
            "2026": {"results": {"D0FT": _row(0.86, 0.77, 0.74)}},
        },
    }


def test_seed42_and_d0ft_sources_must_be_compatible(tmp_path: Path) -> None:
    af2_path, igem_path, d0ft_path = (
        tmp_path / "af2.json",
        tmp_path / "igem.json",
        tmp_path / "d0ft.json",
    )
    af2_path.write_text(json.dumps(_af2_payload()), encoding="utf-8")
    igem_path.write_text(json.dumps(_igem_payload()), encoding="utf-8")
    d0ft_path.write_text(json.dumps(_d0ft_payload()), encoding="utf-8")
    seed42 = _validate_seed42_results(af2_path, igem_path)
    assert seed42["AF2"]["macro_map50_95"] == pytest.approx(0.88)
    assert _validate_d0ft_confirmation(d0ft_path, seed42["D0FT"])["decision"] == "PASS"

    payload = _igem_payload()
    payload["test_opened"] = True
    igem_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="bukan RETAIN"):
        _validate_seed42_results(af2_path, igem_path)


def test_paired_aggregate_and_gate_are_independent_per_candidate() -> None:
    per_seed = {
        "42": {
            "D0FT": _row(0.86, 0.75, 0.72),
            "AF2": _row(0.88, 0.80, 0.79),
            "IGEM1": _row(0.88, 0.82, 0.82),
        },
        "123": {
            "D0FT": _row(0.87, 0.76, 0.73),
            "AF2": _row(0.88, 0.78, 0.77),
            "IGEM1": _row(0.86, 0.75, 0.72),
        },
        "2026": {
            "D0FT": _row(0.86, 0.77, 0.74),
            "AF2": _row(0.87, 0.79, 0.76),
            "IGEM1": _row(0.85, 0.75, 0.70),
        },
    }
    af2 = _aggregate(per_seed, "AF2")
    igem = _aggregate(per_seed, "IGEM1")
    af2_criteria, af2_decision = _decision(af2)
    _, igem_decision = _decision(igem)
    assert af2_decision == "PASS"
    assert all(af2_criteria.values())
    assert igem_decision == "FAIL"
    assert set(af2["macro_map50_95"]["deltas"]) == {"42", "123", "2026"}


def test_runner_rejects_changed_seeds_models_and_missing_authorization(tmp_path: Path) -> None:
    args = (
        tmp_path / "data",
        tmp_path / "grouped.json",
        tmp_path / "af2.json",
        tmp_path / "igem.json",
        tmp_path / "d0ft.json",
        (tmp_path / "d0-123.pt", tmp_path / "d0-2026.pt"),
        tmp_path / "output",
    )
    with pytest.raises(ValueError, match="seed"):
        run_faruq_v3_af2_igem_paired_confirmation(
            *args, seeds=(42, *CONFIRMATION_SEEDS), authorize_training=True
        )
    with pytest.raises(ValueError, match="model"):
        run_faruq_v3_af2_igem_paired_confirmation(
            *args, models=("AF2",), authorize_training=True
        )
    with pytest.raises(RuntimeError, match="belum diotorisasi"):
        run_faruq_v3_af2_igem_paired_confirmation(*args)
    assert ARMS == ("AF2", "IGEM1")


def test_notebook_is_shared_drive_resumable_quiet_and_validation_only() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/Faruq_V3_AF2_IGEM_Paired_Confirmation_Colab.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "agent/af2-igem-paired-confirmation" in source
    assert "resolve_drive_project_root(required_relative_paths=REQUIRED)" in source
    assert "lfdet_afab_seed42_screening.json" in source
    assert "igem_seed42_screening.json" in source
    assert "D0_seed123/weights/best.pt" in source
    assert "D0_seed2026/weights/best.pt" in source
    assert "--models', 'AF2', 'IGEM1'" in source
    assert "--authorize-training" in source
    assert "stdout=log_stream" in source
    assert "stdout=subprocess.PIPE" not in source
    assert "time.sleep(60)" in source
    assert "[STATUS]" in source
    assert "split=test" not in source.lower()
    assert "test tidak" in source.lower()
