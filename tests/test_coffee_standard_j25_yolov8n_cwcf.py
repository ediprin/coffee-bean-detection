import ast
import json
from pathlib import Path

import pytest
import yaml

from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import METRICS
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf import (
    PROTOCOL,
    V8N_BASELINE_PROTOCOL,
    build_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v8n_cwcf1_keeps_baseline_training_and_original_cwcf():
    candidate = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF1.yaml").read_text(encoding="utf-8")
    )
    baseline = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_MATCHED.yaml").read_text(encoding="utf-8")
    )
    original = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/CWCF1.yaml").read_text(encoding="utf-8")
    )
    assert candidate["dataset"] == baseline["dataset"]
    assert candidate["train"] == baseline["train"]
    assert candidate["sampler"] == baseline["sampler"] == "none"
    assert candidate["cwcf"] == original["cwcf"]
    assert candidate["weights"] == baseline["weights"] == "yolov8n.pt"


def test_pinned_yolov8n_model_is_n_scale_p3_detect():
    model = yaml.safe_load(
        (ROOT / "configs/coffee_fg/models/yolov8n-p3.yaml").read_text(encoding="utf-8")
    )
    assert model["scale"] == "n"
    assert model["scales"]["n"] == [0.33, 0.25, 1024]
    assert model["head"][-1][0] == [15, 18, 21]
    assert model["head"][-1][2] == "Detect"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_decision_promotes_only_macro_and_bottom3_positive(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    common = {"seed": 42, "test_images_accessed": False}
    _write(
        baseline,
        {
            **common,
            "protocol": V8N_BASELINE_PROTOCOL,
            "metrics": _metrics(.70, .32, .047),
            "target_class_map50_95": .047,
        },
    )
    _write(
        candidate,
        {
            **common,
            "protocol": PROTOCOL,
            "metrics": _metrics(.71, .33, .040),
            "target_class_map50_95": .040,
        },
    )
    result = build_decision(baseline, candidate, tmp_path / "decision.json")
    assert result["decision"] == "PASS_CONFIRMATION_SCREEN"
    assert result["next_step"] == "PAIRED_MULTI_SEED_CONFIRMATION"
    assert result["v8n_cwcf1_minus_v8n"]["macro_map50_95"] == pytest.approx(.01)
    assert result["v8n_cwcf1_minus_v8n"]["bottom3_class_map50_95"] == pytest.approx(.01)
    assert result["criteria"]["worst_class_is_descriptive_not_a_promotion_gate"] is True
    assert result["target_delta"] == pytest.approx(-.007)
    assert result["test_opened"] is False


def test_decision_rejects_aggregate_no_advantage(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    common = {"seed": 42, "test_images_accessed": False}
    _write(
        baseline,
        {
            **common,
            "protocol": V8N_BASELINE_PROTOCOL,
            "metrics": _metrics(.70, .32, .047),
            "target_class_map50_95": .047,
        },
    )
    _write(
        candidate,
        {
            **common,
            "protocol": PROTOCOL,
            "metrics": _metrics(.69, .31, .040),
            "target_class_map50_95": .040,
        },
    )
    result = build_decision(baseline, candidate, tmp_path / "decision.json")
    assert result["decision"] == "NO_ADVANTAGE_SCREEN"
    assert result["next_step"] == "RETAIN_V8N_MATCHED"


def test_protocol_and_notebook_keep_test_locked():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_YOLOV8N_CWCF1_PROTOCOL_2026-09-20.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "same CWCF" in protocol or "same CWCF1" in protocol
    assert "PASS_CONFIRMATION_SCREEN" in protocol
    assert "Worst-class AP" in protocol

    notebook = json.loads(
        (
            ROOT
            / "notebooks/Coffee_Standard_J25_YOLOv8n_CWCF1_Seed42_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-yolov8n-cwcf1'" in code
    assert "V8N_RESULT" in code
    assert "run_coffee_standard_j25_yolov8n_cwcf" in code
    assert "--authorize-training" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
