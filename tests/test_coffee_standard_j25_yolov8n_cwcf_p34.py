import ast
import json
from pathlib import Path

import pytest
import torch
import yaml

from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import METRICS
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf import (
    PROTOCOL as CWCF1_PROTOCOL,
)
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf_p34 import (
    PROTOCOL,
    _cwcf_without_injection,
    build_decision,
)
from coffee_detector.j25_cwcf import CWCFConfig, build_cwcf_model


ROOT = Path(__file__).resolve().parents[1]


def test_p34_changes_only_injection_location_contract():
    candidate = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF_P34.yaml").read_text(
            encoding="utf-8"
        )
    )
    baseline = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert candidate["dataset"] == baseline["dataset"]
    assert candidate["model"] == baseline["model"]
    assert candidate["weights"] == baseline["weights"]
    assert candidate["sampler"] == baseline["sampler"] == "none"
    assert candidate["train"] == baseline["train"]
    assert _cwcf_without_injection(candidate["cwcf"]) == _cwcf_without_injection(
        baseline["cwcf"]
    )

    base_frozen = CWCFConfig.from_mapping(baseline["cwcf"])
    p34_frozen = CWCFConfig.from_mapping(candidate["cwcf"])
    assert base_frozen.pyramid_injection == "p3p4p5"
    assert p34_frozen.pyramid_injection == "p3p4"


def test_p34_model_state_matches_full_cwcf1_at_initialization():
    model_yaml = str(ROOT / "configs/coffee_fg/models/yolov8n-p3.yaml")
    baseline = CWCFConfig()
    candidate = CWCFConfig(pyramid_injection="p3p4")
    full = build_cwcf_model(
        model_yaml, nc=25, source=None, seed=42, config=baseline, verbose=False
    )
    p34 = build_cwcf_model(
        model_yaml, nc=25, source=None, seed=42, config=candidate, verbose=False
    )
    lhs, rhs = full.state_dict(), p34.state_dict()
    assert list(lhs) == list(rhs)
    assert all(torch.equal(lhs[key], rhs[key]) for key in lhs)
    assert [full.model[-1]._cue_active(i) for i in range(3)] == [True, True, True]
    assert [p34.model[-1]._cue_active(i) for i in range(3)] == [True, True, False]


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_p34_decision_compares_against_cwcf1(tmp_path):
    baseline = tmp_path / "cwcf1.json"
    candidate = tmp_path / "p34.json"
    common = {"seed": 42, "test_images_accessed": False}
    _write(
        baseline,
        {
            **common,
            "protocol": CWCF1_PROTOCOL,
            "metrics": _metrics(.7098, .3339, .0473),
            "target_class_map50_95": .0473,
        },
    )
    _write(
        candidate,
        {
            **common,
            "protocol": PROTOCOL,
            "metrics": _metrics(.7160, .3420, .0500),
            "target_class_map50_95": .0500,
        },
    )
    result = build_decision(baseline, candidate, tmp_path / "decision.json")
    assert result["decision"] == "PASS_P34_SCREEN"
    assert result["v8n_cwcf_p34_minus_cwcf1"]["macro_map50_95"] == pytest.approx(
        .0062
    )
    assert result["v8n_cwcf_p34_minus_cwcf1"][
        "bottom3_class_map50_95"
    ] == pytest.approx(.0081)
    assert result["target_delta"] == pytest.approx(.0027)
    assert result["test_opened"] is False


def test_protocol_and_notebook_keep_single_change_and_test_lock():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_YOLOV8N_CWCF_P34_PROTOCOL_2026-09-26.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "P3: CWCF" in protocol
    assert "P4: CWCF" in protocol
    assert "P5: native" in protocol
    assert "attribute_gain = 0.15" in protocol
    assert "PASS_P34_SCREEN" in protocol

    notebook = json.loads(
        (
            ROOT / "notebooks/Coffee_Standard_J25_YOLOv8n_CWCF_P34_Seed42_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-yolov8n-cwcf-p34'" in code
    assert "V8N_CWCF1_seed42_result.json" in code
    assert "run_coffee_standard_j25_yolov8n_cwcf_p34" in code
    assert "--authorize-training" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
