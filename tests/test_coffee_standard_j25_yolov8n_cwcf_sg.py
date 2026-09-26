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
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf_sg import (
    PROTOCOL,
    _non_gate_state,
    _state_mapping_equal,
    build_decision,
)
from coffee_detector.j25_cwcf import CWCFConfig, build_cwcf_model


ROOT = Path(__file__).resolve().parents[1]


def test_sg1_changes_only_scale_gate_switch():
    candidate = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF_SG1.yaml").read_text(
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

    sg = CWCFConfig.from_mapping(candidate["cwcf"]).to_dict()
    base = CWCFConfig.from_mapping(baseline["cwcf"]).to_dict()
    assert sg.pop("learnable_scale_gates") is True
    assert base.pop("learnable_scale_gates") is False
    assert sg == base


def test_scale_gate_initialization_adds_only_three_parameters_and_gain_one():
    model_yaml = str(ROOT / "configs/coffee_fg/models/yolov8n-p3.yaml")
    baseline = build_cwcf_model(
        model_yaml,
        nc=25,
        source=None,
        seed=42,
        config=CWCFConfig(),
        verbose=False,
    )
    candidate = build_cwcf_model(
        model_yaml,
        nc=25,
        source=None,
        seed=42,
        config=CWCFConfig(learnable_scale_gates=True),
        verbose=False,
    )
    assert _state_mapping_equal(
        _non_gate_state(candidate), _non_gate_state(baseline)
    )
    assert sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    ) == 3
    head = candidate.model[-1]
    assert len(head.scale_gate_logits) == 3
    for parameter in head.scale_gate_logits:
        assert parameter.item() == 0.0
        assert (2.0 * torch.sigmoid(parameter)).item() == 1.0


def test_scale_gate_one_matches_cwcf1_with_active_residual():
    model_yaml = str(ROOT / "configs/coffee_fg/models/yolov8n-p3.yaml")
    baseline = build_cwcf_model(
        model_yaml,
        nc=25,
        source=None,
        seed=42,
        config=CWCFConfig(),
        verbose=False,
    ).train()
    candidate = build_cwcf_model(
        model_yaml,
        nc=25,
        source=None,
        seed=42,
        config=CWCFConfig(learnable_scale_gates=True),
        verbose=False,
    ).train()

    with torch.no_grad():
        for index, (left, right) in enumerate(
            zip(candidate.model[-1].adapters, baseline.model[-1].adapters)
        ):
            value = 0.1 * (index + 1)
            left.affine.bias[0] = value
            right.affine.bias[0] = value

    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        sg = candidate(image)
        cwcf = baseline(image)
    assert torch.equal(sg["boxes"], cwcf["boxes"])
    assert torch.equal(sg["scores"], cwcf["scores"])


def test_scale_gate_changes_scores_not_boxes_after_residual_is_active():
    model_yaml = str(ROOT / "configs/coffee_fg/models/yolov8n-p3.yaml")
    candidate = build_cwcf_model(
        model_yaml,
        nc=25,
        source=None,
        seed=42,
        config=CWCFConfig(learnable_scale_gates=True),
        verbose=False,
    ).train()
    image = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        candidate.model[-1].adapters[2].affine.bias[0] = 0.25
        before = candidate(image)
        candidate.model[-1].scale_gate_logits[2].fill_(-2.0)
        after = candidate(image)
    assert torch.equal(before["boxes"], after["boxes"])
    assert not torch.equal(before["scores"], after["scores"])


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_sg1_decision_compares_against_cwcf1_and_reports_gates(tmp_path):
    baseline = tmp_path / "cwcf1.json"
    candidate = tmp_path / "sg1.json"
    common = {"seed": 42, "test_images_accessed": False}
    learned = {
        "logits": {"P3": 0.1, "P4": -0.2, "P5": -0.4},
        "gains": {"P3": 1.05, "P4": 0.90, "P5": 0.80},
    }
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
            "learned_scale_gates": learned,
        },
    )
    result = build_decision(baseline, candidate, tmp_path / "decision.json")
    assert result["decision"] == "PASS_SG1_SCREEN"
    assert result["v8n_cwcf_sg1_minus_cwcf1"]["macro_map50_95"] == pytest.approx(
        .0062
    )
    assert result["v8n_cwcf_sg1_minus_cwcf1"][
        "bottom3_class_map50_95"
    ] == pytest.approx(.0081)
    assert result["target_delta"] == pytest.approx(.0027)
    assert result["learned_scale_gates"] == learned
    assert result["test_opened"] is False


def test_protocol_and_notebook_keep_single_change_and_test_lock():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_YOLOV8N_CWCF_SG1_PROTOCOL_2026-09-26.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "g_l = 2 * sigmoid(alpha_l)" in protocol
    assert "exactly three new scalar parameters" in protocol
    assert "attribute_gain = 0.15" in protocol
    assert "PASS_SG1_SCREEN" in protocol

    notebook = json.loads(
        (
            ROOT / "notebooks/Coffee_Standard_J25_YOLOv8n_CWCF_SG1_Seed42_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-yolov8n-cwcf-sg1'" in code
    assert "V8N_CWCF1_seed42_result.json" in code
    assert "run_coffee_standard_j25_yolov8n_cwcf_sg" in code
    assert "LEARNED SCALE GATES:" in code
    assert "--authorize-training" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
