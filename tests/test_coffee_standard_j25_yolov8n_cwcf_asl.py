import ast
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import METRICS
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf import (
    PROTOCOL as CWCF1_PROTOCOL,
)
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf_asl import (
    ASL_KEYS,
    PROTOCOL,
    _cwcf_mechanism_mapping,
    build_decision,
)
from coffee_detector.j25_cwcf import CWCFConfig, attribute_asymmetric_loss


ROOT = Path(__file__).resolve().parents[1]


def test_asl1_config_changes_only_attribute_loss_contract():
    candidate = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF_ASL1.yaml").read_text(encoding="utf-8")
    )
    baseline = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF1.yaml").read_text(encoding="utf-8")
    )
    assert candidate["dataset"] == baseline["dataset"]
    assert candidate["model"] == baseline["model"]
    assert candidate["weights"] == baseline["weights"]
    assert candidate["sampler"] == baseline["sampler"] == "none"
    assert candidate["train"] == baseline["train"]
    assert _cwcf_mechanism_mapping(candidate["cwcf"]) == baseline["cwcf"]
    assert set(candidate["cwcf"]) - set(baseline["cwcf"]) == ASL_KEYS
    frozen = CWCFConfig.from_mapping(candidate["cwcf"])
    assert frozen.attribute_loss == "asl"
    assert frozen.asl_gamma_pos == 0.0
    assert frozen.asl_gamma_neg == 4.0
    assert frozen.asl_clip == 0.05
    assert frozen.asl_detach_focal_weight is True


def test_original_cwcf_config_remains_bce_by_default():
    original = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/V8N_CWCF1.yaml").read_text(encoding="utf-8")
    )
    frozen = CWCFConfig.from_mapping(original["cwcf"])
    assert frozen.attribute_loss == "bce"


def test_asl_reduces_to_bce_without_asymmetry():
    logits = torch.tensor([[0.2, -0.7, 1.3], [-1.1, 0.0, 0.8]], dtype=torch.float64)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    actual = attribute_asymmetric_loss(
        logits,
        targets,
        gamma_pos=0.0,
        gamma_neg=0.0,
        clip=0.0,
    )
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_asl_matches_official_formula_for_frozen_parameters():
    logits = torch.tensor([[0.3, -1.2, 2.0], [-0.4, 0.7, -2.2]], dtype=torch.float64)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    p = torch.sigmoid(logits)
    pos = p
    neg = (1.0 - p + 0.05).clamp(max=1.0)
    log_likelihood = targets * torch.log(pos) + (1.0 - targets) * torch.log(neg)
    pt = pos * targets + neg * (1.0 - targets)
    gamma = 0.0 * targets + 4.0 * (1.0 - targets)
    expected = -(log_likelihood * torch.pow((1.0 - pt).detach(), gamma))
    actual = attribute_asymmetric_loss(
        logits,
        targets,
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.05,
        detach_focal_weight=True,
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_asl_decision_compares_against_cwcf1(tmp_path):
    baseline = tmp_path / "cwcf1.json"
    candidate = tmp_path / "asl1.json"
    common = {"seed": 42, "test_images_accessed": False}
    _write(
        baseline,
        {
            **common,
            "protocol": CWCF1_PROTOCOL,
            "metrics": _metrics(.709, .334, .047),
            "target_class_map50_95": .047,
        },
    )
    _write(
        candidate,
        {
            **common,
            "protocol": PROTOCOL,
            "metrics": _metrics(.715, .341, .046),
            "target_class_map50_95": .046,
        },
    )
    result = build_decision(baseline, candidate, tmp_path / "decision.json")
    assert result["decision"] == "PASS_ASL_SCREEN"
    assert result["v8n_cwcf_asl1_minus_cwcf1"]["macro_map50_95"] == pytest.approx(.006)
    assert result["v8n_cwcf_asl1_minus_cwcf1"]["bottom3_class_map50_95"] == pytest.approx(.007)
    assert result["criteria"]["worst_class_is_descriptive_not_a_promotion_gate"] is True
    assert result["target_delta"] == pytest.approx(-.001)
    assert result["test_opened"] is False


def test_protocol_and_notebook_keep_single_change_and_test_lock():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_YOLOV8N_CWCF_ASL1_PROTOCOL_2026-09-20.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "attribute BCE  ->  attribute ASL" in protocol
    assert "gamma_pos = 0" in protocol
    assert "gamma_neg = 4" in protocol
    assert "m = 0.05" in protocol

    notebook = json.loads(
        (
            ROOT / "notebooks/Coffee_Standard_J25_YOLOv8n_CWCF_ASL1_Seed42_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-yolov8n-cwcf-asl1'" in code
    assert "V8N_CWCF1_seed42_result.json" in code
    assert "run_coffee_standard_j25_yolov8n_cwcf_asl" in code
    assert "--authorize-training" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
