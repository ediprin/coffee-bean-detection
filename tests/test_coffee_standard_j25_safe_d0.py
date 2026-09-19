import ast
import json
from pathlib import Path

import pytest
import yaml

from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_strength_sweep import (
    PROTOCOL as STRENGTH_PROTOCOL,
)
from coffee_detector.experiments.run_coffee_standard_j25_safe_d0 import (
    DIRECT_PROTOCOL,
    METRICS,
    PROTOCOL,
    TARGET_CLASS,
    build_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def test_safe_d0_config_exactly_matches_safe_policy_without_frontend():
    control = yaml.safe_load((ROOT / "configs/coffee_standard_j25/SAFED0.yaml").read_text())
    treatment = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/AF2LUMSAFE.yaml").read_text()
    )
    assert control["model"] == treatment["model"]
    assert control["train"] == treatment["train"]
    assert control["sampler"] == treatment["sampler"]
    assert control["frontend"] == "none"
    assert "afab" not in control


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(macro, bottom, worst):
    return dict(zip(METRICS, (macro, bottom, worst)))


def test_decision_separates_policy_and_stochastic_af2_effects(tmp_path):
    direct = tmp_path / "direct.json"
    strength = tmp_path / "strength.json"
    control = tmp_path / "control.json"
    _write(
        direct,
        {
            "protocol": DIRECT_PROTOCOL,
            "metrics": _metrics(0.60, 0.20, 0.02),
            "test_images_accessed": False,
        },
    )
    _write(
        strength,
        {
            "protocol": STRENGTH_PROTOCOL,
            "values": {
                "0p00": {
                    **_metrics(0.66, 0.28, 0.00),
                    "target_class_map50_95": 0.0,
                }
            },
            "test_opened": False,
        },
    )
    _write(
        control,
        {
            "protocol": PROTOCOL,
            "metrics": _metrics(0.67, 0.29, 0.03),
            "target_class_map50_95": 0.03,
            "test_images_accessed": False,
        },
    )
    result = build_decision(direct, strength, control, tmp_path / "decision.json")
    assert result["decision"] == "SAFE_POLICY_DOMINATES_STOCHASTIC_AF2"
    assert result["safe_policy_effect_vs_d0"]["macro_map50_95"] == pytest.approx(0.07)
    assert result["target_recovered_without_af2"] is True
    assert result["target_class"] == TARGET_CLASS
    assert result["test_opened"] is False


def test_protocol_and_colab_freeze_single_fresh_control():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_SAFE_D0_CONTROL_PROTOCOL_2026-09-19.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before training**" in protocol
    assert "same official `yolo26n.pt`" in protocol
    assert "never applies AF2" in protocol

    notebook = json.loads(
        (ROOT / "notebooks/Coffee_Standard_J25_SAFED0_Seed42_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-safe-d0-control'" in code
    assert "run_coffee_standard_j25_safe_d0" in code
    assert "--authorize-training" in code
    assert "last.pt tersimpan di Drive setiap epoch" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
