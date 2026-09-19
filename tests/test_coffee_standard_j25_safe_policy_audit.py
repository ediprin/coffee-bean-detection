import ast
import json
from pathlib import Path

from coffee_detector.analysis.coffee_standard_j25_safe_policy_audit import (
    prioritize_ablation,
)


ROOT = Path(__file__).resolve().parents[1]


def _view(correct):
    stage = {
        "targets": 6,
        "accessible": 6,
        "matched": 6,
        "correct_class": correct,
    }
    return {"raw_top500": stage, "final_conf0001": stage}


def test_priority_removes_sampler_first_when_exposure_shift_and_raw_degradation_coexist():
    result = prioritize_ablation(
        {"D0DIRECT": _view(2), "SAFED0": _view(0)},
        {"sampler_exposure_multiplier": 1.4},
    )
    assert result["priority_arm"] == "SAFE_AUGMENTATION_ONLY_NO_SAMPLER"
    assert result["raw_correct_degraded_vs_d0"] is True


def test_priority_tests_augmentation_when_sampler_barely_changes_exposure():
    result = prioritize_ablation(
        {"D0DIRECT": _view(2), "SAFED0": _view(0)},
        {"sampler_exposure_multiplier": 1.01},
    )
    assert result["priority_arm"] == "IDENTITY_SAMPLER_ONLY_STANDARD_AUGMENTATION"


def test_protocol_and_notebook_are_diagnostic_only():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_SAFE_POLICY_ROOT_CAUSE_PROTOCOL_2026-09-19.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before diagnostic evaluation**" in protocol
    assert "no training is authorized" in protocol

    notebook = json.loads(
        (ROOT / "notebooks/Coffee_Standard_J25_SAFE_Policy_Audit_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-safe-policy-root-cause'" in code
    assert "coffee_standard_j25_safe_policy_audit" in code
    assert "--authorize-diagnostic" in code
    assert "--authorize-training" not in code
    assert "--authorize-test" not in code
    assert "test/images" not in code

