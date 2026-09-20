import ast
import json
from pathlib import Path

from coffee_detector.analysis.coffee_standard_j25_cwcf2_gate_audit import (
    classify_gate_attribution,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gate_attribution_distinguishes_inference_harm():
    active = {
        "macro_map50_95": 0.62,
        "bottom3_class_map50_95": 0.22,
        "worst_class_map50_95": 0.0,
    }
    zero = {
        "macro_map50_95": 0.63,
        "bottom3_class_map50_95": 0.24,
        "worst_class_map50_95": 0.01,
    }
    cwcf1 = {
        "macro_map50_95": 0.64,
        "bottom3_class_map50_95": 0.25,
        "worst_class_map50_95": 0.015,
    }
    assert classify_gate_attribution(active, zero, cwcf1, [0.1, 0.1, 0.1]) == (
        "INFERENCE_COMPOSITION_HARMFUL"
    )


def test_gate_attribution_detects_negligible_gate():
    values = {
        "macro_map50_95": 0.62,
        "bottom3_class_map50_95": 0.22,
        "worst_class_map50_95": 0.0,
    }
    assert classify_gate_attribution(values, values, values, [0.0, 0.0, 0.0]) == (
        "TRAINING_PATH_DOMINANT_GATE_EFFECT_NEGLIGIBLE"
    )


def test_protocol_and_notebook_are_validation_only():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_CWCF2_GATE_AUDIT_2026-09-20.md"
    ).read_text(encoding="utf-8")
    assert "validation-only diagnostic frozen before execution" in protocol
    assert "test split remains closed" in protocol
    notebook = json.loads(
        (
            ROOT / "notebooks/Coffee_Standard_J25_CWCF2_Gate_Audit_Colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "coffee_standard_j25_cwcf2_gate_audit" in code
    assert "--authorize-diagnostic" in code
    assert "--authorize-training" not in code
    assert "test/images" not in code and "split','test" not in code
