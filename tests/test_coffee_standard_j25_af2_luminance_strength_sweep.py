import ast
import json
from pathlib import Path

import pytest
import torch

from coffee_detector.af2_luminance import AF2LuminanceStochasticInputEnhancer
from coffee_detector.afab import AFABConfig
from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_strength_sweep import (
    TARGET_CLASS,
    summarize_sweep,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _config():
    return AFABConfig(mode="af2", patch_size=32, overlap=0.5, chunk_size=8)


def test_eval_strength_is_explicit_bounded_and_checkpoint_backward_compatible():
    torch.manual_seed(91)
    value = 0.1 + 0.7 * torch.rand(2, 3, 64, 64)
    enhancer = AF2LuminanceStochasticInputEnhancer(_config()).eval()
    enhancer.set_inference_strength(0.25)
    assert torch.equal(enhancer(value), enhancer.forward_with_strength(value, 0.25))
    with pytest.raises(ValueError):
        enhancer.set_inference_strength(1.01)
    del enhancer.inference_strength
    assert torch.equal(enhancer(value), enhancer.forward_with_strength(value, 1.0))


def _row(macro, bottom, worst, target):
    return {
        "macro_map50_95": macro,
        "bottom3_class_map50_95": bottom,
        "worst_class_map50_95": worst,
        "map50_95_by_class": {TARGET_CLASS: target},
    }


def test_summary_preserves_endpoint_and_reports_tradeoff_frontier():
    rows = {
        "0p00": _row(0.58, 0.18, 0.01, 0.01),
        "0p25": _row(0.61, 0.21, 0.02, 0.02),
        "0p50": _row(0.63, 0.23, 0.02, 0.02),
        "0p75": _row(0.65, 0.24, 0.02, 0.02),
        "1p00": _row(0.66, 0.25, 0.00, 0.00),
    }
    endpoint = {metric: rows["1p00"][metric] for metric in METRICS}
    result = summarize_sweep(rows, endpoint)
    assert result["endpoint_exact"] is True
    assert result["pareto_frontier"] == ["0p75", "1p00"]
    assert result["target_class_rescued_strengths"] == ["0p00", "0p25", "0p50", "0p75"]
    assert result["interpretation"] == "INFERENCE_STRENGTH_CAN_RESCUE_TARGET"


def test_summary_rejects_nonreproduced_historical_endpoint():
    rows = {code: _row(0.5, 0.4, 0.3, 0.2) for code in ("0p00", "0p25", "0p50", "0p75", "1p00")}
    endpoint = {metric: 0.0 for metric in METRICS}
    with pytest.raises(RuntimeError, match="tidak mereproduksi"):
        summarize_sweep(rows, endpoint)


def test_protocol_and_notebook_lock_validation_only_diagnostic():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_AF2_LUMINANCE_STRENGTH_SWEEP_PROTOCOL_2026-09-19.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before diagnostic evaluation**" in protocol
    assert "no optimization step and no gradient update" in protocol
    assert "no locked-test extraction or model evaluation" in protocol

    path = ROOT / "notebooks/Coffee_Standard_J25_AF2LUMSAFE_Strength_Sweep_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/af2-luminance-strength-sweep'" in code
    assert "--authorize-diagnostic" in code
    assert "run_coffee_standard_j25_af2_luminance_strength_sweep" in code
    assert "--authorize-training" not in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
