from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from coffee_detector.af2_complement.modules import SpaceFrequencySelectionResidual
from coffee_detector.analysis.af2sfs1_root_cause import (
    _adapter_intervention,
    _delta,
    _selector_records,
    _selector_summary,
    _spearman,
    _summary,
)


def test_paired_summary_separates_localization_and_classification():
    rows = [
        {
            "raw_accessible": True,
            "raw_max_iou": 0.8,
            "final_matched": True,
            "final_iou": 0.7,
            "correct_class": True,
        },
        {
            "raw_accessible": True,
            "raw_max_iou": 0.6,
            "final_matched": True,
            "final_iou": 0.55,
            "correct_class": False,
        },
        {
            "raw_accessible": False,
            "raw_max_iou": 0.3,
            "final_matched": False,
            "final_iou": 0.0,
            "correct_class": False,
        },
    ]
    result = _summary(rows)
    assert result["raw_proposal_accessibility"] == 2 / 3
    assert result["final_matched_recall"] == 2 / 3
    assert result["conditional_top1_accuracy"] == 0.5
    assert result["correct_decision_recall"] == 1 / 3


def test_delta_uses_candidate_minus_control():
    control = {
        "raw_proposal_accessibility": 0.8,
        "mean_raw_max_iou": 0.6,
        "final_matched_recall": 0.7,
        "mean_final_matched_iou": 0.55,
        "conditional_top1_accuracy": 0.6,
        "correct_decision_recall": 0.42,
    }
    candidate = {key: value + 0.1 for key, value in control.items()}
    assert all(value == pytest.approx(0.1) for value in _delta(candidate, control).values())


def test_adapter_interventions_are_reversible_and_distinct():
    torch.manual_seed(4)
    adapter = SpaceFrequencySelectionResidual(4)
    torch.nn.init.constant_(adapter.output.weight, 0.2)
    value = torch.rand(1, 4, 8, 8)
    normal = adapter(value)
    with _adapter_intervention(adapter, "bypass"):
        bypass = adapter(value)
    with _adapter_intervention(adapter, "spatial"):
        spatial = adapter(value)
    with _adapter_intervention(adapter, "frequency"):
        frequency = adapter(value)
    restored = adapter(value)
    assert torch.equal(bypass, value)
    assert torch.equal(restored, normal)
    assert not torch.equal(spatial, frequency)


def test_selector_observability_is_finite_and_normalized():
    adapter = SpaceFrequencySelectionResidual(4)
    torch.nn.init.constant_(adapter.output.weight, 0.1)
    value = torch.rand(1, 4, 8, 8)
    captured = {"input": value, "output": adapter(value)}
    rows = _selector_records(
        adapter,
        captured,
        torch.tensor([[0.0, 0.0, 64.0, 64.0]]),
        image_size=64,
    )
    assert len(rows) == 1
    assert rows[0]["spatial_weight"] + rows[0]["frequency_weight"] == pytest.approx(1.0)
    result = _selector_summary(rows)
    assert result["objects"] == 1
    assert result["residual_input_ratio"]["mean"] > 0


def test_spearman_handles_monotonic_and_constant_inputs():
    assert _spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert _spearman([1, 1, 1], [10, 20, 30]) is None


def test_protocol_and_notebook_remain_validation_only():
    protocol = Path("docs/FARUQ_V3_AF2SFS1_ROOT_CAUSE_PROTOCOL_2026-08-28.md").read_text(
        encoding="utf-8"
    )
    notebook_path = Path("notebooks/Faruq_V3_AF2SFS1_Root_Cause_Colab.ipynb")
    notebook = notebook_path.read_text(encoding="utf-8")
    payload = json.loads(notebook)
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(notebook_path), "exec")
    assert "Training: **forbidden**" in protocol
    assert "Test: **closed**" in protocol
    assert "--authorize-training" not in notebook
    assert "--authorize-test" not in notebook
    assert "test/images" not in notebook
