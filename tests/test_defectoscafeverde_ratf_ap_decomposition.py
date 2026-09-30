import json
from pathlib import Path

import numpy as np
import pytest
import torch

from coffee_detector.analysis.defectoscafeverde_ratf_ap_decomposition import (
    MODEL_ORDER,
    _greedy_iou_pairs,
    _update_target_diagnostic,
    attribute_ap_regression,
    summarize_target_diagnostic,
    target_ap_summary,
)


def _target_ap(ap50, ap75, mean_ap):
    values = {
        "0.50": ap50,
        "0.55": mean_ap,
        "0.60": mean_ap,
        "0.65": mean_ap,
        "0.70": mean_ap,
        "0.75": ap75,
        "0.80": ap75,
        "0.85": ap75,
        "0.90": ap75,
        "0.95": ap75,
    }
    return {
        "ap50": ap50,
        "ap75": ap75,
        "ap50_95": mean_ap,
        "ap_by_iou": values,
    }


def test_greedy_iou_pairs_are_one_to_one_and_iou_prioritized():
    predictions = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]]
    )
    targets = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    pairs = _greedy_iou_pairs(predictions, targets, 0.5)
    assert {(prediction, target) for prediction, target, _ in pairs} == {(0, 0), (2, 1)}
    assert all(iou == 1.0 for _, _, iou in pairs)


def test_target_diagnostic_counts_duplicate_confusion_and_background_fp():
    totals = {
        "targets": 0,
        "predictions": 0,
        "tp50": 0,
        "tp75": 0,
        "matched_iou50": [],
        "matched_iou75": [],
        "tp_confidence50": [],
        "fp_confidence50": [],
        "fp_categories50": __import__("collections").Counter(),
    }
    # One target partido (id 9), one other-class target, and four partido predictions:
    # TP, duplicate, overlap-other-class, background.
    target_boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    )
    target_classes = torch.tensor([9, 2])
    prediction = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0, 0.9, 9.0],
            [1.0, 1.0, 9.0, 9.0, 0.8, 9.0],
            [20.0, 20.0, 30.0, 30.0, 0.7, 9.0],
            [40.0, 40.0, 50.0, 50.0, 0.6, 9.0],
        ]
    )
    _update_target_diagnostic(totals, prediction, target_boxes, target_classes)
    result = summarize_target_diagnostic(totals)
    assert result["targets"] == 1
    assert result["tp50"] == 1
    assert result["fp50"] == 3
    assert result["fp_categories50"] == {
        "duplicate_target": 1,
        "overlaps_other_class": 1,
        "background_or_localization": 1,
    }


def test_attribution_separates_residual_recovery_from_high_iou_regression():
    endpoints = {
        "D0DIRECT": {"target_ap": _target_ap(0.90, 0.80, 0.82)},
        "RATF1_ACTIVE": {"target_ap": _target_ap(0.90, 0.70, 0.76)},
        "RATF1_ZERO_RESIDUAL": {"target_ap": _target_ap(0.90, 0.80, 0.82)},
    }
    result = attribute_ap_regression(endpoints)
    assert result["attribution"] == "INFERENCE_RESIDUAL_DOMINANT"
    assert result["components"]["zero_residual_recovers_d0"] is True

    endpoints["RATF1_ZERO_RESIDUAL"] = {
        "target_ap": _target_ap(0.90, 0.70, 0.76)
    }
    result = attribute_ap_regression(endpoints)
    assert result["attribution"] == "HIGH_IOU_LOCALIZATION_OR_RANKING_DOMINANT"


def test_target_ap_summary_exposes_all_ten_iou_thresholds():
    true_positives = np.ones((1, 10), dtype=bool)
    result = target_ap_summary(
        true_positives,
        np.array([0.9]),
        np.array([9.0]),
        np.array([9.0]),
    )
    assert tuple(result["ap_by_iou"]) == (
        "0.50",
        "0.55",
        "0.60",
        "0.65",
        "0.70",
        "0.75",
        "0.80",
        "0.85",
        "0.90",
        "0.95",
    )
    assert result["ap50"] == pytest.approx(result["ap75"])
    assert result["ap50"] == pytest.approx(result["ap50_95"])


def test_notebook_appends_validation_only_ap_decomposition_cell():
    notebook = Path("notebooks/DefectosCafeVerde_RATF1_Seed42_Colab.ipynb")
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "defectoscafeverde_ratf_ap_decomposition" in text
    assert "PARTIDO_AP_DECOMPOSITION" in text
    assert "--grouped-audit" in text
    assert "if mydrive_project.is_dir(): PROJECT=mydrive_project.resolve()" in text
    assert "len(project_candidates)!=1" not in text
    assert tuple(MODEL_ORDER) == (
        "D0DIRECT",
        "RATF1_ACTIVE",
        "RATF1_ZERO_RESIDUAL",
    )
