from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coffee_detector.analysis.af2sfs1_map_intervention import (
    _drift_summary,
    _group_name,
    _metric_payload,
    decompose_metrics,
)


class _FakeBox:
    ap_class_index = np.asarray([0, 1])
    all_ap = np.asarray(
        [
            [0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72],
            [0.70, 0.68, 0.66, 0.64, 0.62, 0.60, 0.58, 0.56, 0.54, 0.52],
        ]
    )

    @property
    def ap(self):
        return self.all_ap.mean(axis=1)


def test_metric_payload_preserves_per_iou_class_ap():
    metrics = SimpleNamespace(
        results_dict={"metrics/mAP50-95(B)": 0.7},
        box=_FakeBox(),
    )
    result = _metric_payload(metrics, {0: "a", 1: "b"})
    assert result["macro_map50_95"] == pytest.approx(0.71)
    assert result["macro_ap50"] == pytest.approx(0.8)
    assert result["macro_ap75"] == pytest.approx(0.7)
    assert result["ap_by_class_and_iou"]["a"]["map50_95"] == pytest.approx(0.81)


def test_map_decomposition_is_exact():
    control = {"macro": 0.88, "bottom": 0.80}
    bypass = {"macro": 0.887, "bottom": 0.81}
    normal = {"macro": 0.89, "bottom": 0.812}
    result = decompose_metrics(control, normal, bypass)
    assert result["total_normal_minus_control"]["macro"] == pytest.approx(0.01)
    assert result["direct_selector_normal_minus_bypass"]["macro"] == pytest.approx(0.003)
    assert result["optimization_mediated_bypass_minus_control"]["macro"] == pytest.approx(0.007)
    assert result["additivity_error"]["macro"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.2.conv.weight", "feature_extractor"),
        ("model.23.base_head.cv2.0.weight", "regression_head"),
        ("model.23.base_head.cv3.0.weight", "classification_head"),
        ("model.23.adapter.selector.weight", "sfs_adapter"),
        ("model.23.base_head.dfl.conv.weight", "other_detector_state"),
    ],
)
def test_checkpoint_grouping(name, expected):
    assert _group_name(name, head_index=23) == expected


def test_drift_summary_reports_relative_change():
    left = torch.tensor([1.0, 2.0])
    right = torch.tensor([2.0, 2.0])
    result = _drift_summary([(left, right)])
    assert result["parameters"] == 2
    assert result["l2_delta"] == pytest.approx(1.0)
    assert result["mean_absolute_delta"] == pytest.approx(0.5)


def test_protocol_and_notebook_are_validation_only():
    protocol = Path("docs/FARUQ_V3_AF2SFS1_MAP_INTERVENTION_PROTOCOL_2026-08-28.md").read_text(
        encoding="utf-8"
    )
    notebook_path = Path("notebooks/Faruq_V3_AF2SFS1_Map_Intervention_Colab.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(notebook_path), "exec")
    assert "VALIDATION-ONLY, NO TRAINING" in protocol
    assert "--authorize-training" not in source
    assert "--authorize-test" not in source
    assert "test/images" not in source
