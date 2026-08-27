import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from coffee_detector.analysis.af2_quality_alignment import (
    _decision,
    _validate_factorial,
    continuous_ece,
    same_class_iou_quality,
    spearman_correlation,
)


def test_same_class_quality_ignores_better_wrong_class_box() -> None:
    predicted_boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=torch.float32)
    predicted_classes = torch.tensor([1, 0])
    target_boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target_classes = torch.tensor([0])
    quality = same_class_iou_quality(
        predicted_boxes, predicted_classes, target_boxes, target_classes
    )
    assert quality.tolist() == pytest.approx([0.0, 0.25])


def test_alignment_metrics_handle_ties_and_perfect_calibration() -> None:
    assert spearman_correlation(np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.5, 1.0])) == pytest.approx(1.0)
    assert spearman_correlation(np.array([0.1, 0.1]), np.array([0.2, 0.3])) == 0.0
    values = np.asarray([0.1, 0.5, 0.9])
    assert continuous_ece(values, values) == pytest.approx(0.0)


def test_factorial_validation_requires_all_gates(tmp_path: Path) -> None:
    payload = {
        "format": "coffee_detector.af2_box_score_factorial.result.v1",
        "decision": "AF2_BOX_SCORE_INTERACTION_NECESSARY",
        "training_executed": False,
        "test_images_accessed": False,
        "gates": {"a": True, "b": True},
    }
    path = tmp_path / "factorial.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_factorial(path)["decision"] == payload["decision"]
    payload["gates"]["b"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="bukan hasil valid"):
        _validate_factorial(path)


def test_decision_authorizes_only_strong_fixed_candidate_headroom() -> None:
    def row(macro: float, bottom: float, worst: float, rho: float):
        return {
            "oracle_minus_native": {
                "macro_map50_95": macro,
                "bottom3_class_map50_95": bottom,
                "worst_class_map50_95": worst,
            },
            "alignment": {
                "spearman_confidence_quality": rho,
                "continuous_ece": 0.2,
                "quality_brier": 0.1,
            },
        }

    comparison, decision, next_action = _decision(
        {"D0FT": row(0.01, 0.01, 0.01, 0.5), "AF2": row(0.02, 0.03, 0.01, 0.6)}
    )
    assert decision == "QUALITY_ALIGNMENT_HEADROOM_SUPPORTED"
    assert next_action == "AUTHORIZE_MATCHED_AF2_QUALITY_LOSS_SCREEN"
    assert all(comparison["criteria"].values())


def test_notebook_is_validation_only_and_uses_factorial_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks/Faruq_V3_AF2_Quality_Alignment_Audit_Colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
    assert "codex/af2-quality-alignment-audit" in source
    assert "af2_box_score_factorial.json" in source
    assert "D0FT_seed42/weights/best.pt" in source
    assert "AF2_seed42/weights/best.pt" in source
    assert "--factorial-summary" in source
    assert "--authorize-training" not in source
    assert "Test tidak boleh tersedia" in source

