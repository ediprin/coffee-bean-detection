import json
from pathlib import Path

import numpy as np

from coffee_detector.analysis.defectoscafeverde_dvf1_review import (
    _mcnemar_exact,
    stratified_pair_bootstrap,
)


def test_mcnemar_exact_for_five_rescues_and_one_regression():
    assert _mcnemar_exact(5, 1) == 0.21875
    assert _mcnemar_exact(0, 0) == 1.0


def test_stratified_bootstrap_is_deterministic_and_keeps_classes():
    labels = np.repeat(np.arange(12), 8)
    paper = np.ones(len(labels), dtype=bool)
    candidate = paper.copy()
    paper[np.arange(12) * 8] = False
    result_a = stratified_pair_bootstrap(
        paper, candidate, labels, iterations=100, seed=17
    )
    result_b = stratified_pair_bootstrap(
        paper, candidate, labels, iterations=100, seed=17
    )

    assert result_a == result_b
    assert result_a["stratified_by_class"] is True
    assert result_a["metrics"]["physical_pair_accuracy"]["point_delta"] > 0
    assert result_a["metrics"]["macro_class_accuracy"]["probability_positive"] > 0


def test_review_protocol_is_frozen_and_forbids_training_and_test():
    text = Path(
        "docs/DEFECTOSCAFEVERDE_DVF1_REVIEW_PROTOCOL_2026-10-06.md"
    ).read_text(encoding="utf-8")
    assert "before" in text
    assert "reading per-pair transitions or running the bootstrap" in text
    assert "10,000 class-stratified physical-pair bootstrap" in text
    assert "No model, detector, fuser, threshold, or cache is trained or changed" in text
    assert "Test is not extracted or accessed" in text


def test_review_colab_compiles_and_has_no_training_command():
    notebook = json.loads(
        Path("notebooks/DefectosCafeVerde_DVF1_Review_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(source, "dvf1_review_colab", "exec")
    assert "defectoscafeverde_dvf1_review" in source
    assert "--authorize-training" not in source
    assert "TRAINING: False" in source
