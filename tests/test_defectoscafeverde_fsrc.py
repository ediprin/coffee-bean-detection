import json
from pathlib import Path

import pytest
import torch
import yaml

from coffee_detector.defectos_fsrc import (
    FSRCConfig,
    SpectralReliabilityCalibrator,
    apply_reliability_suppression,
    candidate_features,
    stationary_haar_energy,
)
from coffee_detector.experiments.run_defectoscafeverde_fsrc import (
    _balanced_weights,
    _decision,
    run_fsrc_static_audit,
    train_calibrator,
)


def _predictions():
    return [
        torch.tensor(
            [
                [2.0, 2.0, 20.0, 20.0, 0.9, 9.0],
                [3.0, 3.0, 19.0, 19.0, 0.7, 9.0],
            ]
        ),
        torch.tensor([[20.0, 20.0, 50.0, 50.0, 0.8, 3.0]]),
    ]


def test_stationary_haar_is_parameter_free_finite_and_spatially_aligned():
    image = torch.rand(2, 3, 64, 80, requires_grad=True)
    cue = stationary_haar_energy(image)
    assert cue.shape == (2, 6, 64, 80)
    assert torch.isfinite(cue).all()
    assert cue.grad_fn is not None
    assert list(stationary_haar_energy.__annotations__)  # fixed function, not a module


def test_candidate_features_are_fixed_dimension_and_capture_duplicate_context():
    config = FSRCConfig()
    features, image_indices = candidate_features(
        torch.rand(2, 3, 64, 64), _predictions(), config
    )
    assert features.shape == (3, config.feature_dimension)
    assert image_indices.tolist() == [0, 0, 1]
    # Context begins after 18 spectral summaries. Candidate two overlaps a
    # higher-confidence prediction of the same class; candidate one does not.
    duplicate_overlap_column = 18 + 6
    assert features[0, duplicate_overlap_column] == 0
    assert features[1, duplicate_overlap_column] > 0.5


def test_calibrator_only_suppresses_scores_and_has_exact_off_switch():
    config = FSRCConfig()
    predictions = _predictions()
    features, _ = candidate_features(torch.rand(2, 3, 64, 64), predictions, config)
    calibrator = SpectralReliabilityCalibrator(config)
    reliability = calibrator.reliability(features)
    active = apply_reliability_suppression(predictions, reliability, config)
    disabled = apply_reliability_suppression(
        predictions, reliability, config, enabled=False
    )
    for source, candidate, control in zip(predictions, active, disabled):
        assert torch.equal(source, control)
        assert torch.equal(source[:, :4], candidate[:, :4])
        assert torch.equal(source[:, 5], candidate[:, 5])
        assert torch.all(candidate[:, 4] <= source[:, 4])
        assert torch.all(candidate[:, 4] >= source[:, 4] * config.suppression_floor)


def test_balanced_weights_equalize_each_observed_class_label_group():
    labels = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    classes = torch.tensor([0, 0, 0, 1, 1, 1])
    weights = _balanced_weights(labels, classes)
    group_totals = []
    for class_id in (0, 1):
        for label in (0.0, 1.0):
            mask = (classes == class_id) & (labels == label)
            group_totals.append(float(weights[mask].sum()))
    assert max(group_totals) == pytest.approx(min(group_totals))


def test_static_audit_passes_and_parameter_budget_is_small(tmp_path: Path):
    result = run_fsrc_static_audit(tmp_path / "static.json")
    assert result["decision"] == "PASS"
    assert all(result["gates"].values())
    assert 0 < result["added_parameters"] < 2000


def test_decision_preserves_numeric_tradeoff_instead_of_hiding_it():
    control = {
        "macro_map50_95": 0.90,
        "bottom3_class_map50_95": 0.84,
        "worst_class_map50_95": 0.83,
    }
    candidate = {
        "macro_map50_95": 0.903,
        "bottom3_class_map50_95": 0.841,
        "worst_class_map50_95": 0.831,
    }
    comparison, decision, next_action = _decision(
        {"D0DIRECT": control, "FSRC1": candidate}
    )
    assert decision == "PASS"
    assert next_action == "REVIEW_BEFORE_MULTISEED"
    assert comparison["deltas"]["macro_map50_95"] == pytest.approx(0.003)


def test_calibrator_training_checkpoint_is_resumable(tmp_path: Path):
    config = FSRCConfig()
    cache = {
        "features": torch.randn(32, config.feature_dimension),
        "score": torch.linspace(0.05, 0.95, 32),
        "label": torch.tensor(([0.0, 1.0] * 16)),
        "class_id": torch.tensor(([0, 0, 1, 1] * 8)),
    }
    training = {
        "epochs": 1,
        "batch": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "ranking_weight": 1.0,
        "preservation_weight": 0.25,
        "seed": 42,
    }
    checkpoint = tmp_path / "calibrator.pt"
    first, history = train_calibrator(
        cache, config, training, checkpoint, device=torch.device("cpu")
    )
    second, restored_history = train_calibrator(
        cache, config, training, checkpoint, device=torch.device("cpu")
    )
    assert history == restored_history
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.equal(left, right)


def test_protocol_config_and_notebook_freeze_safe_frozen_d0_screen():
    protocol = Path("docs/DEFECTOSCAFEVERDE_FSRC1_PROTOCOL_2026-09-30.md").read_text(
        encoding="utf-8"
    )
    assert "Status: frozen before training" in protocol
    assert "cannot create a box, move a box, change a class" in protocol
    assert "No additional seed or test evaluation is authorized" in protocol

    config = yaml.safe_load(
        Path("configs/defectoscafeverde/FSRC1.yaml").read_text(encoding="utf-8")
    )
    assert config["code"] == "FSRC1"
    assert config["fsrc"]["levels"] == 2
    assert config["fsrc"]["suppression_floor"] == 0.25
    assert config["train"]["epochs"] == 20

    path = Path("notebooks/DefectosCafeVerde_FSRC1_Seed42_Colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), str(path), "exec")
    assert "BRANCH='codex/defectoscafeverde-fsrc'" in source
    assert "run_defectoscafeverde_fsrc" in source
    assert "D0DIRECT_seed42/weights/best.pt" in source
    assert "TEST TEREXPOSE" in source
    assert "--authorize-training" in source
    assert "detector tetap frozen" in source
