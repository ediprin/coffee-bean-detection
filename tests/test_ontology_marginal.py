from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from coffee_detector.ontology_marginal import (
    OntologyDetectionModel,
    OntologyMarginalConfig,
    OntologyMarginalizer,
)
from coffee_detector.train import load_experiment


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_semantic_marginal_rewards_correct_group_when_leaf_is_wrong() -> None:
    # Full black (5) and partial black (7) share primary_condition=black.
    logits = torch.full((1, 21), -8.0)
    logits[0, 7] = 8.0
    labels = torch.tensor([5])
    semantic = OntologyMarginalizer(
        {"mode": "semantic", "tasks": ["primary_condition"], "task_weights": [1.0]}
    )
    control = OntologyMarginalizer(
        {
            "mode": "identity_control",
            "tasks": ["primary_condition"],
            "task_weights": [1.0],
        }
    )

    semantic_loss, _ = semantic(logits, labels)
    control_loss, _ = control(logits, labels)

    assert semantic_loss < 1e-4
    assert control_loss > 10.0


def test_identity_control_is_leaf_ce_on_identical_task_mask() -> None:
    torch.manual_seed(3)
    logits = torch.randn(4, 21)
    labels = torch.tensor([5, 7, 3, 9])
    control = OntologyMarginalizer(
        {
            "mode": "identity_control",
            "tasks": ["surface_extent"],
            "task_weights": [1.0],
        }
    )
    loss, details = control(logits, labels)
    assert torch.allclose(loss, F.cross_entropy(logits, labels))
    assert torch.allclose(details["surface_extent"], loss)


def test_protocol_rejects_blocked_or_unreviewed_tasks() -> None:
    for task in ("physical_size_mm", "relative_completeness", "positive_flag"):
        with pytest.raises(ValueError, match="belum diizinkan"):
            OntologyMarginalConfig.from_mapping(
                {"tasks": [task], "task_weights": [1.0]}
            )


def test_semantic_model_adds_no_inference_parameters_and_has_gradients() -> None:
    from ultralytics.nn.tasks import DetectionModel

    baseline = DetectionModel(str(MODEL), nc=21, verbose=False)
    semantic = OntologyDetectionModel(
        str(MODEL),
        nc=21,
        verbose=False,
        ontology_marginal={"mode": "semantic"},
    )
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in semantic.parameters()
    )

    semantic.args = type(
        "Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 2}
    )()
    batch = {
        "img": torch.randn(1, 3, 128, 128),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[5.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
    }
    semantic.train()
    predictions = semantic(batch["img"])
    loss, items = semantic.loss(batch, predictions)
    assert loss.shape == (3,)
    assert items.shape == (3,)
    loss.sum().backward()
    assert any(parameter.grad is not None for parameter in semantic.parameters())


def test_structured_configs_match_baseline_schedule_and_are_frozen() -> None:
    baseline = load_experiment(ROOT / "configs/coffee_fg/D0_yolo26n_p3.yaml")
    control = load_experiment(
        ROOT / "configs/structured_ontology/C0_yolo26n_identity_control.yaml"
    )
    semantic = load_experiment(
        ROOT / "configs/structured_ontology/S0_yolo26n_semantic_marginal.yaml"
    )
    assert baseline["model"] == control["model"] == semantic["model"]
    assert baseline["weights"] == control["weights"] == semantic["weights"]
    assert baseline["train"] == control["train"] == semantic["train"]
    assert control["ontology_marginal"] | {"mode": "semantic"} == semantic[
        "ontology_marginal"
    ]
    assert semantic["ontology_marginal"]["auxiliary_gain"] == 0.20
