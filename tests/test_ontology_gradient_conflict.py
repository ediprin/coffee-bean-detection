import torch

from coffee_detector.analysis.ontology_gradient_conflict import (
    conflict_gate,
    enable_model_gradients,
    route_conflict_decision,
    summarize_cosines,
)


def _summary(cosines):
    return summarize_cosines(cosines)


def test_conflict_gate_requires_negative_median_and_half_batches() -> None:
    assert conflict_gate(_summary([-0.4, -0.2, -0.1, 0.1]))
    assert not conflict_gate(_summary([-0.4, 0.1, 0.2, 0.3]))
    assert not conflict_gate(_summary([-0.1, -0.1, 0.1, 0.1]))


def test_route_conflict_decision_is_frozen_before_runtime() -> None:
    conflict = _summary([-0.5, -0.3, -0.1, 0.2])
    aligned = _summary([0.1, 0.2, 0.3, 0.4])
    both = route_conflict_decision(conflict, conflict)
    assert both["decision"] == "PASS"
    assert both["next_action"] == "AUTHORIZE_DUAL_HEAD_WITH_SHARED_GRADIENT_PROJECTION"
    head = route_conflict_decision(aligned, conflict)
    assert head["next_action"] == "AUTHORIZE_DUAL_HEAD_ISOLATION_ONLY"
    neither = route_conflict_decision(aligned, aligned)
    assert neither["decision"] == "FAIL"
    assert neither["training_authorized"] is False


def test_runtime_reenables_checkpoint_gradients_before_forward() -> None:
    model = torch.nn.Linear(3, 2)
    model.requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert enable_model_gradients(model) == 8
    assert all(parameter.requires_grad for parameter in model.parameters())
