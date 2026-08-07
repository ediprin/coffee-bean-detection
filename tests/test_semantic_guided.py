from pathlib import Path

import torch

from coffee_detector.semantic_guided import (
    SemanticGuidedConfig,
    SemanticGuidedDetectionModel,
    SemanticGuidedDetectHead,
    load_semantic_guided_weights,
)
from coffee_detector.semantic_aux.model import DEFAULT_TASKS

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models():
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    candidate = SemanticGuidedDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False, semantic_guided=SemanticGuidedConfig()
    ).eval()
    load_semantic_guided_weights(candidate, source)
    return source, candidate


def test_sg1_zero_init_is_exact_native_d0() -> None:
    source, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        guided = candidate(image)
    assert isinstance(candidate.model[-1], SemanticGuidedDetectHead)
    assert torch.allclose(native[0], guided[0], rtol=0.0, atol=1e-7)
    assert torch.equal(native[1]["one2one"]["boxes"], guided[1]["one2one"]["boxes"])
    assert torch.allclose(native[1]["one2one"]["scores"], guided[1]["one2one"]["scores"], rtol=0.0, atol=1e-7)


def test_training_exposes_semantics_and_leaf_scores() -> None:
    _, candidate = _models()
    candidate.train()
    output = candidate(torch.randn(1, 3, 128, 128))
    assert "semantic_aux_logits" in output["one2many"]
    semantics = output["one2many"]["semantic_aux_logits"]
    assert set(semantics) == set(DEFAULT_TASKS)
    anchor_count = output["one2many"]["scores"].shape[-1]
    for value in semantics.values():
        assert value.shape[0] == 1 and value.shape[1] == anchor_count


def test_nonzero_guidance_changes_scores_but_not_boxes() -> None:
    source, candidate = _models()
    head = candidate.model[-1]
    with torch.no_grad():
        head.guidance.leaf_corrections[0].bias.fill_(0.05)
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        guided = candidate(image)
    assert torch.equal(native[1]["one2one"]["boxes"], guided[1]["one2one"]["boxes"])
    assert not torch.allclose(native[1]["one2one"]["scores"], guided[1]["one2one"]["scores"])


def test_semantic_and_leaf_correction_parameters_receive_gradients() -> None:
    _, candidate = _models()
    candidate.train()
    output = candidate(torch.randn(1, 3, 128, 128))
    head = candidate.model[-1]
    objective = output["one2many"]["scores"].square().mean()
    objective = objective + output["one2many"]["semantic_aux_logits"]["entity_family"].square().mean()
    objective.backward()
    assert head.guidance.leaf_corrections[0].weight.grad is not None
    assert head.guidance.semantic_heads["entity_family"][0].weight.grad is not None
