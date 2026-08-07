from pathlib import Path

import torch

from coffee_detector.semantic_aux import (
    DEFAULT_TASKS,
    SemanticAuxConfig,
    SemanticAuxDetectionModel,
    SemanticAuxDetectHead,
    load_semantic_aux_weights,
    semantic_task_spec,
)

ROOT=Path(__file__).resolve().parents[1]
MODEL_YAML=ROOT/'configs/coffee_fg/models/yolo26n-p3.yaml'


def _models():
    from ultralytics.nn.tasks import DetectionModel
    source=DetectionModel(str(MODEL_YAML),nc=21,verbose=False).eval()
    candidate=SemanticAuxDetectionModel(str(MODEL_YAML),nc=21,verbose=False,semantic_aux=SemanticAuxConfig()).eval()
    load_semantic_aux_weights(candidate,source)
    return source,candidate


def test_semantic_task_spec_has_21_leaf_mappings():
    spec=semantic_task_spec(DEFAULT_TASKS)
    assert set(spec)==set(DEFAULT_TASKS)
    for task in DEFAULT_TASKS:
        assert len(spec[task]['mapping'])==21
        assert len(spec[task]['values'])>=2


def test_lps1_inference_is_native_d0():
    source,candidate=_models(); image=torch.randn(1,3,128,128)
    with torch.inference_mode(): a=source(image); b=candidate(image)
    assert isinstance(candidate.model[-1],SemanticAuxDetectHead)
    assert torch.allclose(a[0],b[0],rtol=0.0,atol=1e-7)
    assert torch.equal(a[1]['one2one']['boxes'],b[1]['one2one']['boxes'])
    assert torch.equal(a[1]['one2one']['scores'],b[1]['one2one']['scores'])


def test_training_adds_semantic_logits_without_altering_native_score_shape():
    _,candidate=_models(); candidate.train()
    output=candidate(torch.randn(1,3,128,128))
    assert 'semantic_aux_logits' in output['one2many']
    semantic=output['one2many']['semantic_aux_logits']
    assert set(semantic)==set(DEFAULT_TASKS)
    anchor_count=output['one2many']['scores'].shape[-1]
    for value in semantic.values():
        assert value.shape[0]==1 and value.shape[1]==anchor_count


def test_semantic_heads_receive_gradient_separately_from_leaf_logits():
    _,candidate=_models(); candidate.train()
    output=candidate(torch.randn(1,3,128,128))
    semantic=output['one2many']['semantic_aux_logits']['entity_family']
    semantic.square().mean().backward()
    head=candidate.model[-1]
    assert head.semantic_aux.heads['entity_family'][0].weight.grad is not None
    # Semantic-only backward must not directly traverse native classification head.
    assert head.base_head.cv3[0][-1].weight.grad is None
