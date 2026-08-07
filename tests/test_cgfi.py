from pathlib import Path

import torch

from coffee_detector.cgfi import (
    CGFIConfig,
    CGFIDetectionModel,
    CGFIDetectHead,
    CGFIFeatureEnhancer,
    load_cgfi_weights,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models():
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    candidate = CGFIDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False, cgfi=CGFIConfig()
    ).eval()
    load_cgfi_weights(candidate, source)
    return source, candidate


def test_cgfi_feature_enhancer_is_identity_at_initialization() -> None:
    module = CGFIFeatureEnhancer(16, CGFIConfig()).eval()
    value = torch.randn(2, 16, 17, 19)
    with torch.inference_mode():
        output = module(value)
    assert torch.allclose(output, value, rtol=1e-5, atol=1e-5)


def test_cg1_initial_prediction_matches_native_and_boxes_are_exact() -> None:
    source, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        enhanced = candidate(image)
    assert isinstance(candidate.model[-1], CGFIDetectHead)
    assert torch.equal(native[1]["one2one"]["boxes"], enhanced[1]["one2one"]["boxes"])
    assert torch.allclose(native[1]["one2one"]["scores"], enhanced[1]["one2one"]["scores"], rtol=1e-5, atol=1e-5)
    assert torch.allclose(native[0], enhanced[0], rtol=1e-5, atol=1e-5)


def test_frequency_filter_change_alters_scores_not_boxes() -> None:
    source, candidate = _models()
    head = candidate.model[-1]
    last = head.enhancers[0].filter.filter_net[-1]
    with torch.no_grad():
        last.bias[: head.enhancers[0].filter.channels].fill_(1.25)
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        enhanced = candidate(image)
    assert torch.equal(native[1]["one2one"]["boxes"], enhanced[1]["one2one"]["boxes"])
    assert not torch.allclose(native[1]["one2one"]["scores"], enhanced[1]["one2one"]["scores"])


def test_cgfi_filter_receives_gradient() -> None:
    module = CGFIFeatureEnhancer(8, CGFIConfig()).train()
    value = torch.randn(2, 8, 12, 14, requires_grad=True)
    module(value).square().mean().backward()
    assert module.filter.filter_net[-1].weight.grad is not None
    assert torch.isfinite(module.filter.filter_net[-1].weight.grad).all()
    assert value.grad is not None
