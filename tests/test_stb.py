from pathlib import Path

import torch

from coffee_detector.stb import (
    STBConfig,
    STBDetectionModel,
    STBDetectHead,
    ClassificationSTB,
    load_stb_weights,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models():
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    candidate = STBDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False, stb=STBConfig()
    ).eval()
    load_stb_weights(candidate, source)
    return source, candidate


def test_stb_config_matches_paper_window_and_heads() -> None:
    config = STBConfig()
    assert config.window_size == 4
    assert config.num_heads == 4


def test_gate_zero_is_exact_identity_for_nonwindow_multiple_shapes() -> None:
    module = ClassificationSTB(32, STBConfig()).eval()
    for height, width in ((17, 19), (80, 80), (40, 40)):
        value = torch.randn(2, 32, height, width)
        with torch.inference_mode():
            output = module(value)
        assert torch.equal(output, value)


def test_nonzero_gate_changes_feature_shape_safely() -> None:
    module = ClassificationSTB(32, STBConfig()).eval()
    with torch.no_grad():
        module.gate.fill_(0.25)
    value = torch.randn(2, 32, 17, 19)
    with torch.inference_mode():
        output = module(value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, value)


def test_gate_and_swin_parameters_receive_gradients_after_gate_opens() -> None:
    module = ClassificationSTB(32, STBConfig()).train()
    with torch.no_grad():
        module.gate.fill_(0.1)
    value = torch.randn(2, 32, 17, 19, requires_grad=True)
    module(value).square().mean().backward()
    assert module.gate.grad is not None
    swin_grads = [p.grad for p in module.wmsa.parameters() if p.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in swin_grads)
    assert value.grad is not None


def test_stb1_identity_start_matches_native_and_preserves_boxes() -> None:
    source, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        stb = candidate(image)
    assert isinstance(candidate.model[-1], STBDetectHead)
    assert torch.equal(native[1]["one2one"]["boxes"], stb[1]["one2one"]["boxes"])
    assert torch.equal(native[1]["one2one"]["scores"], stb[1]["one2one"]["scores"])
    assert torch.equal(native[0], stb[0])
