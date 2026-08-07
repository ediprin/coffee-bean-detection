from pathlib import Path

import torch

from coffee_detector.sfr_spatial import (
    SFRSpatialConfig,
    SFRSpatialDetectionModel,
    SFRSpatialDetectHead,
    WindowSpatialFormer,
    load_sfr_spatial_weights,
    sinusoidal_position,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc=5):
    from ultralytics.nn.tasks import DetectionModel
    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = SFRSpatialDetectionModel(
        str(MODEL_YAML), nc=nc, verbose=False,
        sfr_spatial=SFRSpatialConfig(hidden_dim=32, num_heads=4, window_size=7),
    ).eval()
    load_sfr_spatial_weights(candidate, source)
    return source, candidate


def test_sinusoidal_position_is_fixed_and_finite():
    first = sinusoidal_position(49, 32, device="cpu", dtype=torch.float32)
    second = sinusoidal_position(49, 32, device="cpu", dtype=torch.float32)
    assert first.shape == (49, 32)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_window_spatial_former_restores_nondivisible_shape():
    block = WindowSpatialFormer(16, SFRSpatialConfig(hidden_dim=32, num_heads=4, window_size=7))
    value = torch.randn(2, 16, 13, 17)
    output = block(value)
    assert output.shape == (2, 32, 13, 17)
    assert torch.isfinite(output).all()


def test_sf1_starts_at_native_d0_and_preserves_boxes():
    source, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    assert isinstance(candidate.model[-1], SFRSpatialDetectHead)
    assert torch.allclose(candidate_output[0], source_output[0], rtol=0.0, atol=1e-7)
    assert torch.equal(candidate_output[1]["one2one"]["boxes"], source_output[1]["one2one"]["boxes"])
    assert torch.equal(candidate_output[1]["one2one"]["scores"], source_output[1]["one2one"]["scores"])


def test_active_sf1_changes_scores_without_changing_boxes():
    _, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode(): before = candidate(image)
    head = candidate.model[-1]
    with torch.no_grad(): head.spatial.classifiers[0].bias.fill_(0.2)
    with torch.inference_mode(): after = candidate(image)
    assert torch.equal(before[1]["one2one"]["boxes"], after[1]["one2one"]["boxes"])
    assert not torch.equal(before[1]["one2one"]["scores"], after[1]["one2one"]["scores"])


def test_training_gradient_reaches_spatial_attention():
    _, candidate = _models()
    candidate.train()
    output = candidate(torch.randn(1, 3, 128, 128))
    loss = output["one2many"]["scores"].square().mean()
    loss.backward()
    block = candidate.model[-1].spatial.blocks[0]
    assert block.attn.in_proj_weight.grad is not None
    assert torch.isfinite(block.attn.in_proj_weight.grad).all()
