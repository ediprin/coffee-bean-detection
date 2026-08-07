from pathlib import Path

import torch
import yaml

from coffee_detector.dsr_sscb import (
    CalibratedMSDALevel,
    SSCBConfig,
    SSCBDetectHead,
    SSCBDetectionModel,
    load_sscb_detector_weights,
    rasterize_bbox_foreground,
    semantic_foreground_loss,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(mode: str = "calibrated", nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = SSCBDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        sscb=SSCBConfig(mode=mode, hidden_dim=16, sampling_points=2, max_offset_pixels=1.0),
    ).eval()
    load_sscb_detector_weights(candidate, source)
    return source, candidate


def test_sscb_initial_inference_is_exact_native():
    source, candidate = _models("calibrated")
    assert isinstance(candidate.model[-1], SSCBDetectHead)
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        sscb = candidate(image)
    assert torch.equal(sscb[0], native[0])
    assert torch.equal(sscb[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"])
    assert torch.equal(sscb[1]["one2one"]["scores"], native[1]["one2one"]["scores"])


def test_sscb_training_preserves_native_boxes_and_scores_at_initialization():
    source, candidate = _models("calibrated")
    source.train()
    candidate.train()
    image = torch.randn(1, 3, 128, 128)
    native = source(image)
    sscb = candidate(image)
    for branch in ("one2many", "one2one"):
        assert torch.equal(sscb[branch]["boxes"], native[branch]["boxes"])
        assert torch.equal(sscb[branch]["scores"], native[branch]["scores"])


def test_semantic_output_only_exists_for_semantic_arms():
    for mode, expected in (("msda", False), ("semantic_aux", True), ("calibrated", True)):
        _, candidate = _models(mode)
        candidate.train()
        output = candidate(torch.randn(1, 3, 128, 128))
        assert ("sscb_semantic_logits" in output["one2many"]) is expected
        assert "sscb_semantic_logits" not in output["one2one"]


def test_bbox_foreground_rasterizer_and_loss_are_finite():
    batch_idx = torch.tensor([0.0, 0.0])
    bboxes = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.25, 0.25]])
    target = rasterize_bbox_foreground(
        batch_idx,
        bboxes,
        batch_size=1,
        height=16,
        width=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert target.shape == (1, 1, 16, 16)
    assert 0 < target.sum().item() < target.numel()
    logits = [torch.zeros_like(target, requires_grad=True), torch.zeros(1, 1, 8, 8, requires_grad=True)]
    loss = semantic_foreground_loss(logits, {"batch_idx": batch_idx, "bboxes": bboxes})
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert all(value.grad is not None for value in logits)


def test_calibrated_msda_is_differentiable_and_semantics_change_output():
    torch.manual_seed(7)
    config = SSCBConfig(mode="calibrated", hidden_dim=8, sampling_points=2, max_offset_pixels=1.0)
    module = CalibratedMSDALevel(8, levels=3, points=2, config=config)
    # Make calibration observably active for this isolated operator test.
    with torch.no_grad():
        module.lambda_p.fill_(0.5)
        module.lambda_a.fill_(0.5)
        module.lambda_v.fill_(0.5)
        module.offset_map.weight.normal_(0, 0.02)
    query = torch.randn(1, 8, 8, 8, requires_grad=True)
    values = [
        torch.randn(1, 8, 8, 8, requires_grad=True),
        torch.randn(1, 8, 4, 4, requires_grad=True),
        torch.randn(1, 8, 2, 2, requires_grad=True),
    ]
    semantic_query = torch.randn(1, 8, 8, 8, requires_grad=True)
    semantic_values = [
        semantic_query,
        torch.randn(1, 8, 4, 4, requires_grad=True),
        torch.randn(1, 8, 2, 2, requires_grad=True),
    ]
    calibrated, _ = module(query, values, semantic_query, semantic_values)
    vanilla, _ = module(query, values, torch.zeros_like(semantic_query), [torch.zeros_like(v) for v in values])
    assert calibrated.shape == query.shape
    assert not torch.allclose(calibrated, vanilla)
    calibrated.mean().backward()
    assert module.lambda_p.grad is not None
    assert module.lambda_a.grad is not None
    assert module.lambda_v.grad is not None


def test_frozen_configs_define_three_attribution_arms():
    expected = {
        "M0_msda.yaml": "msda",
        "S0_semantic_aux_msda.yaml": "semantic_aux",
        "S1_calibrated_sscb.yaml": "calibrated",
    }
    for filename, mode in expected.items():
        payload = yaml.safe_load((ROOT / "configs/dsr_sscb" / filename).read_text(encoding="utf-8"))
        assert payload["sscb"]["mode"] == mode
        assert payload["sscb"]["hidden_dim"] == 64
        assert payload["sscb"]["sampling_points"] == 4
        assert payload["train"]["epochs"] == 50
        assert payload["train"]["imgsz"] == 640
        assert payload["train"]["seed"] == 42
