import inspect
from pathlib import Path

import torch

from coffee_detector.safpn_alignment.model import (
    SAFPNAlignmentConfig,
    SAFPNAlignmentDetectionModel,
    SAFPNAlignmentDetectHead,
    SpatialAwareAlignmentFusion,
    load_safpn_alignment_detector_weights,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = SAFPNAlignmentDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        safpn_alignment=SAFPNAlignmentConfig(),
    ).eval()
    load_safpn_alignment_detector_weights(candidate, source)
    return source, candidate


def test_safm_shapes_and_zero_offset_initialization() -> None:
    module = SpatialAwareAlignmentFusion(16, 32, offset_init_zero=True)
    shallow = torch.randn(2, 16, 20, 20)
    deep = torch.randn(2, 32, 10, 10)
    fused, diagnostics = module(shallow, deep)
    assert fused.shape == shallow.shape
    assert diagnostics["weight"].shape == (2, 1, 20, 20)
    assert diagnostics["shallow_offset"].shape == (2, 2, 20, 20)
    assert diagnostics["deep_offset"].shape == (2, 2, 20, 20)
    assert torch.count_nonzero(diagnostics["shallow_offset"]) == 0
    assert torch.count_nonzero(diagnostics["deep_offset"]) == 0


def test_fresh_safpn_wrapper_is_exact_d0_before_learning() -> None:
    source, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    head = candidate.model[-1]
    assert isinstance(head, SAFPNAlignmentDetectHead)
    assert torch.allclose(candidate_output[0], source_output[0], rtol=0.0, atol=1e-7)
    assert set(candidate_output[1]) == {"one2many", "one2one"}
    assert torch.equal(
        candidate_output[1]["one2one"]["boxes"],
        source_output[1]["one2one"]["boxes"],
    )
    assert torch.equal(
        candidate_output[1]["one2one"]["scores"],
        source_output[1]["one2one"]["scores"],
    )
    source_code = inspect.getsource(type(head)) + inspect.getsource(type(head.alignment))
    assert "roi_align" not in source_code
    assert ".topk(" not in source_code
    assert "_get_decode_boxes" not in source_code


def test_alignment_changes_only_scores_when_correction_is_active() -> None:
    _, candidate = _models()
    head = candidate.model[-1]
    candidate.eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        before = candidate(image)[1]["one2one"]
        head.alignment.class_corrections[0].weight.fill_(0.01)
        head.alignment.class_corrections[1].weight.fill_(0.01)
        after = candidate(image)[1]["one2one"]
    assert torch.equal(before["boxes"], after["boxes"])
    assert not torch.equal(before["scores"], after["scores"])


def test_gradients_reach_alignment_after_classifier_becomes_active() -> None:
    _, candidate = _models()
    head = candidate.model[-1]
    with torch.no_grad():
        head.alignment.class_corrections[0].weight.fill_(0.01)
        head.alignment.class_corrections[1].weight.fill_(0.01)
    candidate.train()
    output = candidate(torch.randn(1, 3, 128, 128))
    loss = output["one2many"]["scores"].square().mean()
    loss.backward()
    assert head.alignment.class_corrections[0].weight.grad is not None
    assert head.alignment.p4_to_p3.shallow_offset.weight.grad is not None
    assert head.alignment.p5_to_p4.deep_offset.weight.grad is not None
    assert head.base_head.cv2[0][-1].weight.grad is None
    assert head.base_head.cv3[0][-1].weight.grad is not None


def test_fused_inference_keeps_one_to_one_contract() -> None:
    _, candidate = _models()
    head = candidate.model[-1]
    head.fuse()
    candidate.eval()
    with torch.inference_mode():
        detections, predictions = candidate(torch.randn(1, 3, 128, 128))
    assert detections.shape[0] == 1
    assert set(predictions) == {"one2one"}
    assert predictions["one2one"]["boxes"].shape[0] == 1
