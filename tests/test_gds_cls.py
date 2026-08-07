from pathlib import Path

import torch

from coffee_detector.gds_cls import (
    GDSClsConfig,
    GDSClsDetectionModel,
    GDSClsDetectHead,
    axis_aligned_grid_distance,
    load_gds_cls_weights,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_grid_distance_zero_for_identical_boxes() -> None:
    boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 25.0, 40.0]])
    distance = axis_aligned_grid_distance(boxes, boxes.clone(), grid_size=7)
    assert torch.all(distance < 1e-6)


def test_grid_distance_increases_for_shift_and_aspect_mismatch() -> None:
    target = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    close = torch.tensor([[11.0, 10.0, 51.0, 50.0]])
    far = torch.tensor([[20.0, 10.0, 60.0, 50.0]])
    aspect = torch.tensor([[10.0, 20.0, 50.0, 40.0]])
    d_close = axis_aligned_grid_distance(close, target, grid_size=7)
    d_far = axis_aligned_grid_distance(far, target, grid_size=7)
    d_aspect = axis_aligned_grid_distance(aspect, target, grid_size=7)
    assert d_far > d_close
    assert d_aspect > 0


def test_paper_k_and_threshold_are_frozen() -> None:
    config = GDSClsConfig()
    assert config.grid_size == 7
    assert config.threshold == 0.05


def test_gdsc1_wrapper_is_exact_native_inference() -> None:
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    candidate = GDSClsDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False, gds_cls=GDSClsConfig()
    ).eval()
    load_gds_cls_weights(candidate, source)
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        gds = candidate(image)
    assert isinstance(candidate.model[-1], GDSClsDetectHead)
    assert torch.equal(native[0], gds[0])
    assert torch.equal(native[1]["one2one"]["boxes"], gds[1]["one2one"]["boxes"])
    assert torch.equal(native[1]["one2one"]["scores"], gds[1]["one2one"]["scores"])
