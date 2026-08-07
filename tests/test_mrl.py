from pathlib import Path

import torch

from coffee_detector.mrl import (
    MRLConfig,
    MRLDetectionModel,
    MRLDetectHead,
    grouped_euclidean_distance,
    load_mrl_detector_weights,
    multi_roi_loss,
    square_ring_masks,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_7x7_grouping_matches_four_center_to_outer_rings() -> None:
    masks = square_ring_masks(7)
    assert len(masks) == 4
    assert [int(mask.sum()) for mask in masks] == [1, 8, 16, 24]
    stacked = torch.stack(masks).to(torch.int64)
    assert torch.equal(stacked.sum(dim=0), torch.ones(7, 7, dtype=torch.int64))


def test_grouped_euclidean_distance_is_zero_for_identical_rois() -> None:
    value = torch.randn(5, 12, 7, 7)
    distance = grouped_euclidean_distance(value, value.clone())
    assert torch.equal(distance, torch.zeros_like(distance))


def test_mrl_prefers_closer_positive_than_negative() -> None:
    anchor = torch.zeros(4, 3, 7, 7)
    positive = anchor.clone()
    negative = torch.ones_like(anchor)
    good = multi_roi_loss(anchor, positive, negative)
    bad = multi_roi_loss(anchor, negative, positive)
    assert good < bad
    assert torch.isfinite(good) and torch.isfinite(bad)


def test_mrl_gradient_reaches_roi_features() -> None:
    anchor = torch.randn(3, 4, 7, 7, requires_grad=True)
    positive = torch.randn(3, 4, 7, 7, requires_grad=True)
    negative = torch.randn(3, 4, 7, 7, requires_grad=True)
    loss = multi_roi_loss(anchor, positive, negative)
    loss.backward()
    for tensor in (anchor, positive, negative):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_mrl_wrapper_is_exact_native_inference() -> None:
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    candidate = MRLDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False, mrl=MRLConfig()
    ).eval()
    load_mrl_detector_weights(candidate, source)
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native = source(image)
        metric = candidate(image)
    assert isinstance(candidate.model[-1], MRLDetectHead)
    assert torch.equal(native[0], metric[0])
    assert torch.equal(native[1]["one2one"]["boxes"], metric[1]["one2one"]["boxes"])
    assert torch.equal(native[1]["one2one"]["scores"], metric[1]["one2one"]["scores"])


def test_mrl_config_is_four_groups_at_paper_roi_size() -> None:
    config = MRLConfig.from_mapping({"roi_size": 7, "loss_weight": 1.0})
    assert config.groups == 4
    assert config.loss_weight == 1.0
