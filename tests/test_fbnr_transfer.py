import torch

from coffee_detector.fbnr import (
    FBNRConfig,
    apply_fbnr_transfer,
    background_soft_replace,
    build_foreground_soft_mask,
    foreground_random_conceal,
)


def _sample_batch():
    images = torch.zeros(2, 3, 32, 32)
    images[0].fill_(0.25)
    images[1].fill_(0.75)
    bboxes = torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]])
    batch_idx = torch.tensor([0.0, 1.0])
    return images, bboxes, batch_idx


def test_foreground_mask_peaks_near_box_center() -> None:
    images, bboxes, batch_idx = _sample_batch()
    mask = build_foreground_soft_mask(images, bboxes, batch_idx, sigma_ratio=3.0)
    assert mask.shape == (2, 1, 32, 32)
    center = mask[:, 0, 16, 16]
    corner = mask[:, 0, 0, 0]
    assert torch.all(center > 0.9)
    assert torch.all(center > corner)
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0


def test_foreground_conceal_preserves_geometry_and_changes_pixels() -> None:
    torch.manual_seed(7)
    images, bboxes, batch_idx = _sample_batch()
    output = foreground_random_conceal(
        images, bboxes, batch_idx, radius_min=0.15, radius_max=0.35
    )
    assert output.shape == images.shape
    assert not torch.equal(output, images)
    assert torch.all(output <= images + 1e-7)


def test_background_replace_keeps_source_foreground_more_than_background() -> None:
    images, bboxes, batch_idx = _sample_batch()
    mask = build_foreground_soft_mask(images, bboxes, batch_idx, sigma_ratio=3.0)
    torch.manual_seed(1)
    output = background_soft_replace(images, mask)
    assert output.shape == images.shape
    # The soft foreground center should remain closer to each source image than
    # a background corner after donor replacement.
    for index, source_value in enumerate((0.25, 0.75)):
        center_error = abs(float(output[index, 0, 16, 16]) - source_value)
        corner_error = abs(float(output[index, 0, 0, 0]) - source_value)
        assert center_error <= corner_error + 1e-6


def test_stochastic_transfer_is_bounded_and_same_shape() -> None:
    torch.manual_seed(11)
    images, bboxes, batch_idx = _sample_batch()
    output = apply_fbnr_transfer(
        images,
        bboxes,
        batch_idx,
        FBNRConfig(
            mode="stochastic_decoupled",
            original_probability=0.34,
            foreground_probability=0.33,
            background_probability=0.33,
        ),
    )
    assert output.shape == images.shape
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_foreground_only_config_does_not_require_probability_sum() -> None:
    config = FBNRConfig.from_mapping(
        {
            "mode": "foreground_only",
            "original_probability": 0.0,
            "foreground_probability": 1.0,
            "background_probability": 0.0,
        }
    )
    assert config.mode == "foreground_only"
