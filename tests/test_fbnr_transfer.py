import torch

from coffee_detector.fbnr import (
    FBNRConfig,
    apply_fbnr_transfer,
    background_gradient_blend,
    background_linear_blend,
    build_foreground_soft_mask,
    foreground_random_conceal,
)


def _sample_batch():
    images = torch.zeros(2, 3, 32, 32)
    images[0].fill_(0.25)
    images[1].fill_(0.75)
    images[0, :, 8:24, 8:24] = 0.55
    images[1, :, 8:24, 8:24] = 0.45
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


def test_foreground_conceal_uses_paper_radius_range_and_changes_pixels() -> None:
    torch.manual_seed(7)
    images, bboxes, batch_idx = _sample_batch()
    output = foreground_random_conceal(
        images,
        bboxes,
        batch_idx,
        radius_min=0.5,
        radius_max=0.8,
        sigma_ratio=3.0,
    )
    assert output.shape == images.shape
    assert not torch.equal(output, images)
    assert torch.all(output <= images + 1e-7)
    assert torch.isfinite(output).all()


def test_linear_background_control_preserves_foreground_better_than_corner() -> None:
    images, bboxes, batch_idx = _sample_batch()
    mask = build_foreground_soft_mask(images, bboxes, batch_idx, sigma_ratio=3.0)
    output = background_linear_blend(images, mask)
    for index in range(2):
        center_error = abs(float(output[index, 0, 16, 16] - images[index, 0, 16, 16]))
        corner_error = abs(float(output[index, 0, 0, 0] - images[index, 0, 0, 0]))
        assert center_error <= corner_error + 1e-6


def test_gradient_brbb_is_finite_bounded_and_not_linear_control() -> None:
    images, bboxes, batch_idx = _sample_batch()
    mask = build_foreground_soft_mask(images, bboxes, batch_idx, sigma_ratio=3.0)
    linear = background_linear_blend(images, mask)
    gradient = background_gradient_blend(images, mask)
    assert gradient.shape == images.shape
    assert torch.isfinite(gradient).all()
    assert float(gradient.min()) >= 0.0
    assert float(gradient.max()) <= 1.0
    assert not torch.allclose(gradient, linear)


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


def test_all_discovery_modes_are_valid() -> None:
    for mode in (
        "foreground_only",
        "background_linear",
        "background_gradient",
        "stochastic_decoupled",
    ):
        assert FBNRConfig.from_mapping({"mode": mode}).mode == mode
