from __future__ import annotations

import pytest
import torch

from coffee_detector.af2_rn import AF2RNConfig, AF2RNInputEnhancer
from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer


def test_config_is_frozen_to_no_repeat_contract() -> None:
    config = AF2RNConfig.from_mapping()
    assert config.patch_size == 32
    assert config.overlap == 0.50
    assert config.angular_bins == 360
    assert config.gamma == 0.10
    assert config.annulus_width == 1.0
    with pytest.raises(ValueError, match="dikunci"):
        AF2RNConfig.from_mapping({"angular_bins": 180})


def test_geometry_preserves_legacy_angle_bins_and_assigns_every_coefficient() -> None:
    candidate = AF2RNInputEnhancer()
    legacy = AFABInputEnhancer(AFABConfig(mode="af2"))
    assert torch.equal(candidate.angle_bin, legacy.angle_bin)
    assert candidate.annulus_bin.shape == candidate.angle_bin.shape == (32, 32)
    counts = torch.bincount(candidate.annulus_bin.flatten())
    assert counts.numel() == candidate.annulus_count
    assert torch.all(counts > 0)
    assert int(counts.sum()) == 32 * 32
    assert counts[0] == 1
    assert torch.all(counts[1:] > 1)


def test_annulus_normalization_removes_only_radial_baseline() -> None:
    enhancer = AF2RNInputEnhancer()
    magnitude = torch.ones(1, 1, 32, 32)
    for ring_id in range(enhancer.annulus_count):
        magnitude[0, 0][enhancer.annulus_bin == ring_id] *= float(ring_id + 1)
    normalized = enhancer.radial_normalize_magnitude(magnitude)
    assert torch.count_nonzero(normalized) == 0

    target_ring = min(5, enhancer.annulus_count - 1)
    coordinate = torch.nonzero(enhancer.annulus_bin == target_ring)[0]
    magnitude[0, 0, coordinate[0], coordinate[1]] *= 3.0
    normalized = enhancer.radial_normalize_magnitude(magnitude)
    assert normalized[0, 0, coordinate[0], coordinate[1]] > 0
    other_rings = enhancer.annulus_bin != target_ring
    assert torch.count_nonzero(normalized[0, 0][other_rings]) == 0


def test_dc_is_zero_and_no_ring_is_hard_removed() -> None:
    enhancer = AF2RNInputEnhancer()
    torch.manual_seed(9)
    magnitude = torch.rand(2, 3, 32, 32).add_(0.01)
    normalized = enhancer.radial_normalize_magnitude(magnitude)
    center = enhancer.af2rn_config.patch_size // 2
    assert torch.equal(
        normalized[..., center, center], torch.zeros_like(normalized[..., center, center])
    )
    # Every non-DC annulus can transmit an above-median coefficient. Radius is
    # therefore not a fixed pass/stop mask as in AF1 or AF2_RADIAL.
    for ring_id in range(1, enhancer.annulus_count):
        selected = enhancer.annulus_bin == ring_id
        assert torch.any(normalized[..., selected] > 0)


def test_candidate_is_parameter_free_finite_and_differs_from_legacy_af2() -> None:
    torch.manual_seed(17)
    value = torch.rand(1, 3, 64, 64, requires_grad=True)
    candidate = AF2RNInputEnhancer()
    legacy = AFABInputEnhancer(AFABConfig(mode="af2"))
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 0
    actual = candidate(value)
    expected = legacy(value.detach())
    assert actual.shape == value.shape
    assert actual.dtype == value.dtype
    assert torch.isfinite(actual).all()
    assert not torch.allclose(actual.detach(), expected, rtol=1.0e-5, atol=1.0e-6)
    actual.mean().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert torch.count_nonzero(value.grad) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_is_preserved(dtype: torch.dtype) -> None:
    torch.manual_seed(23)
    value = torch.rand(1, 3, 48, 48, dtype=dtype)
    output = AF2RNInputEnhancer()(value)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
