import torch

from coffee_detector.af2_parent_residual.igem_confirmation import (
    ATOL,
    _feature_residual_live,
    _numerically_preserved,
    _zero_information_residual,
)


def test_predeclared_igem_tolerance_accepts_repeat_forward_cuda_scale_jitter():
    left = torch.zeros(8)
    right = torch.full((8,), ATOL * 0.5)
    assert _numerically_preserved(left, right)


def test_predeclared_igem_tolerance_rejects_material_residual_change():
    left = torch.zeros(8)
    right = torch.full((8,), ATOL * 100.0)
    assert not _numerically_preserved(left, right)


def test_residual_activity_contract_is_structural_not_logit_magnitude_threshold():
    assert _zero_information_residual(0.0)
    assert not _zero_information_residual(ATOL * 0.5)
    assert _feature_residual_live(ATOL * 0.5)
    assert not _feature_residual_live(0.0)
    assert not _feature_residual_live(float("nan"))
    assert not _feature_residual_live(float("inf"))
