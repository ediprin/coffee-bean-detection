import torch

from coffee_detector.af2_parent_residual.igem_confirmation import ATOL, _numerically_preserved


def test_predeclared_igem_tolerance_accepts_repeat_forward_cuda_scale_jitter():
    left = torch.zeros(8)
    right = torch.full((8,), ATOL * 0.5)
    assert _numerically_preserved(left, right)


def test_predeclared_igem_tolerance_rejects_material_residual_change():
    left = torch.zeros(8)
    right = torch.full((8,), ATOL * 100.0)
    assert not _numerically_preserved(left, right)
