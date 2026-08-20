from __future__ import annotations

import torch

from coffee_detector.af2_iso import (
    AF2IsolatedInputEnhancer,
    frozen_arm_config,
)
from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer


def test_base_geometry_matches_legacy_af2() -> None:
    isolated = AF2IsolatedInputEnhancer(frozen_arm_config("AF2_BASE"))
    legacy = AFABInputEnhancer(
        AFABConfig(
            mode="af2",
            patch_size=32,
            overlap=0.50,
            gamma=0.10,
            angular_bins=360,
            chunk_size=128,
            eps=1.0e-8,
        )
    )
    assert torch.equal(isolated.angle_bin, legacy.angle_bin)
    assert isolated.config.radial_bands == 1
    assert isolated.config.orientation_period == 360.0


def test_base_output_matches_legacy_af2() -> None:
    torch.manual_seed(7)
    value = torch.rand(1, 3, 48, 48)
    isolated = AF2IsolatedInputEnhancer(frozen_arm_config("AF2_BASE"))
    legacy = AFABInputEnhancer(AFABConfig(mode="af2"))
    expected = legacy(value)
    actual = isolated(value)
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_radial_arm_changes_only_radial_geometry() -> None:
    base = frozen_arm_config("AF2_BASE")
    radial = frozen_arm_config("AF2_RADIAL")
    assert radial.patch_size == base.patch_size == 32
    assert radial.overlap == base.overlap == 0.50
    assert radial.gamma == base.gamma == 0.10
    assert radial.angular_bins == base.angular_bins == 360
    assert radial.orientation_period == base.orientation_period == 360.0
    assert radial.radial_boundaries == (1.0 / 3.0, 2.0 / 3.0)
    assert radial.radial_bands == 3


def test_orientation_arm_changes_only_direction_period_and_bin_count() -> None:
    base = frozen_arm_config("AF2_BASE")
    orient = frozen_arm_config("AF2_ORIENT")
    assert orient.patch_size == base.patch_size == 32
    assert orient.overlap == base.overlap == 0.50
    assert orient.gamma == base.gamma == 0.10
    assert orient.radial_boundaries == base.radial_boundaries == ()
    assert orient.orientation_period == 180.0
    assert orient.angular_bins == 180
    # Keep the angular resolution matched: one degree per bin in both arms.
    assert base.orientation_period / base.angular_bins == orient.orientation_period / orient.angular_bins == 1.0


def test_unsigned_orientation_maps_conjugate_directions_together() -> None:
    enhancer = AF2IsolatedInputEnhancer(frozen_arm_config("AF2_ORIENT"))
    m = enhancer.config.patch_size
    center = m // 2
    # Exclude the Nyquist edge (-16) whose +16 counterpart is not represented
    # in the fftshifted even-sized grid.
    for y in range(-15, 16):
        for x in range(-15, 16):
            if x == 0 and y == 0:
                continue
            a = enhancer.angle_bin[center + y, center + x]
            b = enhancer.angle_bin[center - y, center - x]
            assert int(a) == int(b)


def test_arms_are_parameter_free_and_shape_preserving() -> None:
    torch.manual_seed(11)
    value = torch.rand(1, 3, 64, 64)
    outputs = {}
    for arm in ("AF2_BASE", "AF2_RADIAL", "AF2_ORIENT"):
        enhancer = AF2IsolatedInputEnhancer(frozen_arm_config(arm))
        assert sum(parameter.numel() for parameter in enhancer.parameters()) == 0
        output = enhancer(value)
        assert output.shape == value.shape
        assert output.dtype == value.dtype
        assert torch.isfinite(output).all()
        outputs[arm] = output
    # Both isolated hypotheses must be genuine operator changes, not aliases.
    assert not torch.allclose(outputs["AF2_BASE"], outputs["AF2_RADIAL"])
    assert not torch.allclose(outputs["AF2_BASE"], outputs["AF2_ORIENT"])
