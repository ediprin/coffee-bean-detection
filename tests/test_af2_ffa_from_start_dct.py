from __future__ import annotations

from pathlib import Path

import torch
import yaml

from coffee_detector.af2_ffa import AF2FFAConfig, FeatureFrequencyAdapter
from coffee_detector.af2_ffa.dct import DCT_HIGH_FREQUENCY_PAIRS, selected_dct_descriptor


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dct_descriptor_is_finite_bounded_and_channelwise() -> None:
    torch.manual_seed(7)
    value = torch.rand(2, 5, 20, 20)
    descriptor = selected_dct_descriptor(value)
    assert descriptor.shape == (2, 5)
    assert torch.isfinite(descriptor).all()
    assert (descriptor >= 0).all()
    assert (descriptor < 1).all()
    assert descriptor.abs().sum() > 0


def test_dct_frequency_set_is_small_fixed_and_high_frequency() -> None:
    assert len(DCT_HIGH_FREQUENCY_PAIRS) == 8
    assert len(set(DCT_HIGH_FREQUENCY_PAIRS)) == 8
    for fy, fx in DCT_HIGH_FREQUENCY_PAIRS:
        assert (fy * fy + fx * fx) ** 0.5 >= 0.5


def test_dct_adapter_is_exact_identity_at_zero_alpha() -> None:
    config = AF2FFAConfig(
        conditioning="spectral",
        descriptor_type="dct_selected",
        residual_gain_cap=0.10,
        gradient_matched_cap=True,
    )
    adapter = FeatureFrequencyAdapter(4, config)
    value = torch.rand(2, 4, 16, 16)
    assert torch.equal(adapter(value), value)


def test_dct_and_rfft_adapters_have_identical_parameter_count() -> None:
    rfft = FeatureFrequencyAdapter(
        32,
        AF2FFAConfig(
            descriptor_type="rfft_ratio",
            residual_gain_cap=0.10,
            gradient_matched_cap=True,
        ),
    )
    dct = FeatureFrequencyAdapter(
        32,
        AF2FFAConfig(
            descriptor_type="dct_selected",
            residual_gain_cap=0.10,
            gradient_matched_cap=True,
        ),
    )
    assert sum(p.numel() for p in rfft.parameters()) == sum(p.numel() for p in dct.parameters())
    assert rfft.state_dict().keys() == dct.state_dict().keys()


def test_from_start_train_schedule_exactly_matches_original_af2() -> None:
    af2 = yaml.safe_load((REPO_ROOT / "configs/afab/AF2_yolo26n_chaotic_amplitude.yaml").read_text())
    ffab2 = yaml.safe_load((REPO_ROOT / "configs/af2_ffa/AF2FFAB2FS_yolo26n_from_start.yaml").read_text())
    dct = yaml.safe_load((REPO_ROOT / "configs/af2_ffa/AF2FFADCTFS_yolo26n_from_start.yaml").read_text())
    assert af2["model"] == ffab2["model"] == dct["model"]
    assert af2["afab"] == ffab2["afab"] == dct["afab"]
    assert af2["train"] == ffab2["train"] == dct["train"]
    left = dict(ffab2["af2_ffa"])
    right = dict(dct["af2_ffa"])
    assert left.pop("descriptor_type") == "rfft_ratio"
    assert right.pop("descriptor_type") == "dct_selected"
    assert left == right
