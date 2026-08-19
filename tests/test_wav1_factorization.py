"""Executable contracts for the WAV1 mechanism-factorization study."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_spectral import SpectralInputEnhancer
from coffee_detector.af2_spectral.config import frozen_arm_config as frozen_spectral_arm_config
from coffee_detector.wav1_factorization import (
    ARMS,
    TRAIN_ARMS,
    WAV1FactorizationEnhancer,
    fixed_local_highpass,
    frozen_arm_config,
    stable_minmax_spatial,
    wavelet_detail_levels,
)


ROOT = Path(__file__).resolve().parents[1]


def test_wav1_ref_is_bitwise_confirmed_operator_on_cpu():
    torch.manual_seed(19)
    image = torch.rand(2, 3, 65, 63)
    confirmed = SpectralInputEnhancer(frozen_spectral_arm_config("WAV1"))
    reference = WAV1FactorizationEnhancer(frozen_arm_config("WAV1_REF"))
    assert torch.equal(reference(image), confirmed(image))
    assert torch.equal(reference.recover(image), confirmed.recover(image))


def test_all_factorization_frontends_are_parameter_free_finite_active_and_differentiable():
    torch.manual_seed(23)
    for arm in ARMS:
        image = torch.rand(1, 3, 67, 61, requires_grad=True)
        frontend = WAV1FactorizationEnhancer(frozen_arm_config(arm))
        output = frontend(image)
        assert output.shape == image.shape
        assert output.dtype == image.dtype
        assert torch.isfinite(output).all()
        assert not torch.equal(output.detach(), image.detach())
        assert not frontend.state_dict()
        output.square().mean().backward()
        assert image.grad is not None and torch.isfinite(image.grad).all()


def test_stable_minmax_maps_numerically_flat_cues_to_zero():
    exact = torch.full((1, 1, 16, 16), 0.25)
    near = exact.clone()
    near[..., 0, 0] += 2.0e-7
    assert torch.equal(stable_minmax_spatial(exact), torch.zeros_like(exact))
    assert torch.equal(stable_minmax_spatial(near), torch.zeros_like(near))


def test_constant_image_has_zero_effect_for_new_causal_controls():
    image = torch.full((1, 3, 64, 64), 0.4)
    for arm in TRAIN_ARMS:
        frontend = WAV1FactorizationEnhancer(frozen_arm_config(arm))
        assert torch.equal(frontend(image), image), arm


def test_hp1_is_fixed_binomial_highpass_without_state():
    image = torch.zeros(1, 3, 17, 17)
    image[:, :, 8, 8] = 1.0
    cue = fixed_local_highpass(image)
    assert cue.shape == (1, 1, 17, 17)
    assert cue[0, 0, 8, 8] > 0
    assert torch.isfinite(cue).all()


def test_l1_l2_and_rawfuse_match_declared_formulas():
    torch.manual_seed(29)
    image = torch.rand(1, 3, 63, 65)
    d1, d2 = wavelet_detail_levels(image)

    l1 = WAV1FactorizationEnhancer(frozen_arm_config("WAV_L1")).recover(image)[:, :1]
    l2 = WAV1FactorizationEnhancer(frozen_arm_config("WAV_L2")).recover(image)[:, :1]
    raw = WAV1FactorizationEnhancer(frozen_arm_config("WAV_RAWFUSE")).recover(image)[:, :1]

    assert torch.allclose(l1, stable_minmax_spatial(d1), atol=1.0e-7, rtol=1.0e-7)
    assert torch.allclose(l2, stable_minmax_spatial(d2), atol=1.0e-7, rtol=1.0e-7)
    assert torch.allclose(raw, stable_minmax_spatial(d1 + d2), atol=1.0e-7, rtol=1.0e-7)


def test_new_training_configs_are_schedule_matched_and_do_not_redefine_wav1():
    payloads = {
        path.stem.rsplit("_yolo26n", 1)[0]: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (ROOT / "configs/wav1_factorization").glob("*.yaml")
    }
    assert set(payloads) == set(TRAIN_ARMS)
    assert "WAV1_REF" not in payloads
    schedules = {json.dumps(payload["train"], sort_keys=True) for payload in payloads.values()}
    assert len(schedules) == 1
    assert {payload["model"] for payload in payloads.values()} == {
        "configs/coffee_fg/models/yolo26n-p3.yaml"
    }
    for arm, payload in payloads.items():
        assert payload["code"] == arm
        assert payload["factorization"] == {"arm": arm, "eps": 1.0e-8}


def test_protocol_freezes_explanatory_scope_and_test_lock():
    protocol = (ROOT / "docs/FARUQ_V3_WAV1_MECHANISM_FACTORIZATION_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert "Status: **frozen before training**" in protocol
    assert "WAV1_REF" in protocol
    assert "HP1" in protocol and "WAV_L1" in protocol and "WAV_L2" in protocol
    assert "WAV_RAWFUSE" in protocol
    assert "locked test remains closed" in protocol
    assert "seed 42" in protocol
