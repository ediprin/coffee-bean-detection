from __future__ import annotations

import torch
from torch import nn

from coffee_detector.af2_ffa.model import AF2FFAConfig, AF2FFADetectHead, FeatureFrequencyAdapter
from coffee_detector.experiments.run_faruq_v3_af2_ffab2_selectivity_analysis import (
    _score_against_af2,
    _variant_specs,
)


def test_historical_config_defaults_remain_replace_all_levels_full_strength():
    config = AF2FFAConfig.from_mapping(
        {
            "conditioning": "spectral",
            "descriptor_type": "rfft_ratio",
            "radial_cutoff": 0.35,
            "eps": 1.0e-6,
            "max_added_fraction": 0.01,
            "residual_gain_cap": 0.10,
            "gradient_matched_cap": True,
        }
    )
    assert config.adapter_strength_scale == 1.0
    assert config.active_levels == (True, True, True)
    assert config.fusion_mode == "replace"
    assert config.ambiguity_gate == "none"


def test_runtime_strength_zero_is_identity_even_after_adapter_is_active():
    config = AF2FFAConfig(residual_gain_cap=0.10, gradient_matched_cap=True)
    adapter = FeatureFrequencyAdapter(4, config)
    with torch.no_grad():
        adapter.alpha.fill_(0.5)
        adapter.bias.fill_(0.25)
    value = torch.rand(2, 4, 8, 8)
    adapter.set_runtime_strength(0.0)
    assert torch.equal(adapter(value), value)


def test_level_normalization_and_variant_families_are_complete():
    assert AF2FFADetectHead._normalize_levels(("P3", "P5")) == (True, False, True)
    variants = _variant_specs()
    families = {row["family"] for row in variants}
    assert {"strength", "levels", "parent_residual", "ambiguity"}.issubset(families)
    semantic = [
        (
            row["strength"],
            tuple(row["active_levels"]),
            row.get("fusion_mode"),
            row.get("residual_mix"),
            row.get("ambiguity_gate"),
            row.get("ambiguity_margin"),
        )
        for row in variants
    ]
    assert len(semantic) == len(set(semantic))


def test_diagnostic_gate_requires_macro_and_tail_improvement():
    baseline = {
        "macro_map50_95": {"values": {"42": 0.87, "123": 0.87, "2026": 0.87}},
        "bottom3_class_map50_95": {"values": {"42": 0.78, "123": 0.78, "2026": 0.78}},
        "worst_class_map50_95": {"values": {"42": 0.75, "123": 0.75, "2026": 0.75}},
    }
    candidate = {
        "macro_map50_95": {"values": {"42": 0.874, "123": 0.874, "2026": 0.874}},
        "bottom3_class_map50_95": {"values": {"42": 0.787, "123": 0.787, "2026": 0.787}},
        "worst_class_map50_95": {"values": {"42": 0.752, "123": 0.752, "2026": 0.752}},
    }
    comparison = _score_against_af2(candidate, baseline)
    assert comparison["eligible_for_retrain_screen"] is True


def _fake_detect(channels=(4, 6, 8), nc=3):
    Detect = type("Detect", (nn.Module,), {})
    base = Detect()
    base.end2end = True
    base.cv2 = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 1)) for ch in channels])
    base.nc = nc
    base.nl = 3
    base.reg_max = 1
    base.stride = torch.tensor([8.0, 16.0, 32.0])
    base.max_det = 500
    base.export = False
    base.format = ""
    base.dynamic = False
    base.agnostic_nms = False
    base.one2many = {
        "box_head": nn.ModuleList([nn.Conv2d(ch, 4, 1) for ch in channels]),
        "cls_head": nn.ModuleList([nn.Conv2d(ch, nc, 1) for ch in channels]),
    }
    base.one2one = {
        "box_head": nn.ModuleList([nn.Conv2d(ch, 4, 1) for ch in channels]),
        "cls_head": nn.ModuleList([nn.Conv2d(ch, nc, 1) for ch in channels]),
    }
    return base


def test_parent_residual_is_exact_parent_at_zero_initialized_adapter():
    config = AF2FFAConfig(
        residual_gain_cap=0.10,
        gradient_matched_cap=True,
        fusion_mode="parent_residual",
        residual_mix=1.0,
        ambiguity_gate="margin",
    )
    head = AF2FFADetectHead(_fake_detect(), config)
    feature = torch.rand(2, 4, 8, 8)
    cls_head = head.one2one["cls_head"][0]
    native = cls_head(feature)
    refined = head._classification_scores(0, feature, cls_head)
    assert torch.equal(native, refined)
    gate = head._ambiguity_weight(native)
    assert isinstance(gate, torch.Tensor)
    assert gate.shape == (2, 1, 8, 8)
    assert bool((gate >= 0).all() and (gate <= 1).all())
