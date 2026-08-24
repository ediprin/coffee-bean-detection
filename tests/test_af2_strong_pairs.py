from __future__ import annotations

import torch

from coffee_detector.af2_pairs.audit import _safety_decision
from coffee_detector.af2_pairs.model import AF2_CONFIG, _build_pair_model
from coffee_detector.experiments.run_faruq_v3_af2_strong_pair import _decision


MODEL = "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIGS = {
    "AF2STB1": {"window_size": 4, "num_heads": 4, "mlp_ratio": 4.0},
    "AF2IGEM1": {
        "reference_depth": 3,
        "mask_loss_weight": 0.05,
        "kernel_size": 3,
        "attention_heads": 4,
        "channel_reduction": 4,
        "correction_scale": 1.0,
    },
    "AF2SAF1": {
        "correction_scale": 1.0,
        "offset_init_zero": True,
        "sampling_ratio_note": "grid_sample_bilinear",
    },
}


def test_all_pairs_have_parameter_free_active_af2_frontend():
    sample = torch.rand(1, 3, 64, 64)
    for arm, strong in CONFIGS.items():
        model = _build_pair_model(
            arm, MODEL, nc=21, ch=3, verbose=False, strong=strong, af2=AF2_CONFIG
        ).eval()
        assert sum(value.numel() for value in model.af2.parameters()) == 0
        with torch.inference_mode():
            enhanced = model.af2(sample)
            output = model(sample)
        assert not torch.equal(sample, enhanced)
        assert torch.isfinite(enhanced).all()
        assert torch.isfinite(output[0]).all()


def test_decision_retains_strict_and_pareto_without_rigid_half_point_macro_gate():
    _, strict = _decision({
        "macro_map50_95": 0.001,
        "bottom3_class_map50_95": 0.002,
        "worst_class_map50_95": 0.003,
    })
    _, pareto = _decision({
        "macro_map50_95": -0.0005,
        "bottom3_class_map50_95": 0.006,
        "worst_class_map50_95": 0.011,
    })
    _, rejected = _decision({
        "macro_map50_95": 0.010,
        "bottom3_class_map50_95": -0.001,
        "worst_class_map50_95": 0.020,
    })
    assert strict == "RETAIN_STRICT_SUPERIOR"
    assert pareto == "RETAIN_PARETO"
    assert rejected == "REJECT"


def test_static_safety_decision_requires_test_to_remain_unaccessed():
    safe = {"wiring": True, "finite": True, "test_accessed": False}
    unsafe_test = {"wiring": True, "finite": True, "test_accessed": True}
    broken = {"wiring": False, "finite": True, "test_accessed": False}
    assert _safety_decision(safe) == "PASS"
    assert _safety_decision(unsafe_test) == "FAIL"
    assert _safety_decision(broken) == "FAIL"
