from pathlib import Path

import torch
import yaml

from coffee_detector.af2r import (
    AF2RConfig,
    AF2RDetectionModel,
    AF2ResidualGateEnhancer,
    illumination_features,
)
from coffee_detector.afab import AFABConfig, AFABInputEnhancer
from coffee_detector.experiments.run_faruq_v3_af2r_decision import (
    run_faruq_v3_af2r_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def afab_config():
    return AFABConfig(mode="af2", patch_size=32, overlap=0.5, chunk_size=8)


def test_illumination_features_have_six_aligned_finite_channels():
    raw = torch.rand(2, 3, 48, 64)
    recovered = torch.rand_like(raw)
    features = illumination_features(raw, recovered, local_kernel=15)
    assert features.shape == (2, 6, 48, 64)
    assert torch.isfinite(features).all()


def test_zero_initialized_adaptive_enhancer_is_exact_fixed_af2():
    raw = torch.rand(1, 3, 64, 64)
    fixed = AFABInputEnhancer(afab_config())
    adaptive = AF2ResidualGateEnhancer(
        afab_config(), AF2RConfig(conditioning="illumination", hidden_channels=4)
    )
    output, gate = adaptive.forward_with_gate(raw)
    assert torch.equal(gate, torch.ones_like(gate))
    assert torch.equal(output, fixed(raw))


def test_zero_control_and_candidate_have_identical_schema_and_initial_output():
    raw = torch.rand(1, 3, 64, 64)
    control = AF2ResidualGateEnhancer(
        afab_config(), AF2RConfig(conditioning="zero", hidden_channels=4)
    )
    candidate = AF2ResidualGateEnhancer(
        afab_config(), AF2RConfig(conditioning="illumination", hidden_channels=4)
    )
    candidate.load_state_dict(control.state_dict())
    assert control.state_dict().keys() == candidate.state_dict().keys()
    assert torch.equal(control(raw), candidate(raw))
    recovered = candidate.af2.recover(raw)
    normalized = (recovered - recovered.amin((-2, -1), keepdim=True)) / (
        recovered.amax((-2, -1), keepdim=True)
        - recovered.amin((-2, -1), keepdim=True)
    ).clamp_min(candidate.afab_config.eps)
    assert control.conditioning(raw, normalized).abs().sum() == 0
    assert candidate.conditioning(raw, normalized).abs().sum() > 0


def test_gate_can_suppress_af2_residual_and_receives_finite_gradients():
    raw = torch.rand(1, 3, 64, 64)
    module = AF2ResidualGateEnhancer(
        afab_config(), AF2RConfig(conditioning="illumination", hidden_channels=4)
    )
    module.gate[-1].bias.data.fill_(-0.5)
    output, gate = module.forward_with_gate(raw)
    fixed = module.af2(raw)
    assert gate.min() >= 0 and gate.max() <= 2
    assert not torch.equal(output, fixed)
    output.mean().backward()
    gradients = [parameter.grad for parameter in module.gate.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_af2r_model_arms_match_parameter_count_and_schema():
    control = AF2RDetectionModel(
        str(MODEL_YAML),
        nc=5,
        verbose=False,
        afab=afab_config(),
        af2r=AF2RConfig(conditioning="zero"),
    )
    candidate = AF2RDetectionModel(
        str(MODEL_YAML),
        nc=5,
        verbose=False,
        afab=afab_config(),
        af2r=AF2RConfig(conditioning="illumination"),
    )
    assert sum(p.numel() for p in control.parameters()) == sum(
        p.numel() for p in candidate.parameters()
    )
    assert {
        key: tuple(value.shape) for key, value in control.state_dict().items()
    } == {key: tuple(value.shape) for key, value in candidate.state_dict().items()}


def test_af2r_configs_differ_only_in_conditioning():
    paths = {
        path.stem.split("_")[0]: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (ROOT / "configs/af2r").glob("AF2R*.yaml")
    }
    assert set(paths) == {"AF2R0", "AF2R1"}
    assert paths["AF2R0"]["model"] == paths["AF2R1"]["model"]
    assert paths["AF2R0"]["afab"] == paths["AF2R1"]["afab"]
    assert paths["AF2R0"]["train"] == paths["AF2R1"]["train"]
    left = dict(paths["AF2R0"]["af2r"])
    right = dict(paths["AF2R1"]["af2r"])
    assert left.pop("conditioning") == "zero"
    assert right.pop("conditioning") == "illumination"
    assert left == right


def test_af2r_decision_requires_matched_and_fixed_af2_gains(tmp_path):
    output = tmp_path / "experiment"
    reports = output / "val_reports"
    reports.mkdir(parents=True)
    fixed_metrics = {
        "macro_map50_95": 0.88,
        "bottom3_class_map50_95": 0.80,
        "worst_class_map50_95": 0.78,
    }
    reference = {
        "protocol": "faruq-v3-lfdet-afab-breadth-screening-v1",
        "seed": 42,
        "test_images_accessed": False,
        "decisions": {"AF2": {"decision": "RETAIN"}},
        "candidate": {"AF2": {"metrics": fixed_metrics}},
    }
    reference_path = tmp_path / "af2.json"
    reference_path.write_text(__import__("json").dumps(reference), encoding="utf-8")
    values = {
        "AF2R0": (0.879, 0.798, 0.775),
        "AF2R1": (0.886, 0.806, 0.781),
    }
    for arm, metrics in values.items():
        payload = {
            "test_images_accessed": False,
            "metrics": dict(zip(("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95"), metrics)),
        }
        (reports / f"{arm}_seed42_result.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )
    result = run_faruq_v3_af2r_decision(output, reference_path)
    assert result["decision"] == "PASS"
    assert result["next"] == "AUTHORIZE_PAIRED_ILLUMINATION_SCREEN"
    assert result["test_opened"] is False
