import json
from pathlib import Path

import torch
import yaml

from coffee_detector.afab import AFABConfig
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.af2_ffa import (
    AF2FFAConfig,
    AF2FFADetectHead,
    AF2FFADetectionModel,
    FeatureFrequencyAdapter,
    load_af2_ffa_weights,
    run_af2_ffa_static_audit,
)
from coffee_detector.experiments.run_faruq_v3_af2_ffa_decision import (
    run_faruq_v3_af2_ffa_decision,
)
from coffee_detector.experiments.run_faruq_v3_af2_ffa_bounded_decision import (
    run_faruq_v3_af2_ffa_bounded_decision,
)
from coffee_detector.experiments.run_faruq_v3_af2_ffa_gradient_matched_decision import (
    run_faruq_v3_af2_ffa_gradient_matched_decision,
)
from coffee_detector.experiments.run_faruq_v3_af2_ffa_paired_confirmation import (
    run_faruq_v3_af2_ffa_paired_confirmation,
)
from coffee_detector.experiments.run_faruq_v3_af2_ffa_arm import (
    run_faruq_v3_af2_ffa_arm,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(conditioning="spectral", nc=5):
    afab = AFABConfig(mode="af2")
    torch.manual_seed(57)
    source = AFABDetectionModel(
        str(MODEL_YAML), nc=nc, verbose=False, afab=afab
    ).eval()
    candidate = AF2FFADetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        afab=afab,
        af2_ffa=AF2FFAConfig(conditioning=conditioning),
    ).eval()
    load_af2_ffa_weights(candidate, source)
    return source, candidate


def test_identity_start_is_bitwise_af2_and_adds_under_one_percent():
    source, candidate = _models()
    features = [
        torch.rand(1, adapter.channels, size, size)
        for adapter, size in zip(candidate.model[-1].adapters, (16, 8, 4))
    ]
    with torch.inference_mode():
        native = source.model[-1]([item.clone() for item in features])
        adapted = candidate.model[-1]([item.clone() for item in features])
    assert isinstance(candidate.model[-1], AF2FFADetectHead)
    assert torch.equal(
        native[1]["one2one"]["boxes"], adapted[1]["one2one"]["boxes"]
    )
    assert torch.equal(
        native[1]["one2one"]["scores"], adapted[1]["one2one"]["scores"]
    )
    source_parameters = sum(p.numel() for p in source.parameters())
    added = sum(p.numel() for p in candidate.parameters()) - source_parameters
    assert 0 < added < source_parameters * 0.01


def test_active_adapter_changes_only_classification_scores():
    source, candidate = _models()
    with torch.no_grad():
        torch.manual_seed(91)
        for source_parameter, candidate_parameter in zip(
            source.model[-1].one2one["cls_head"].parameters(),
            candidate.model[-1].base_head.one2one["cls_head"].parameters(),
        ):
            source_parameter.normal_(0.0, 0.05)
            candidate_parameter.copy_(source_parameter)
        for adapter in candidate.model[-1].adapters:
            adapter.alpha.fill_(0.3)
            adapter.bias.fill_(0.2)
    features = [
        torch.rand(1, adapter.channels, size, size)
        for adapter, size in zip(candidate.model[-1].adapters, (16, 8, 4))
    ]
    with torch.inference_mode():
        native = source.model[-1]([item.clone() for item in features])
        adapted = candidate.model[-1]([item.clone() for item in features])
    assert torch.equal(
        native[1]["one2one"]["boxes"], adapted[1]["one2one"]["boxes"]
    )
    assert not torch.equal(
        native[1]["one2one"]["scores"], adapted[1]["one2one"]["scores"]
    )


def test_capacity_control_is_schema_matched_but_hides_spectrum():
    _, control = _models("zero")
    _, candidate = _models("spectral")
    control_state, candidate_state = control.state_dict(), candidate.state_dict()
    assert control_state.keys() == candidate_state.keys()
    assert all(
        control_state[key].shape == candidate_state[key].shape
        for key in control_state
    )
    feature = torch.rand(2, 128, 16, 16)
    assert control.model[-1].adapters[0].spectral_descriptor(feature).eq(0).all()
    assert candidate.model[-1].adapters[0].spectral_descriptor(feature).gt(0).any()


def test_adapter_is_finite_differentiable_and_amp_safe():
    adapter = FeatureFrequencyAdapter(16, AF2FFAConfig())
    with torch.no_grad():
        adapter.alpha.fill_(0.2)
        adapter.bias.fill_(0.1)
    value = torch.rand(2, 16, 12, 12, requires_grad=True)
    output = adapter(value)
    output.square().mean().backward()
    assert output.dtype == value.dtype
    assert torch.isfinite(output).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())


def test_bounded_adapter_preserves_identity_and_caps_residual_gain():
    adapter = FeatureFrequencyAdapter(
        16, AF2FFAConfig(residual_gain_cap=0.10)
    )
    value = torch.rand(2, 16, 12, 12).clamp_min(1.0e-4)
    assert torch.equal(adapter(value), value)
    with torch.no_grad():
        adapter.alpha.fill_(100.0)
        adapter.bias.fill_(100.0)
    multiplier = (adapter(value) / value).detach()
    assert float(multiplier.min()) >= 0.9 - 1.0e-6
    assert float(multiplier.max()) <= 1.1 + 1.0e-6


def test_gradient_matched_bound_preserves_unbounded_initial_derivative():
    unbounded = FeatureFrequencyAdapter(4, AF2FFAConfig())
    legacy = FeatureFrequencyAdapter(
        4, AF2FFAConfig(residual_gain_cap=0.10)
    )
    matched = FeatureFrequencyAdapter(
        4,
        AF2FFAConfig(residual_gain_cap=0.10, gradient_matched_cap=True),
    )

    def slope(adapter):
        return torch.autograd.grad(
            adapter.residual_amplitude().sum(), adapter.alpha
        )[0]

    assert torch.equal(slope(unbounded), torch.ones(4))
    assert torch.allclose(slope(legacy), torch.full((4,), 0.10))
    assert torch.equal(slope(matched), torch.ones(4))
    with torch.no_grad():
        matched.alpha.fill_(100.0)
    assert float(matched.residual_amplitude().detach().abs().max()) <= 0.10 + 1.0e-6


def test_configs_are_capacity_matched_and_frozen():
    paths = sorted((ROOT / "configs/af2_ffa").glob("AF2FFA*.yaml"))
    assert len(paths) == 4
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    by_code = {item["code"]: item for item in payloads}
    assert set(by_code) == {"AF2FFA0", "AF2FFA1", "AF2FFAB1", "AF2FFAB2"}
    assert len({json.dumps(item["afab"], sort_keys=True) for item in payloads}) == 1
    assert len({json.dumps(item["train"], sort_keys=True) for item in payloads}) == 1
    assert payloads[0]["train"]["epochs"] == 30
    assert {item["af2_ffa"]["conditioning"] for item in payloads} == {"zero", "spectral"}
    assert by_code["AF2FFAB1"]["af2_ffa"]["residual_gain_cap"] == 0.10
    assert by_code["AF2FFAB2"]["af2_ffa"]["residual_gain_cap"] == 0.10
    assert by_code["AF2FFAB2"]["af2_ffa"]["gradient_matched_cap"] is True


def test_static_audit_passes_on_af2_checkpoint(tmp_path: Path):
    source, _ = _models()
    checkpoint = tmp_path / "af2.pt"
    torch.save({"model": source, "train_args": {"seed": 42}}, checkpoint)
    result = run_af2_ffa_static_audit(
        checkpoint, tmp_path / "static.json", device="cpu", image_size=64
    )
    assert result["decision"] == "PASS"
    assert result["gates"]["classification_path_only"]
    assert result["gates"]["same_parameter_count"]
    assert result["gates"]["bounded_same_parameter_count"]
    assert result["gates"]["bounded_gain_cap_is_10_percent"]
    assert result["gates"]["legacy_bound_gradient_confound_detected"]
    assert result["gates"]["matched_bound_initial_gradient_matches_unbounded"]
    assert result["records"]["AF2FFA1"]["initial_amplitude_gradient_mean"] == 1.0
    assert abs(
        result["records"]["AF2FFAB1"]["initial_amplitude_gradient_mean"] - 0.1
    ) < 1.0e-6
    assert result["records"]["AF2FFAB2"]["initial_amplitude_gradient_mean"] == 1.0
    assert result["added_fraction"] < 0.01


def test_decision_gate_requires_macro_and_tail(tmp_path: Path):
    reports = tmp_path / "val_reports"
    reports.mkdir()
    base = {
        "test_images_accessed": False,
        "metrics": {
            "macro_map50_95": 0.88,
            "bottom3_class_map50_95": 0.80,
            "worst_class_map50_95": 0.75,
        },
    }
    candidate = json.loads(json.dumps(base))
    candidate["metrics"].update(
        macro_map50_95=0.886,
        bottom3_class_map50_95=0.801,
        worst_class_map50_95=0.745,
    )
    for arm, payload in (("AF2FFA0", base), ("AF2FFA1", candidate)):
        (reports / f"{arm}_seed42_result.json").write_text(json.dumps(payload))
    result = run_faruq_v3_af2_ffa_decision(tmp_path)
    assert result["decision"] == "PASS"


def test_protocol_and_notebook_are_val_only():
    protocol = (ROOT / "docs/FARUQ_V3_AF2_FEATURE_FREQUENCY_ADAPTER_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert "Status: frozen before training" in protocol
    assert "AF2FFA0" in protocol and "AF2FFA1" in protocol
    notebook = ROOT / "notebooks/Faruq_V3_AF2_Feature_Frequency_Adapter_Colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "run_faruq_v3_af2_ffa_arm" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()


def test_bounded_pareto_decision_reuses_matching_seed42_reports(tmp_path: Path):
    def report(arm, macro, bottom3, worst):
        return {
            "arm": arm,
            "seed": 42,
            "initial_af2_checkpoint_sha256": "same-source",
            "test_images_accessed": False,
            "metrics": {
                "macro_map50_95": macro,
                "bottom3_class_map50_95": bottom3,
                "worst_class_map50_95": worst,
            },
        }

    paths = {}
    for arm, values in {
        "AF2FFA0": (0.889, 0.808, 0.775),
        "AF2FFA1": (0.886, 0.816, 0.804),
        "AF2FFAB1": (0.8885, 0.814, 0.801),
    }.items():
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(report(arm, *values)), encoding="utf-8")
        paths[arm] = path
    result = run_faruq_v3_af2_ffa_bounded_decision(
        paths["AF2FFA0"],
        paths["AF2FFA1"],
        paths["AF2FFAB1"],
        tmp_path / "decision.json",
    )
    assert result["decision"] == "RETAIN_PARETO"
    assert result["training_executed_for_this_study"] == ["AF2FFAB1"]
    assert result["test_opened"] is False


def test_bounded_protocol_and_notebook_are_single_arm_and_val_only():
    protocol = (
        ROOT / "docs/FARUQ_V3_AF2_FFA_BOUNDED_REFINEMENT_PROTOCOL_2026-08-22.md"
    ).read_text(encoding="utf-8")
    assert "Status: frozen before bounded training" in protocol
    assert "Only one new arm" in protocol or "Only one new arm" in protocol.replace("\n", " ")
    notebook = ROOT / "notebooks/Faruq_V3_AF2_FFA_Bounded_Seed42_Colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "AF2FFAB1" in source
    assert "run_faruq_v3_af2_ffa_bounded_decision" in source
    assert "for arm in" not in source
    assert "split=test" not in source.lower()


def test_gradient_matched_decision_reuses_only_valid_comparators(tmp_path: Path):
    def report(arm, macro, bottom3, worst):
        return {
            "arm": arm,
            "seed": 42,
            "initial_af2_checkpoint_sha256": "same-source",
            "test_images_accessed": False,
            "metrics": {
                "macro_map50_95": macro,
                "bottom3_class_map50_95": bottom3,
                "worst_class_map50_95": worst,
            },
        }

    paths = {}
    for arm, values in {
        "AF2FFA0": (0.889, 0.808, 0.775),
        "AF2FFA1": (0.886, 0.816, 0.804),
        "AF2FFAB2": (0.8891, 0.8135, 0.805),
    }.items():
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(report(arm, *values)), encoding="utf-8")
        paths[arm] = path
    result = run_faruq_v3_af2_ffa_gradient_matched_decision(
        paths["AF2FFA0"],
        paths["AF2FFA1"],
        paths["AF2FFAB2"],
        tmp_path / "decision.json",
    )
    assert result["decision"] == "RETAIN_PARETO"
    assert result["training_executed_for_this_study"] == ["AF2FFAB2"]
    assert result["invalid_historical_arm_excluded_from_comparison"] == "AF2FFAB1"
    assert result["test_opened"] is False


def test_gradient_matched_protocol_and_notebook_are_single_arm_and_val_only():
    protocol = (
        ROOT
        / "docs/FARUQ_V3_AF2_FFA_GRADIENT_MATCHED_BOUND_PROTOCOL_2026-08-22.md"
    ).read_text(encoding="utf-8")
    assert "Status: frozen before AF2FFAB2 training" in protocol
    assert "0.10*tanh(alpha/0.10)" in protocol
    notebook = (
        ROOT
        / "notebooks/Faruq_V3_AF2_FFA_Gradient_Matched_Bound_Seed42_Colab.ipynb"
    )
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "AF2FFAB2" in source
    assert "run_faruq_v3_af2_ffa_gradient_matched_decision" in source
    assert "for arm in" not in source
    assert "split=test" not in source.lower()


def test_non42_arm_requires_retained_screening_decision(tmp_path: Path):
    try:
        run_faruq_v3_af2_ffa_arm(
            "AF2FFAB2",
            tmp_path,
            tmp_path / "grouped.json",
            tmp_path / "af2.pt",
            tmp_path / "static.json",
            tmp_path / "output",
            seed=123,
            authorize_training=True,
        )
    except RuntimeError as error:
        assert "keputusan screening B2" in str(error)
    else:
        raise AssertionError("Seed non-42 lolos tanpa keputusan screening")


def test_paired_confirmation_requires_repeatable_pareto_gain(tmp_path: Path):
    controls, candidates = [], []
    for seed, deltas in {
        42: (0.00003, 0.0127, 0.0296),
        123: (-0.0005, 0.006, 0.010),
        2026: (0.001, -0.002, 0.004),
    }.items():
        for arm, values, destination in (
            ("AF2FFA0", (0.88, 0.80, 0.77), controls),
            (
                "AF2FFAB2",
                (0.88 + deltas[0], 0.80 + deltas[1], 0.77 + deltas[2]),
                candidates,
            ),
        ):
            payload = {
                "format": "coffee_detector.af2_ffa.arm_result.v1",
                "arm": arm,
                "seed": seed,
                "initial_af2_checkpoint_sha256": f"seed-{seed}",
                "test_images_accessed": False,
                "metrics": dict(
                    zip(
                        (
                            "macro_map50_95",
                            "bottom3_class_map50_95",
                            "worst_class_map50_95",
                        ),
                        values,
                    )
                ),
            }
            path = tmp_path / f"{arm}_{seed}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            destination.append(path)
    result = run_faruq_v3_af2_ffa_paired_confirmation(
        tuple(controls), tuple(candidates), tmp_path / "paired.json"
    )
    assert result["decision"] == "PASS"
    assert result["aggregate"]["macro_map50_95"]["improved_seeds"] == 2
    assert result["aggregate"]["bottom3_class_map50_95"]["improved_seeds"] == 2
    assert result["aggregate"]["worst_class_map50_95"]["improved_seeds"] == 3
    assert result["test_opened"] is False


def test_paired_protocol_and_notebooks_are_seed_matched_and_test_locked():
    protocol = (
        ROOT
        / "docs/FARUQ_V3_AF2_FFA_GRADIENT_MATCHED_PAIRED_PROTOCOL_2026-08-22.md"
    ).read_text(encoding="utf-8")
    assert "Status: frozen before seed 123/2026 confirmation training" in protocol
    assert "AF2FFA0" in protocol and "AF2FFAB2" in protocol
    for seed in (123, 2026):
        notebook = (
            ROOT / f"notebooks/Faruq_V3_AF2_FFA_B2_Paired_Seed{seed}_Colab.ipynb"
        )
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in payload.get("cells", [])
        )
        for cell in payload["cells"]:
            if cell.get("cell_type") == "code":
                compile("".join(cell["source"]), str(notebook), "exec")
        assert f"SEED={seed}" in source
        assert "AF2FFA0','AF2FFAB2" in source
        assert "--screening-decision" in source
        assert "AF2_seed{SEED}/weights/best.pt" in source
        assert "run_faruq_v3_af2_ffa_paired_confirmation" in source
        assert "split=test" not in source.lower()
