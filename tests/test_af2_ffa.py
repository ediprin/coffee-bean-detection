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


def test_configs_are_capacity_matched_and_frozen():
    paths = sorted((ROOT / "configs/af2_ffa").glob("AF2FFA*.yaml"))
    assert len(paths) == 2
    payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
    assert {item["code"] for item in payloads} == {"AF2FFA0", "AF2FFA1"}
    assert payloads[0]["afab"] == payloads[1]["afab"]
    assert payloads[0]["train"] == payloads[1]["train"]
    assert payloads[0]["train"]["epochs"] == 30
    assert {item["af2_ffa"]["conditioning"] for item in payloads} == {"zero", "spectral"}


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
