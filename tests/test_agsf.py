import json
from pathlib import Path

import torch
import yaml

from coffee_detector.agsf import (
    AGSFConfig,
    AGSFDetectHead,
    AGSFDetectionModel,
    load_agsf_detector_weights,
)
from coffee_detector.agsf.audit import static_agsf_audit
from coffee_detector.experiments.run_faruq_v3_agsf_synthesis import _comparison


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIG_ROOT = ROOT / "configs/agsf"


def _models(mode: str, nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(29)
    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = AGSFDetectionModel(
        str(MODEL_YAML),
        nc=nc,
        verbose=False,
        agsf=AGSFConfig(frequency_mode=mode, hidden_dim=16),
    ).eval()
    load_agsf_detector_weights(candidate, source)
    return source, candidate


def test_all_agsf_arms_are_exact_native_start_and_keep_boxes_native():
    image = torch.randn(1, 3, 128, 128)
    for mode in ("none", "additive", "gated"):
        source, candidate = _models(mode)
        with torch.inference_mode():
            native = source(image)
            zero = candidate(image)
        head = candidate.model[-1]
        assert isinstance(head, AGSFDetectHead)
        assert torch.equal(
            zero[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"]
        )
        assert torch.equal(
            zero[1]["one2one"]["scores"], native[1]["one2one"]["scores"]
        )

        with torch.no_grad():
            for layer in head.correction.class_corrections:
                layer.bias.fill_(0.1)
        with torch.inference_mode():
            active = candidate(image)
        assert torch.equal(
            active[1]["one2one"]["boxes"], native[1]["one2one"]["boxes"]
        )
        assert not torch.equal(
            active[1]["one2one"]["scores"], native[1]["one2one"]["scores"]
        )


def test_syn1_syn2_are_parameter_and_schema_matched():
    _, syn1 = _models("additive")
    _, syn2 = _models("gated")
    assert sum(p.numel() for p in syn1.parameters()) == sum(
        p.numel() for p in syn2.parameters()
    )
    state1, state2 = syn1.state_dict(), syn2.state_dict()
    assert state1.keys() == state2.keys()
    assert all(state1[key].shape == state2[key].shape for key in state1)


def test_gated_frequency_path_receives_gradients_after_active_correction():
    _, candidate = _models("gated")
    head = candidate.model[-1]
    with torch.no_grad():
        for layer in head.correction.class_corrections:
            torch.nn.init.normal_(layer.weight, std=0.02)
    candidate.train()
    output = candidate(torch.randn(1, 3, 128, 128))
    output["one2many"]["scores"].square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in head.correction.frequency_encoders.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in head.correction.frequency_gates.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in head.base_head.one2many["box_head"].parameters()
    )


def test_gated_synthesis_runs_native_yolo26_detection_loss():
    _, candidate = _models("gated")
    candidate.args = type(
        "Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 2}
    )()
    candidate.train()
    batch = {
        "img": torch.randn(1, 3, 128, 128),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    loss, components = candidate(batch)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(components).all()


def test_agsf_configs_freeze_three_arm_ablation():
    expected = {"SYN0": "none", "SYN1": "additive", "SYN2": "gated"}
    files = sorted(CONFIG_ROOT.glob("SYN*.yaml"))
    assert len(files) == 3
    for path in files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["agsf"]["frequency_mode"] == expected[payload["code"]]
        assert payload["train"]["epochs"] == 50
        assert payload["train"]["imgsz"] == 640
        assert payload["train"]["batch"] == 16


def test_static_audit_passes_and_capacity_matches(tmp_path: Path):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=5, verbose=False)
    checkpoint = tmp_path / "d0.pt"
    torch.save({"model": source}, checkpoint)
    result = static_agsf_audit(
        MODEL_YAML,
        checkpoint,
        tmp_path / "static.json",
        nc=5,
        image_size=64,
        hidden_dim=16,
    )
    assert result["decision"] == "PASS"
    assert result["capacity_gates"]["syn1_syn2_same_parameter_count"]
    assert result["capacity_gates"]["syn1_syn2_same_state_schema"]
    assert all(arm["decision"] == "PASS" for arm in result["arms"].values())


def test_agsf_notebook_is_branch_correct_resumable_and_val_only():
    notebook = ROOT / "notebooks/Faruq_V3_AGSF_Synthesis_Colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "agent/agsf-synthesis-screening" in source
    assert "resolve_drive_project_root(required_relative_paths=REQUIRED)" in source
    assert "run_faruq_v3_agsf_synthesis" in source
    assert "--stage','core'" in source
    assert "--stage','frequency'" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()


def test_agsf_protocol_is_frozen_before_training():
    protocol = (ROOT / "docs/FARUQ_V3_AGSF_SYNTHESIS_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert "Status: frozen before training" in protocol
    assert "SYN0" in protocol and "SYN1" in protocol and "SYN2" in protocol
    assert "STB1" in protocol
    assert "Test must not be extracted" in protocol
    assert "4,638,114" in protocol


def test_synthesis_gate_requires_macro_gain_and_lower_tail_preservation():
    reference = {
        "macro_map50_95": 0.88,
        "bottom3_class_map50_95": 0.80,
        "worst_class_map50_95": 0.75,
    }
    passing = {
        "macro_map50_95": 0.886,
        "bottom3_class_map50_95": 0.81,
        "worst_class_map50_95": 0.745,
    }
    failing = dict(passing, bottom3_class_map50_95=0.79)
    assert _comparison(passing, reference)["decision"] == "PASS"
    assert _comparison(failing, reference)["decision"] == "FAIL"
