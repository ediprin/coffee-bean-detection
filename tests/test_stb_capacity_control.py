import json
import csv
from pathlib import Path

import torch

from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _comparison,
    _control_validity,
    _epochs,
)
from coffee_detector.stb import STBConfig, STBDetectionModel, load_stb_weights
from coffee_detector.stb_control import (
    ClassificationChannelControl,
    STBCapacityControlDetectionModel,
    load_stb_control_weights,
    static_stb_capacity_control_audit,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models():
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    stb = STBDetectionModel(str(MODEL_YAML), nc=21, verbose=False, stb=STBConfig()).eval()
    control = STBCapacityControlDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False, stb=STBConfig()
    ).eval()
    load_stb_weights(stb, source)
    load_stb_control_weights(control, source)
    return source, stb, control


def test_control_is_two_block_pointwise_channel_mixer():
    module = ClassificationChannelControl(32, STBConfig())
    assert len(module.blocks) == 2
    assert not any(isinstance(child, torch.nn.Conv2d) for child in module.modules())
    value = torch.rand(2, 32, 9, 11)
    with torch.inference_mode():
        assert torch.equal(module(value), value)


def test_stb_and_control_are_capacity_near_matched_and_identity_safe():
    source, stb, control = _models()
    stb_params = sum(parameter.numel() for parameter in stb.parameters())
    control_params = sum(parameter.numel() for parameter in control.parameters())
    assert abs(control_params - stb_params) / stb_params <= 0.0005
    image = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        native, left, right = source(image), stb(image), control(image)
    for candidate in (left, right):
        assert torch.equal(native[0], candidate[0])
        assert torch.equal(native[1]["one2one"]["boxes"], candidate[1]["one2one"]["boxes"])
        assert torch.equal(native[1]["one2one"]["scores"], candidate[1]["one2one"]["scores"])


def test_static_audit_passes(tmp_path):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    checkpoint = tmp_path / "d0.pt"
    torch.save({"model": source}, checkpoint)
    result = static_stb_capacity_control_audit(
        MODEL_YAML, checkpoint, tmp_path / "audit.json", image_size=64
    )
    assert result["decision"] == "PASS"
    assert result["parameter_relative_gap"] <= 0.0005


def test_gate_requires_viable_control_and_stb_advantage():
    d0ft = {"macro_map50_95": 0.86, "bottom3_class_map50_95": 0.75, "worst_class_map50_95": 0.72}
    control = {"macro_map50_95": 0.87, "bottom3_class_map50_95": 0.78, "worst_class_map50_95": 0.76}
    stb = {"macro_map50_95": 0.88, "bottom3_class_map50_95": 0.83, "worst_class_map50_95": 0.80}
    assert _control_validity(control, d0ft)["decision"] == "PASS"
    assert _comparison(stb, control)["decision"] == "PASS"
    assert _comparison(dict(stb, macro_map50_95=0.873), control)["decision"] == "FAIL"


def test_epoch_audit_rejects_interleaved_concurrent_resume(tmp_path):
    path = tmp_path / "results.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch"])
        writer.writeheader()
        writer.writerows({"epoch": value} for value in (1, 2, 3, 2, 4))
    try:
        _epochs(path)
    except RuntimeError as error:
        assert "tidak monotonik" in str(error)
    else:
        raise AssertionError("Interleaved CSV seharusnya ditolak")


def test_protocol_and_notebook_are_frozen_resumable_and_val_only():
    protocol = (ROOT / "docs/FARUQ_V3_STB_CAPACITY_CAUSAL_CONTROL_PROTOCOL.md").read_text(encoding="utf-8")
    assert "Status: frozen before training" in protocol and "Test remains locked" in protocol
    payload = json.loads((ROOT / "notebooks/Faruq_V3_STB_Capacity_Control_Colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert "agent/stb-capacity-causal-control" in source
    assert "resolve_drive_project_root(required_relative_paths=REQUIRED)" in source
    assert "--stage','static'" in source and "--stage','train'" in source
    assert "--authorize-training" in source and "split=test" not in source.lower()
    assert "--recover-from-best" in source
