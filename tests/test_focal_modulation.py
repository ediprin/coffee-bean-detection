import json
from pathlib import Path

import torch

from coffee_detector.experiments.run_faruq_v3_focal_modulation import (
    _fct0_comparison,
    _stb_comparison,
)
from coffee_detector.focal_modulation import (
    ClassificationFocalModulation,
    FocalModulationConfig,
    FocalModulationDetectionModel,
    FocalModulationDetectHead,
    load_focal_modulation_weights,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models():
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=21, verbose=False).eval()
    candidate = FocalModulationDetectionModel(
        str(MODEL_YAML), nc=21, verbose=False, focal_modulation=FocalModulationConfig()
    ).eval()
    load_focal_modulation_weights(candidate, source)
    return source, candidate


def test_config_and_nested_kernels_match_official_focalnet_defaults():
    config = FocalModulationConfig()
    module = ClassificationFocalModulation(32, config)
    kernels = [layer[0].kernel_size[0] for layer in module.layers[0].modulation.focal_layers]
    assert config.focal_level == 2 and config.depth == 2
    assert kernels == [3, 5]


def test_identity_gate_is_exact_and_active_operator_receives_gradients():
    module = ClassificationFocalModulation(32, FocalModulationConfig()).eval()
    value = torch.randn(2, 32, 17, 19)
    with torch.inference_mode():
        assert torch.equal(module(value), value)
    module.train()
    with torch.no_grad():
        module.gate.fill_(0.1)
    output = module(value.requires_grad_())
    assert output.shape == value.shape and not torch.allclose(output, value)
    output.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.layers.parameters()
    )


def test_fmh1_identity_start_and_active_path_preserve_native_boxes():
    source, candidate = _models()
    image = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        native, zero = source(image), candidate(image)
    assert isinstance(candidate.model[-1], FocalModulationDetectHead)
    assert torch.equal(native[0], zero[0])
    assert torch.equal(native[1]["one2one"]["boxes"], zero[1]["one2one"]["boxes"])
    assert torch.equal(native[1]["one2one"]["scores"], zero[1]["one2one"]["scores"])
    with torch.no_grad():
        for block in candidate.model[-1].blocks:
            block.gate.fill_(0.1)
    with torch.inference_mode():
        active = candidate(image)
    assert torch.equal(zero[1]["one2one"]["boxes"], active[1]["one2one"]["boxes"])
    assert not torch.equal(zero[1]["one2one"]["scores"], active[1]["one2one"]["scores"])


def test_frozen_gates_require_stb_gain_and_fct0_retention():
    stb = {"macro_map50_95": 0.886, "bottom3_class_map50_95": 0.836, "worst_class_map50_95": 0.808}
    fct0 = {"macro_map50_95": 0.894, "bottom3_class_map50_95": 0.848, "worst_class_map50_95": 0.841}
    candidate = {"macro_map50_95": 0.900, "bottom3_class_map50_95": 0.850, "worst_class_map50_95": 0.840}
    assert _stb_comparison(candidate, stb)["decision"] == "PASS"
    assert _fct0_comparison(candidate, fct0)["decision"] == "PASS"
    assert _stb_comparison(dict(candidate, macro_map50_95=0.889), stb)["decision"] == "FAIL"


def test_protocol_notebook_and_config_are_val_only_and_resume_persistent():
    protocol = (ROOT / "docs/FARUQ_V3_FOCAL_MODULATION_PROTOCOL.md").read_text(encoding="utf-8")
    assert "Status: frozen before training" in protocol and "Test remains locked" in protocol
    notebook = json.loads((ROOT / "notebooks/Faruq_V3_Focal_Modulation_Colab.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "agent/focal-modulation-classification-screening" in source
    assert "resolve_drive_project_root(required_relative_paths=REQUIRED)" in source
    assert "--stage','static'" in source and "--stage','train'" in source
    assert "--authorize-training" in source and "split=test" not in source.lower()
