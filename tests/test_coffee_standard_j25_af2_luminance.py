import json
import ast
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_luminance import (
    AF2LuminanceDetectionModel,
    AF2LuminanceInputEnhancer,
    rec709_luminance,
)
from coffee_detector.afab import AFABConfig, AFABInputEnhancer, minmax_spatial
from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance import (
    ARM,
    PROTOCOL,
    REFERENCE_PROTOCOL,
    build_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _config() -> AFABConfig:
    return AFABConfig(mode="af2", patch_size=32, overlap=0.5, chunk_size=8)


def test_shared_gate_is_invariant_to_isoluminant_chroma_but_rgb_gate_is_not():
    torch.manual_seed(7)
    first = 0.4 + 0.2 * torch.rand(1, 3, 64, 64)
    delta = 0.025 * torch.sin(torch.linspace(0, 12, 64)).view(1, 1, 1, 64)
    second = first.clone()
    second[:, 0:1] += delta
    second[:, 1:2] -= delta * (0.2126 / 0.7152)
    assert torch.allclose(
        rec709_luminance(first), rec709_luminance(second), atol=1e-6, rtol=0
    )
    luminance = AF2LuminanceInputEnhancer(_config())
    legacy = AFABInputEnhancer(_config())
    with torch.inference_mode():
        shared_difference = (luminance.shared_gate(first) - luminance.shared_gate(second)).abs().max()
        rgb_first = minmax_spatial(legacy.recover(first))
        rgb_second = minmax_spatial(legacy.recover(second))
        rgb_difference = (rgb_first - rgb_second).abs().max()
    assert shared_difference <= 1e-4
    assert rgb_difference > 1e-3


def test_luminance_frontend_is_parameter_free_finite_and_raw_preserving():
    frontend = AF2LuminanceInputEnhancer(_config())
    value = torch.rand(2, 3, 65, 71, requires_grad=True)
    output = frontend(value)
    assert output.shape == value.shape
    assert sum(parameter.numel() for parameter in frontend.parameters()) == 0
    assert torch.isfinite(output).all()
    assert not torch.equal(output, value)
    output.mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_luminance_model_has_native_detector_state_schema():
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(11)
    native = DetectionModel(str(MODEL_YAML), nc=5, verbose=False)
    torch.manual_seed(11)
    candidate = AF2LuminanceDetectionModel(
        str(MODEL_YAML), nc=5, verbose=False, afab=_config()
    )
    assert list(native.state_dict()) == list(candidate.state_dict())
    assert all(
        torch.equal(native.state_dict()[key], candidate.state_dict()[key])
        for key in native.state_dict()
    )


def test_luminance_config_is_matched_to_rgb_direct():
    native = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/D0DIRECT_TRAIN_SIBLINGS.yaml").read_text()
    )
    rgb = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/AF2DIRECT_TRAIN_SIBLINGS.yaml").read_text()
    )
    luminance = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/AF2LUMDIRECT_TRAIN_SIBLINGS.yaml").read_text()
    )
    assert native["model"] == rgb["model"] == luminance["model"]
    assert native["train"] == rgb["train"] == luminance["train"]
    assert rgb["afab"] == luminance["afab"]
    assert luminance["frontend"] == "luminance_shared"


def _write_result(path: Path, arm: str, metrics: tuple[float, float, float]) -> None:
    protocol = PROTOCOL if arm == ARM else REFERENCE_PROTOCOL
    payload = {
        "protocol": protocol,
        "reference_protocol": REFERENCE_PROTOCOL if arm == ARM else None,
        "test_images_accessed": False,
        "metrics": dict(
            zip(
                ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95"),
                metrics,
            )
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_decision_identifies_luminance_tail_recovery(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_result(reference / "val_reports/D0DIRECT_seed42_result.json", "D0DIRECT", (0.60, 0.20, 0.10))
    _write_result(reference / "val_reports/AF2DIRECT_seed42_result.json", "AF2DIRECT", (0.61, 0.18, 0.08))
    _write_result(candidate / f"val_reports/{ARM}_seed42_result.json", ARM, (0.609, 0.20, 0.10))
    result = build_decision(reference, candidate, tmp_path / "decision.json")
    assert result["criteria"]["luminance_lower_tail_route"] is True
    assert result["interpretation"] == "SUPPORTS_RGB_CHANNEL_INTERFERENCE"
    assert result["test_opened"] is False


def test_colab_notebook_is_fresh_matched_and_test_locked():
    path = ROOT / "notebooks/Coffee_Standard_J25_AF2LUMDIRECT_Seed42_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "codex/af2-luminance-isolation" in code
    assert "retain_train_siblings=True" in code
    assert "run_coffee_standard_j25_af2_luminance" in code
    assert "coffee-standard-j25-af2-luminance-v1" in code
    assert "--authorize-training" in code
    assert "build_decision" in code
    assert "test/images" not in code and "--authorize-test" not in code
