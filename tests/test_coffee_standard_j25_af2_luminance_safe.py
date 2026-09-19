import json
import ast
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_luminance import (
    AF2LuminanceInputEnhancer,
    AF2LuminanceSafeDetectionModel,
    AF2LuminanceStochasticInputEnhancer,
    EpochWeightedSampler,
    identity_repeat_factor_weights,
)
from coffee_detector.afab import AFABConfig
from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_safe import (
    ARM,
    LUMINANCE_PROTOCOL,
    PROTOCOL,
    REFERENCE_PROTOCOL,
    build_decision,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _config() -> AFABConfig:
    return AFABConfig(mode="af2", patch_size=32, overlap=0.5, chunk_size=8)


def test_stochastic_luminance_endpoints_and_chromaticity():
    torch.manual_seed(19)
    value = 0.1 + 0.8 * torch.rand(2, 3, 64, 64)
    safe = AF2LuminanceStochasticInputEnhancer(_config())
    fixed = AF2LuminanceInputEnhancer(_config())
    raw = safe.forward_with_strength(value, 0.0)
    half = safe.forward_with_strength(value, torch.tensor([0.25, 0.75]))
    full = safe.forward_with_strength(value, 1.0)
    assert torch.equal(raw, value)
    assert torch.equal(full, fixed(value))
    ratio = value[:, 0] / value[:, 1]
    assert torch.allclose(ratio, half[:, 0] / half[:, 1], atol=2e-6, rtol=2e-6)
    safe.eval()
    assert torch.equal(safe(value), full)


def test_safe_model_has_native_detector_state_schema():
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(23)
    native = DetectionModel(str(MODEL_YAML), nc=5, verbose=False)
    torch.manual_seed(23)
    candidate = AF2LuminanceSafeDetectionModel(
        str(MODEL_YAML), nc=5, verbose=False, afab=_config()
    )
    assert list(native.state_dict()) == list(candidate.state_dict())
    assert all(
        torch.equal(native.state_dict()[key], candidate.state_dict()[key])
        for key in native.state_dict()
    )


def test_identity_repeat_weights_split_derivative_weight_and_repeat_rare_class():
    paths = []
    class_sets = []
    identities = []
    for index in range(16):
        identity = f"{index:064x}"
        identities.append(identity)
        classes = {0} if index < 10 else ({1} if index < 15 else {2})
        suffixes = ("_00", "_01") if index == 15 else ("",)
        for suffix in suffixes:
            paths.append(Path(f"{identity}{suffix}.jpg"))
            class_sets.append(classes)
    weights, summary = identity_repeat_factor_weights(
        paths, class_sets, class_count=3, maximum_repeat=4.0
    )
    rare_indexes = [index for index, path in enumerate(paths) if path.stem.startswith(identities[-1])]
    common_index = next(index for index, path in enumerate(paths) if path.stem.startswith(identities[0]))
    assert summary["source_identities"] == 16
    assert summary["class_repeat_factors"]["2"] > 1.0
    assert torch.isclose(weights[rare_indexes].sum(), torch.tensor(summary["class_repeat_factors"]["2"], dtype=torch.double))
    assert weights[rare_indexes[0]] < weights[common_index] * summary["class_repeat_factors"]["2"]


def test_epoch_weighted_sampler_is_resume_deterministic():
    sampler = EpochWeightedSampler(torch.tensor([1.0, 2.0, 3.0]), 20, seed=99)
    sampler.set_epoch(7)
    first = list(sampler)
    sampler.set_epoch(7)
    assert list(sampler) == first
    sampler.set_epoch(8)
    assert list(sampler) != first


def test_safe_config_freezes_semantic_safe_schedule():
    config = yaml.safe_load(
        (ROOT / "configs/coffee_standard_j25/AF2LUMSAFE.yaml").read_text()
    )
    train = config["train"]
    assert config["frontend"] == "luminance_shared_stochastic"
    assert train["epochs"] == 50
    assert train["mosaic"] == train["scale"] == train["hsv_h"] == 0.0
    assert train["erasing"] == train["mixup"] == train["copy_paste"] == 0.0
    assert config["sampler"]["threshold"] == "median_positive_train_identity_frequency"


def _result(path: Path, protocol: str, values: tuple[float, float, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "protocol": protocol,
                "test_images_accessed": False,
                "metrics": dict(
                    zip(
                        ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95"),
                        values,
                    )
                ),
            }
        )
    )


def test_decision_uses_threshold_free_pareto_dominance(tmp_path):
    direct, luminance, candidate = (tmp_path / name for name in ("direct", "lum", "safe"))
    _result(direct / "val_reports/D0DIRECT_seed42_result.json", REFERENCE_PROTOCOL, (0.60, 0.20, 0.10))
    _result(luminance / "val_reports/AF2LUMDIRECT_seed42_result.json", LUMINANCE_PROTOCOL, (0.61, 0.19, 0.09))
    _result(candidate / f"val_reports/{ARM}_seed42_result.json", PROTOCOL, (0.62, 0.20, 0.10))
    result = build_decision(direct, luminance, candidate, tmp_path / "decision.json")
    assert result["decision"] == "PARETO_ADVANCE"
    assert set(result["dominates"]) == {"D0DIRECT", "AF2LUMDIRECT"}
    assert result["test_opened"] is False


def test_colab_notebook_is_self_contained_resumable_and_test_locked():
    path = ROOT / "notebooks/Coffee_Standard_J25_AF2LUMSAFE_Seed42_Colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/af2-luminance-safe'" in code
    assert "retain_train_siblings=True" in code
    assert "coffee_standard_j25_train_siblings_summary.json" in code
    assert "run_coffee_standard_j25_af2_luminance_safe" in code
    assert "coffee-standard-j25-af2-luminance-safe-v1" in code
    assert "--authorize-training" in code
    assert "last.pt tersimpan di Drive setiap epoch" in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
