import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from coffee_detector.defectos_ratf import (
    RATFConfig,
    build_ratf_model,
    load_ratf_weights,
    ratf_texture_cue,
)
from coffee_detector.experiments.run_defectoscafeverde_cwcf_direct import (
    PROTOCOL as D0_REFERENCE_PROTOCOL,
)
from coffee_detector.experiments.run_defectoscafeverde_ratf import (
    CANDIDATE_CONFIG,
    HARD_CLASSES,
    NATIVE_CONFIG,
    PROTOCOL,
    build_decision,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def test_ratf_config_is_fresh_and_schedule_matched():
    native = yaml.safe_load(Path(NATIVE_CONFIG).read_text(encoding="utf-8"))
    candidate = yaml.safe_load(Path(CANDIDATE_CONFIG).read_text(encoding="utf-8"))
    assert native["model"] == candidate["model"]
    assert native["train"] == candidate["train"]
    assert candidate["weights"] == "yolo26n.pt"
    config = RATFConfig.from_mapping(candidate["ratf"])
    assert config.dwt_levels == 3
    assert config.gabor_orientations_degrees == (114.0, 172.0)


def test_ratf_texture_cue_is_deterministic_finite_and_directional():
    image = torch.zeros(2, 3, 64, 64)
    image[:, :, :, 28:36] = 1.0
    first = ratf_texture_cue(image)
    second = ratf_texture_cue(image)
    assert first.shape == (2, 3, 64, 64)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.abs().sum() > 0
    assert not torch.equal(first[:, 1], first[:, 2])


def test_ratf_starts_native_exact_then_changes_only_scores():
    from ultralytics.nn.tasks import DetectionModel

    model_yaml = str(ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml")
    torch.manual_seed(42)
    native = DetectionModel(model_yaml, ch=3, nc=12, verbose=False)
    candidate = build_ratf_model(
        model_yaml, nc=12, source=None, seed=42, config=RATFConfig(), verbose=False
    )
    image = torch.rand(1, 3, 64, 64)
    native.train()
    candidate.train()
    with torch.no_grad():
        native_raw = native(image)["one2many"]
        candidate_raw = candidate(image)["one2many"]
    assert torch.equal(native_raw["boxes"], candidate_raw["boxes"])
    assert torch.equal(native_raw["scores"], candidate_raw["scores"])

    adapter = candidate.model[-1].adapters[0]
    feature = torch.randn(2, adapter.channels, 8, 8)
    cue = torch.randn(2, 3, 32, 32)
    novel, reference = adapter.decompose(feature, cue)
    cosine_numerator = (novel * reference).sum(dim=1).abs().max()
    cosine_denominator = (
        novel.norm(dim=1) * reference.norm(dim=1)
    ).clamp_min(1e-8).max()
    assert float((cosine_numerator / cosine_denominator).detach()) < 1e-4

    candidate.zero_grad(set_to_none=True)
    raw = candidate(image)["one2many"]
    raw["scores"].sum().backward()
    gradient = candidate.model[-1].adapters[0].class_residual.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0

    with torch.no_grad():
        adapter.class_residual.weight[0, 0, 0, 0] = 0.25
        active = candidate(image)["one2many"]
    assert torch.equal(native_raw["boxes"], active["boxes"])
    assert not torch.equal(native_raw["scores"], active["scores"])


def test_ratf_native_loss_step_and_strict_checkpoint_reload():
    model_yaml = str(ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml")
    candidate = build_ratf_model(
        model_yaml, nc=12, source=None, seed=42, config=RATFConfig(), verbose=False
    )
    candidate.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [9.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]]),
    }
    loss, components = candidate(batch)
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.sum().backward()
    score_gradient = candidate.model[-1].adapters[0].class_residual.weight.grad
    assert score_gradient is not None and score_gradient.abs().sum() > 0

    restored = build_ratf_model(
        model_yaml, nc=12, source=None, seed=2026, config=RATFConfig(), verbose=False
    )
    load_ratf_weights(restored, {"model": candidate})
    for key, value in candidate.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def _result(arm, protocol, values, *, candidate=False):
    contract = {
        "dataset_audit_sha256": "dataset",
        "pretrained_checkpoint_sha256": "pretrained",
        "native_config_sha256": "native-config",
        "train": {"epochs": 50},
    }
    if candidate:
        contract["candidate_config_sha256"] = "candidate-config"
    classwise = {name: 0.5 for name in HARD_CLASSES}
    return {
        "protocol": protocol,
        "arm": arm,
        "seed": 42,
        "metrics": dict(zip(METRICS, values)),
        "map50_95_by_class": classwise,
        "run_contract": contract,
        "test_images_accessed": False,
    }


@pytest.mark.parametrize(
    ("candidate_values", "expected"),
    [
        ((0.801, 0.701, 0.601), "PROMOTE_TO_PAIRED_CONFIRMATION"),
        ((0.7995, 0.71, 0.61), "PROMOTE_TO_PAIRED_CONFIRMATION"),
        ((0.798, 0.71, 0.61), "STOP_AFTER_SEED42"),
    ],
)
def test_ratf_decision_uses_pareto_routes(tmp_path, candidate_values, expected):
    native = _result(
        "D0DIRECT", D0_REFERENCE_PROTOCOL, (0.80, 0.70, 0.60)
    )
    candidate = _result(                                                                             
        "RATF1", PROTOCOL, candidate_values, candidate=True
    )
    native_path, candidate_path = tmp_path / "native.json", tmp_path / "candidate.json"
    native_path.write_text(json.dumps(native))
    candidate_path.write_text(json.dumps(candidate))
    result = build_decision(native_path, candidate_path, tmp_path / "decision.json")
    assert result["screen"]["decision"] == expected
    assert result["paired"]["test_not_opened"] is True
    assert result["test_opened"] is False
