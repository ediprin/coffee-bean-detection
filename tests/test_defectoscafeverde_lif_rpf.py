import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from coffee_detector.defectos_lif_rpf import (
    LIFRPFConfig,
    build_lif_rpf_model,
    luminance_illumination_preprocess,
    luminance_residual_cue,
)
from coffee_detector.experiments.run_defectoscafeverde_cwcf_direct import (
    PROTOCOL as D0_PROTOCOL,
)
from coffee_detector.experiments.run_defectoscafeverde_lif_rpf import (
    ARM,
    CANDIDATE_CONFIG,
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


def test_luminance_preprocessing_is_bounded_finite_and_color_difference_preserving():
    torch.manual_seed(7)
    image = 0.25 + 0.5 * torch.rand(2, 3, 64, 64)
    config = LIFRPFConfig()
    processed, delta, reflectance = luminance_illumination_preprocess(image, config)
    cue = luminance_residual_cue(image, config)
    assert processed.shape == image.shape
    assert delta.shape == reflectance.shape == (2, 1, 64, 64)
    assert cue.shape == (2, 2, 64, 64)
    assert torch.isfinite(processed).all() and torch.isfinite(cue).all()
    assert float(delta.abs().max()) <= config.maximum_luminance_delta + 1e-7
    assert float(delta.abs().max()) > 0.0
    assert torch.allclose(
        processed[:, 0] - processed[:, 1], image[:, 0] - image[:, 1], atol=1e-6
    )
    assert torch.equal(cue, luminance_residual_cue(image, config))


def test_lif_rpf_initially_matches_native_and_learns_only_classification_adapter():
    from ultralytics.nn.tasks import DetectionModel

    model_yaml = str(ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        native = DetectionModel(model_yaml, ch=3, nc=12, verbose=False)
    candidate = build_lif_rpf_model(
        model_yaml, nc=12, source=None, seed=42, config=LIFRPFConfig(), verbose=False
    )
    image = torch.rand(2, 3, 64, 64)
    native.train()
    candidate.train()
    with torch.no_grad():
        native_raw, candidate_raw = native(image), candidate(image)
    native_raw = native_raw["one2many"]
    candidate_raw = candidate_raw["one2many"]
    assert torch.equal(native_raw["boxes"], candidate_raw["boxes"])
    assert torch.equal(native_raw["scores"], candidate_raw["scores"])

    candidate.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [9.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]]),
    }
    loss, _ = candidate(batch)
    loss.sum().backward()
    gate_gradients = [adapter.gate.grad for adapter in candidate.model[-1].adapters]
    assert all(value is not None and torch.isfinite(value).all() for value in gate_gradients)
    assert sum(float(value.abs()) for value in gate_gradients) > 0.0

    with torch.no_grad():
        for adapter in candidate.model[-1].adapters:
            adapter.gate.fill_(1.0)
    active = candidate(image)["one2many"]
    assert torch.equal(native_raw["boxes"], active["boxes"])
    assert not torch.equal(native_raw["scores"], active["scores"])


def test_config_matches_frozen_native_schedule():
    native = yaml.safe_load(Path(NATIVE_CONFIG).read_text(encoding="utf-8"))
    candidate = yaml.safe_load(Path(CANDIDATE_CONFIG).read_text(encoding="utf-8"))
    assert native["model"] == candidate["model"]
    assert native["train"] == candidate["train"]
    assert candidate["weights"] == "yolo26n.pt"
    assert LIFRPFConfig.from_mapping(candidate["lif_rpf"]) == LIFRPFConfig()


def _result(arm, protocol, values, contract):
    return {
        "protocol": protocol,
        "arm": arm,
        "seed": 42,
        "metrics": dict(zip(METRICS, values)),
        "run_contract": contract,
        "test_images_accessed": False,
    }


def test_decision_requires_matched_historical_control(tmp_path):
    common = {
        "dataset_audit_sha256": "dataset",
        "pretrained_checkpoint_sha256": "pretrained",
        "native_config_sha256": "native",
        "train": {"epochs": 50},
    }
    native = _result("D0DIRECT", D0_PROTOCOL, (0.90, 0.80, 0.75), common)
    candidate = _result(ARM, PROTOCOL, (0.91, 0.81, 0.76), common)
    native_path, candidate_path = tmp_path / "d0.json", tmp_path / "candidate.json"
    native_path.write_text(json.dumps(native))
    candidate_path.write_text(json.dumps(candidate))
    decision = build_decision(native_path, candidate_path, tmp_path / "decision.json")
    assert all(decision["paired_contract"].values())
    assert decision["deltas"]["macro_map50_95"] == pytest.approx(0.01)
    assert decision["screen"]["decision"] == "PROMOTE_TO_PAIRED_3_SEED"
    assert decision["test_opened"] is False
