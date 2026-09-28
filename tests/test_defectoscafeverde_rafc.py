from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from coffee_detector.defectos_rafc import RAFCConfig, build_rafc_model
from coffee_detector.defectos_rafc.perturb import (
    _smooth_low_frequency_mask,
    _unsigned_angular_mask,
    angular_low_frequency_amplitude_mix,
)
from coffee_detector.experiments.run_defectoscafeverde_cwcf_direct import (
    PROTOCOL as D0_PROTOCOL,
)
from coffee_detector.experiments.run_defectoscafeverde_rafc import (
    ARM,
    CANDIDATE_CONFIG,
    NATIVE_CONFIG,
    PROTOCOL,
    _screen_decision,
    build_decision,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def test_config_rejects_invalid_frequency_contract():
    with pytest.raises(ValueError):
        RAFCConfig.from_mapping({"low_radius": 0.3, "high_radius": 0.2})
    with pytest.raises(ValueError):
        RAFCConfig.from_mapping({"orientation_bins": 1})
    with pytest.raises(ValueError):
        RAFCConfig.from_mapping({"consistency_gain": 0.0})


def test_radial_mask_is_bounded_and_monotone():
    radius = torch.linspace(0.0, 0.5, 101)
    mask = _smooth_low_frequency_mask(radius, 0.04, 0.22)
    assert torch.all((0.0 <= mask) & (mask <= 1.0))
    assert torch.all(mask[:-1] >= mask[1:])
    assert mask[0] == 1.0
    assert mask[-1] == 0.0


def test_angular_mask_is_modulo_pi_and_bounded():
    angle = torch.tensor([[0.13, 0.13 + math.pi]])
    centers = torch.tensor([0.0])
    mask = _unsigned_angular_mask(angle, centers, 45.0)
    assert torch.all((0.0 <= mask) & (mask <= 1.0))
    assert mask[0, 0, 0, 0] == pytest.approx(mask[0, 0, 0, 1])


def test_fourier_view_is_active_finite_deterministic_and_shape_safe():
    torch.manual_seed(8)
    image = torch.rand(4, 3, 32, 48)
    config = RAFCConfig()
    orientations = torch.tensor([0, 1, 2, 3])
    first = angular_low_frequency_amplitude_mix(
        image, config, orientation_indices=orientations
    )
    second = angular_low_frequency_amplitude_mix(
        image, config, orientation_indices=orientations
    )
    assert first.shape == image.shape
    assert first.dtype == image.dtype
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert float((first - image).abs().max()) > 0.0
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0


def test_single_image_batch_fallback_remains_active():
    image = torch.rand(1, 3, 32, 32)
    transformed = angular_low_frequency_amplitude_mix(
        image, orientation_indices=torch.tensor([4])
    )
    assert float((transformed - image).abs().max()) > 0.0


def test_amp_dtype_roundtrip():
    image = torch.rand(2, 3, 32, 32, dtype=torch.float16)
    transformed = angular_low_frequency_amplitude_mix(
        image, orientation_indices=torch.tensor([0, 3])
    )
    assert transformed.dtype == torch.float16
    assert torch.isfinite(transformed).all()


def test_model_is_native_at_inference_and_paired_objective_backpropagates():
    from ultralytics.nn.tasks import DetectionModel

    model_yaml = str(ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        native = DetectionModel(model_yaml, ch=3, nc=12, verbose=False)
    candidate = build_rafc_model(
        model_yaml, nc=12, source=None, seed=42, config=RAFCConfig(), verbose=False
    )
    image = torch.rand(2, 3, 64, 64)
    native.eval()
    candidate.eval()
    with torch.inference_mode():
        native_output, candidate_output = native(image), candidate(image)
    assert torch.equal(native_output[0], candidate_output[0])

    candidate.train()
    candidate.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    batch = {
        "img": image,
        "img_fourier": angular_low_frequency_amplitude_mix(
            image, orientation_indices=torch.tensor([0, 3])
        ),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [9.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]]),
    }
    loss, _ = candidate(batch)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    class_gradients = [
        value.grad
        for name, value in candidate.named_parameters()
        if value.grad is not None and ("cv3" in name or "one2one_cv3" in name)
    ]
    assert class_gradients
    assert all(torch.isfinite(value).all() for value in class_gradients)
    assert sum(float(value.abs().sum()) for value in class_gradients) > 0.0


def test_config_matches_native_fresh_training_contract():
    native = yaml.safe_load(Path(NATIVE_CONFIG).read_text(encoding="utf-8"))
    candidate = yaml.safe_load(Path(CANDIDATE_CONFIG).read_text(encoding="utf-8"))
    assert native["model"] == candidate["model"]
    assert native["train"] == candidate["train"]
    assert candidate["weights"] == "yolo26n.pt"
    assert RAFCConfig.from_mapping(candidate["rafc"]) == RAFCConfig()


def test_screen_routes_are_not_rigid_half_point_macro_gate():
    pareto = _screen_decision(
        {
            "macro_map50_95": 0.001,
            "bottom3_class_map50_95": 0.002,
            "worst_class_map50_95": -0.002,
        }
    )
    assert pareto["pareto_route"] is True
    assert pareto["decision"] == "RETAIN_FOR_MATCHED_NON_FOURIER_CONTROL"

    failed = _screen_decision(
        {
            "macro_map50_95": -0.01,
            "bottom3_class_map50_95": 0.0,
            "worst_class_map50_95": 0.0,
        }
    )
    assert failed["decision"] == "STOP_AFTER_SEED42"


def _result(arm, protocol, values, contract):
    return {
        "protocol": protocol,
        "arm": arm,
        "seed": 42,
        "metrics": dict(zip(METRICS, values)),
        "run_contract": contract,
        "test_images_accessed": False,
    }


def test_decision_reuses_only_strictly_matched_native_reference(tmp_path):
    common = {
        "dataset_audit_sha256": "dataset",
        "pretrained_checkpoint_sha256": "pretrained",
        "initial_state_sha256": "initial",
        "native_config_sha256": "native",
        "train": {"epochs": 50},
    }
    native = _result("D0DIRECT", D0_PROTOCOL, (0.90, 0.80, 0.75), common)
    candidate = _result(ARM, PROTOCOL, (0.901, 0.802, 0.748), common)
    native_path, candidate_path = tmp_path / "d0.json", tmp_path / "candidate.json"
    native_path.write_text(json.dumps(native), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    decision = build_decision(native_path, candidate_path, tmp_path / "decision.json")
    assert all(decision["paired_contract"].values())
    assert decision["screen"]["decision"] == "RETAIN_FOR_MATCHED_NON_FOURIER_CONTROL"
    assert decision["test_opened"] is False

    candidate["run_contract"] = {**common, "initial_state_sha256": "different"}
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(RuntimeError, match="initial_detector_state_sha256_equal"):
        build_decision(native_path, candidate_path, tmp_path / "bad.json")
