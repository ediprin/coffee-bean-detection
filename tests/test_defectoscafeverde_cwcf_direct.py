import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from coffee_detector.defectos_cwcf import (
    DEFECTOS_ATTRIBUTE_NAMES,
    build_defectos_attribute_matrix,
    build_defectos_cwcf_model,
)
from coffee_detector.experiments.run_defectoscafeverde_cwcf_direct import (
    CANDIDATE_CONFIG,
    NATIVE_CONFIG,
    PROTOCOL,
    build_decision,
)
from coffee_detector.j25_cwcf import CWCFConfig


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def test_attribute_matrix_is_unique_and_freezes_paper_confusions():
    matrix = build_defectos_attribute_matrix()
    assert matrix.shape == (12, 15)
    assert len(DEFECTOS_ATTRIBUTE_NAMES) == 15
    assert torch.unique(matrix, dim=0).shape[0] == 12
    assert matrix[[0, 5, 7], 0].tolist() == [1.0, 1.0, 1.0]
    assert matrix[[1, 6], 1].tolist() == [1.0, 1.0]
    assert matrix[[2, 8, 9], 2].tolist() == [1.0, 1.0, 1.0]


def test_configs_are_fresh_and_schedule_matched():
    native = yaml.safe_load(Path(NATIVE_CONFIG).read_text(encoding="utf-8"))
    candidate = yaml.safe_load(Path(CANDIDATE_CONFIG).read_text(encoding="utf-8"))
    assert native["model"] == candidate["model"]
    assert native["train"] == candidate["train"]
    assert candidate["weights"] == "yolo26n.pt"
    config = CWCFConfig.from_mapping(candidate["cwcf"])
    assert config.attribute_gain == 0.15
    assert config.explicit_composition is False


def test_candidate_starts_native_exact_and_auxiliary_reaches_adapter():
    from ultralytics.nn.tasks import DetectionModel

    model_yaml = str(ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml")
    torch.manual_seed(42)
    native = DetectionModel(model_yaml, ch=3, nc=12, verbose=False)
    candidate = build_defectos_cwcf_model(
        model_yaml, nc=12, source=None, seed=42, config=CWCFConfig(), verbose=False
    )
    image = torch.rand(2, 3, 64, 64)
    native.train()
    candidate.train()
    with torch.no_grad():
        native_raw, candidate_raw = native(image), candidate(image)
    assert torch.equal(native_raw["one2many"]["boxes"], candidate_raw["one2many"]["boxes"])
    assert torch.equal(native_raw["one2many"]["scores"], candidate_raw["one2many"]["scores"])

    candidate.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [9.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]]),
    }
    loss, _ = candidate(batch)
    loss.sum().backward()
    gradients = [
        parameter.grad
        for parameter in candidate.model[-1].adapters.parameters()
        if parameter.grad is not None
    ]
    assert candidate.last_attribute_loss is not None
    assert torch.isfinite(candidate.last_attribute_loss)
    assert gradients and sum(float(value.abs().sum()) for value in gradients) > 0


def _result(arm, values):
    return {
        "protocol": PROTOCOL,
        "arm": arm,
        "metrics": dict(zip(METRICS, values)),
        "map50_95_by_class": {f"class-{index}": 0.5 for index in range(12)},
        "test_images_accessed": False,
    }


def test_decision_promotes_only_frozen_routes(tmp_path):
    native = _result("D0DIRECT", (0.80, 0.70, 0.60))
    candidate = _result("DCWCF1", (0.81, 0.705, 0.595))
    native_path, candidate_path = tmp_path / "native.json", tmp_path / "candidate.json"
    native_path.write_text(json.dumps(native))
    candidate_path.write_text(json.dumps(candidate))
    result = build_decision(native_path, candidate_path, tmp_path / "decision.json")
    assert result["screen"]["overall_route"] is True
    assert result["screen"]["decision"] == "PROMOTE_TO_PAIRED_3_SEED"
    assert result["deltas"]["macro_map50_95"] == pytest.approx(0.01)
    assert result["test_opened"] is False
