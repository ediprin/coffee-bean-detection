from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from coffee_detector.af2_scaffold import AF2ScaffoldConfig, MultilevelTrainingScaffold
from coffee_detector.af2_scaffold import AF2ScaffoldDetectionModel, strip_training_scaffold
from coffee_detector.afab import AFABConfig
from coffee_detector.experiments.run_faruq_v3_af2_scaffold_decision import (
    run_af2_scaffold_decision,
)


def _activate(module: MultilevelTrainingScaffold) -> None:
    for adapter in module.adapters:
        nn.init.constant_(adapter.output.weight, 0.05)


def test_frozen_schedule_has_full_decay_and_native_tail():
    config = AF2ScaffoldConfig()
    assert all(config.strength(epoch) == 1.0 for epoch in range(18))
    assert 0.0 < config.strength(18) < 1.0
    assert 0.0 < config.strength(26) < 1.0
    assert all(config.strength(epoch) == 0.0 for epoch in (27, 28, 29))
    assert config.strength(18) > config.strength(22) > config.strength(26)


def test_config_rejects_partial_pyramid():
    with pytest.raises(ValueError, match="P3/P4/P5"):
        AF2ScaffoldConfig.from_mapping({"feature_levels": [0, 1]})


def test_multilevel_scaffold_is_identity_then_active_on_every_level():
    module = MultilevelTrainingScaffold((8, 16, 32))
    features = [
        torch.rand(2, 8, 20, 20),
        torch.rand(2, 16, 10, 10),
        torch.rand(2, 32, 5, 5),
    ]
    module.train()
    initial = module(features)
    assert all(torch.equal(a, b) for a, b in zip(initial, features))
    _activate(module)
    active = module(features)
    assert all(not torch.equal(a, b) for a, b in zip(active, features))


def test_multilevel_scaffold_is_always_bypassed_in_eval_and_at_zero_strength():
    module = MultilevelTrainingScaffold((8, 16, 32))
    _activate(module)
    features = [
        torch.rand(1, 8, 12, 12),
        torch.rand(1, 16, 6, 6),
        torch.rand(1, 32, 3, 3),
    ]
    module.eval()
    observed = module(features)
    assert all(torch.equal(a, b) for a, b in zip(observed, features))
    module.train()
    module.set_strength(0.0)
    observed = module(features)
    assert all(torch.equal(a, b) for a, b in zip(observed, features))


def test_multilevel_scaffold_gradients_are_finite_for_every_level():
    module = MultilevelTrainingScaffold((8, 16, 32)).train()
    _activate(module)
    features = [
        torch.rand(2, 8, 12, 12, requires_grad=True),
        torch.rand(2, 16, 6, 6, requires_grad=True),
        torch.rand(2, 32, 3, 3, requires_grad=True),
    ]
    sum(value.mean() for value in module(features)).backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in features)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_full_detector_train_step_reaches_every_scaffold_and_can_be_stripped():
    model = AF2ScaffoldDetectionModel(
        "configs/coffee_fg/models/yolo26n-p3.yaml",
        nc=21,
        verbose=False,
        afab=AFABConfig(mode="af2"),
        scaffold=AF2ScaffoldConfig(),
    )
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=30)
    model.train()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    loss, _items = model.loss(batch)
    loss.sum().backward()
    assert all(
        any(parameter.grad is not None for parameter in adapter.parameters())
        for adapter in model.model[-1].scaffold.adapters
    )
    original = sum(parameter.numel() for parameter in model.parameters())
    strip_training_scaffold(model)
    stripped = sum(parameter.numel() for parameter in model.parameters())
    assert type(model.model[-1]).__name__ == "Detect"
    assert stripped < original


def _write_result(path: Path, arm: str, values: tuple[float, float, float], *, candidate=False):
    macro, bottom, worst = values
    payload = {
        "arm": arm,
        "seed": 42,
        "metrics": {
            "macro_map50_95": macro,
            "bottom3_class_map50_95": bottom,
            "worst_class_map50_95": worst,
            "classes_without_ground_truth": [],
        },
        "test_images_accessed": False,
    }
    if candidate:
        payload.update(
            format="coffee_detector.af2_scaffold.arm_result.v1",
            native_export_deltas={
                "macro_map50_95": 0.0,
                "bottom3_class_map50_95": 0.0,
                "worst_class_map50_95": 0.0,
            },
            native_export_raw_output_numerically_consistent=True,
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_kill_gate_requires_large_gain_and_preserved_tail(tmp_path):
    control = tmp_path / "control.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "decision.json"
    _write_result(control, "AF2CTRL", (0.8900, 0.8388, 0.8354))
    _write_result(candidate, "AF2MTS1", (0.9060, 0.8500, 0.8360), candidate=True)
    result = run_af2_scaffold_decision(control, candidate, output)
    assert result["decision"] == "PASS_KILL_GATE"

    _write_result(candidate, "AF2MTS1", (0.9049, 0.8500, 0.8360), candidate=True)
    result = run_af2_scaffold_decision(control, candidate, output)
    assert result["decision"] == "FAIL_KILL_GATE"


def test_protocol_and_notebook_freeze_one_seed_without_test():
    protocol = Path(
        "docs/FARUQ_V3_AF2_MULTILEVEL_TRAINING_SCAFFOLD_PROTOCOL_2026-08-28.md"
    ).read_text(encoding="utf-8")
    notebook_path = Path("notebooks/Faruq_V3_AF2MTS1_Kill_Gate_Colab.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "90.50%" in protocol
    assert "84.50%" in protocol
    assert "'--seed','42'" in source
    assert "--authorize-training" in source
    assert "run_faruq_v3_af2_scaffold_audit" in source
    assert "run_faruq_v3_af2_scaffold_decision" in source
    assert "test/images" not in source
    assert "--authorize-test" not in source
    assert "epochs % 5 == 0" in source
