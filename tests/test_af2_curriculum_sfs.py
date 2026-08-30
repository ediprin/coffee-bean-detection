from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from coffee_detector.af2_curriculum_sfs import (
    AF2CurriculumSFSConfig,
    AF2CurriculumSFSDetectionModel,
    AF2CurriculumSFSHead,
    aligned_auxiliary_scale,
    curriculum_state,
    load_af2_curriculum_sfs_weights,
)
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig
from coffee_detector.experiments.run_faruq_v3_af2_curriculum_sfs import _decision


MODEL_CFG = "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_curriculum_schedule_is_frozen_and_reaches_expected_boundaries():
    config = AF2CurriculumSFSConfig()
    assert curriculum_state(config, epoch=0, epochs=30).phase == "coffee_warmup"
    assert curriculum_state(config, epoch=4, epochs=30).auxiliary_gain == 0.0
    assert curriculum_state(config, epoch=5, epochs=30).sfs_strength == 0.0
    assert curriculum_state(config, epoch=14, epochs=30).sfs_strength == 1.0
    assert curriculum_state(config, epoch=19, epochs=30).auxiliary_gain == 0.10
    assert curriculum_state(config, epoch=20, epochs=30).auxiliary_gain == 0.10
    assert curriculum_state(config, epoch=29, epochs=30).auxiliary_gain == pytest.approx(0.0)
    with pytest.raises(ValueError, match="epochs harus 30"):
        curriculum_state(config, epoch=0, epochs=50)


def test_gradient_alignment_passes_positive_and_blocks_negative_updates():
    positive, positive_cosine = aligned_auxiliary_scale(
        torch.ones(8), torch.ones(8)
    )
    negative, negative_cosine = aligned_auxiliary_scale(
        torch.ones(8), -torch.ones(8)
    )
    assert float(positive) == pytest.approx(1.0)
    assert float(positive_cosine) == pytest.approx(1.0)
    assert float(negative) == 0.0
    assert float(negative_cosine) == pytest.approx(-1.0)


def test_head_interpolates_between_identity_and_active_sfs():
    source = AFABDetectionModel(
        MODEL_CFG, nc=21, verbose=False, afab=AFABConfig(mode="af2")
    )
    head = AF2CurriculumSFSHead(source.model[-1], AF2CurriculumSFSConfig()).eval()
    channels = [
        next(
            module.in_channels
            for module in branch.modules()
            if isinstance(module, torch.nn.Conv2d)
        )
        for branch in head.base_head.cv2
    ]
    features = [
        torch.rand(1, channel, side, side)
        for channel, side in zip(channels, (16, 8, 4))
    ]
    with torch.no_grad():
        torch.nn.init.constant_(head.adapter.output.weight, 0.02)
        head.sfs_strength = 0.0
        identity = head(features)
        head.sfs_strength = 1.0
        active = head(features)
    identity_tensors = _flatten(identity)
    active_tensors = _flatten(active)
    assert len(identity_tensors) == len(active_tensors)
    assert any(not torch.equal(a, b) for a, b in zip(identity_tensors, active_tensors))


def _flatten(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        result = []
        for key in sorted(value):
            result.extend(_flatten(value[key]))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return []


def test_af2_parent_transfer_and_resume_are_exact():
    afab = AFABConfig(mode="af2")
    source = AFABDetectionModel(MODEL_CFG, nc=21, verbose=False, afab=afab)
    candidate = AF2CurriculumSFSDetectionModel(
        MODEL_CFG, nc=21, verbose=False, afab=afab,
        curriculum=AF2CurriculumSFSConfig(),
    )
    transfer = load_af2_curriculum_sfs_weights(candidate, source)
    assert transfer["resume"] == 0
    assert all(
        torch.equal(source.model[-1].state_dict()[key], candidate.model[-1].base_head.state_dict()[key])
        for key in source.model[-1].state_dict()
    )
    resumed = AF2CurriculumSFSDetectionModel(
        MODEL_CFG, nc=21, verbose=False, afab=afab,
        curriculum=AF2CurriculumSFSConfig(),
    )
    resume_transfer = load_af2_curriculum_sfs_weights(resumed, candidate)
    assert resume_transfer["resume"] == 1
    assert all(
        torch.equal(candidate.state_dict()[key], resumed.state_dict()[key])
        for key in candidate.state_dict()
    )


def test_curriculum_loss_is_finite_during_warmup_and_joint_phase():
    model = AF2CurriculumSFSDetectionModel(
        MODEL_CFG,
        nc=21,
        verbose=False,
        afab=AFABConfig(mode="af2"),
        curriculum=AF2CurriculumSFSConfig(),
    )
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=30)
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]]),
    }
    model.train()
    model.af2_curriculum_epoch = 0
    warmup, _ = model.loss(batch)
    assert torch.isfinite(warmup).all()
    assert model.last_curriculum_diagnostics["scheduled_auxiliary_gain"] == 0.0

    model.zero_grad(set_to_none=True)
    model.af2_curriculum_epoch = 15
    joint, _ = model.loss(batch)
    assert torch.isfinite(joint).all()
    assert model.last_curriculum_diagnostics["scheduled_auxiliary_gain"] == 0.10
    assert 0.0 <= model.last_curriculum_diagnostics["alignment_scale"] <= 1.0
    joint.sum().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.model[-1].decoders.parameters()
    )


def test_decision_has_macro_and_lower_tail_routes():
    control = {"metrics": {metric: value for metric, value in zip(
        ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95"),
        (0.89, 0.83, 0.82),
    )}}
    candidate = {"metrics": {metric: value for metric, value in zip(
        ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95"),
        (0.90, 0.835, 0.818),
    )}}
    assert _decision(control, candidate)["decision"] == "RETAIN_SEED42"
    failed = {"metrics": {metric: value for metric, value in zip(
        ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95"),
        (0.888, 0.82, 0.80),
    )}}
    assert _decision(control, failed)["decision"] == "FAIL_KILL_GATE"


def test_colab_runs_single_candidate_with_explicit_matched_control_and_test_lock():
    path = Path("notebooks/Faruq_V3_AF2_Curriculum_SFS_Seed42_Colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "codex/af2-curriculum-sfs" in source
    assert "AF2CURR1" in source
    assert "AF2CTRL_seed42_result.json" in source
    assert "run_af2_curriculum_sfs_static_audit" in source
    assert "--authorize-training" in source
    assert "123" not in source and "2026" not in source
    assert "test" in source.lower()


def test_protocol_freezes_matched_control_curriculum_and_claim_boundary():
    protocol = Path(
        "docs/FARUQ_V3_AF2_CURRICULUM_SFS_PROTOCOL_2026-08-30.md"
    ).read_text(encoding="utf-8")
    for phrase in (
        "FROZEN BEFORE TRAINING",
        "AF2CURR1 - AF2CTRL",
        "gradient",
        "0–4",
        "20–29",
        "Test remains",
    ):
        assert phrase in protocol
