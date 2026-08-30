from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from coffee_detector.af2_sfs_cue import (
    AF2SFSCUEConfig,
    AF2SFSCUEDetectHead,
    AF2SFSCUEDetectionModel,
    factorized_dual_cue_loss,
    load_af2_sfs_cue_weights,
)
from coffee_detector.afab.model import AFABDetectionModel
from coffee_detector.afab.operator import AFABConfig
from coffee_detector.experiments.run_faruq_v3_af2_sfs_cue_direct import _screen


MODEL_CFG = "configs/coffee_fg/models/yolo26n-p3.yaml"


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


def test_config_is_frozen_single_arm():
    assert AF2SFSCUEConfig.from_mapping({}).code == "AF2SFSCUE1"
    with pytest.raises(ValueError, match="feature_level"):
        AF2SFSCUEConfig.from_mapping({"feature_level": 1})
    with pytest.raises(ValueError, match="auxiliary_gain"):
        AF2SFSCUEConfig.from_mapping({"auxiliary_gain": 0.2})
    with pytest.raises(ValueError, match="signal_mix"):
        AF2SFSCUEConfig.from_mapping({"signal_mix": 0.25})


def test_factorized_dual_target_uses_one_gate_prediction_for_cue_and_spds():
    predictions = [
        torch.rand(2, 3, side, side, requires_grad=True) for side in (16, 8, 4)
    ]
    gate = torch.rand(2, 3, 64, 64)
    raw = torch.rand(2, 3, 64, 64)
    signal = raw * gate
    combined, gate_loss, signal_loss = factorized_dual_cue_loss(
        predictions, gate, raw, signal, signal_mix=0.50
    )
    torch.testing.assert_close(combined, 0.5 * gate_loss + 0.5 * signal_loss)
    assert torch.isfinite(combined)
    combined.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in predictions)


def test_head_cue_reads_pre_adapter_and_sfs_starts_identity():
    source = AFABDetectionModel(
        MODEL_CFG, nc=21, verbose=False, afab=AFABConfig(mode="af2")
    )
    head = AF2SFSCUEDetectHead(source.model[-1], AF2SFSCUEConfig()).train()
    channels = [
        next(module.in_channels for module in branch.modules() if isinstance(module, torch.nn.Conv2d))
        for branch in head.base_head.cv2
    ]
    features = [torch.rand(2, channel, side, side) for channel, side in zip(channels, (16, 8, 4))]
    originals = [feature.clone() for feature in features]
    head(features)
    assert head.last_pre_adapter_features is not None
    assert all(torch.equal(left, right) for left, right in zip(features, originals))
    assert [tuple(value.shape) for value in head.last_auxiliary_predictions] == [
        (2, 3, 16, 16), (2, 3, 8, 8), (2, 3, 4, 4)
    ]
    assert torch.count_nonzero(head.adapter.output.weight) == 0


def test_combined_model_starts_from_exact_af2_detector_output():
    torch.manual_seed(7)
    afab = AFABConfig(mode="af2")
    source = AFABDetectionModel(MODEL_CFG, nc=21, verbose=False, afab=afab).eval()
    candidate = AF2SFSCUEDetectionModel(
        MODEL_CFG,
        nc=21,
        verbose=False,
        afab=afab,
        sfs_cue=AF2SFSCUEConfig(),
    ).eval()
    load_af2_sfs_cue_weights(candidate, source)
    sample = torch.rand(1, 3, 64, 64)
    with torch.inference_mode():
        expected = source(sample)
        observed = candidate(sample)
    first, second = _flatten(expected), _flatten(observed)
    assert len(first) == len(second)
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_combined_checkpoint_state_reloads_exactly():
    afab = AFABConfig(mode="af2")
    source = AF2SFSCUEDetectionModel(
        MODEL_CFG, nc=21, verbose=False, afab=afab, sfs_cue=AF2SFSCUEConfig()
    )
    with torch.no_grad():
        source.model[-1].adapter.output.weight.fill_(0.03)
        source.model[-1].decoders[0].weight.fill_(0.04)
    target = AF2SFSCUEDetectionModel(
        MODEL_CFG, nc=21, verbose=False, afab=afab, sfs_cue=AF2SFSCUEConfig()
    )
    transfer = load_af2_sfs_cue_weights(target, source)
    assert transfer["missing_after_partial_load"] == 0
    assert all(
        torch.equal(source.state_dict()[key], target.state_dict()[key])
        for key in source.state_dict()
    )


def test_combined_loss_is_finite_and_trains_cue_and_sfs():
    model = AF2SFSCUEDetectionModel(
        MODEL_CFG,
        nc=21,
        verbose=False,
        afab=AFABConfig(mode="af2"),
        sfs_cue=AF2SFSCUEConfig(),
    )
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, epochs=50)
    model.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]]),
    }
    loss, items = model.loss(batch)
    assert loss.shape == items.shape == (3,)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    head = model.model[-1]
    decoder_gradients = [parameter.grad for parameter in head.decoders.parameters()]
    adapter_gradients = [parameter.grad for parameter in head.adapter.parameters()]
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in decoder_gradients)
    assert any(g is not None and torch.count_nonzero(g) > 0 for g in adapter_gradients)
    assert model.last_auxiliary_components is not None
    assert set(model.last_auxiliary_components) == {"gate", "signal", "combined"}


def test_screen_only_authorizes_followup_after_large_single_arm_signal():
    historical = {
        "metrics": {
            "macro_map50_95": 0.80,
            "bottom3_class_map50_95": 0.70,
            "worst_class_map50_95": 0.67,
        }
    }
    candidate = {
        "metrics": {
            "macro_map50_95": 0.81,
            "bottom3_class_map50_95": 0.71,
            "worst_class_map50_95": 0.68,
        }
    }
    result = _screen(historical, candidate)
    assert result["decision"] == "AUTHORIZE_MATCHED_CONTROL_AND_ABLATION"
    assert "not a causal" in result["claim_status"]


def test_colab_runs_one_direct_arm_and_preserves_test_lock():
    path = Path("notebooks/Faruq_V3_AF2_SFS_CUE_Direct_Seed42_Colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "codex/af2-sfs-cue-direct" in source
    assert "run_faruq_v3_af2_sfs_cue_direct" in source
    assert "AF2SFSCUE1" in source
    assert "AF2SFS-DIRECT" not in source
    assert "AF2CUE-DIRECT" not in source
    assert "bundles/af2-direct-from-pretrained-seed42-state.zip" in source
    assert "yolo26n.pt" in source
    assert "--authorize-training" in source
    assert "test" in source.lower()
