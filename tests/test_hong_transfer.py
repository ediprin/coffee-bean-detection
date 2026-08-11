from pathlib import Path

import torch
import pytest

from coffee_detector.hong_transfer.audit import static_architecture_audit
from coffee_detector.hong_transfer.model import (
    DistributionShiftConvBlock,
    HongSPPFAttention,
    PartialConvBlock,
    inject_hong_transfer,
)
from coffee_detector.train import load_experiment
from coffee_detector.experiments.run_hong_yolo26_transfer import (
    _decision,
    run_hong_yolo26_transfer,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIG = ROOT / "configs/hong/HF_yolo26n_full_hong_transfer.yaml"


def _model():
    from ultralytics.nn.tasks import DetectionModel

    return DetectionModel(str(MODEL), nc=21, verbose=False)


def test_full_hong_injection_covers_paper_paths_and_is_idempotent() -> None:
    model = _model()
    head = model.model[-1]
    terminal_ids = {
        (name, level): id(getattr(head, name)[level][2])
        for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
        for level in range(3)
    }

    audit = inject_hong_transfer(model)

    assert audit["dsconv_paths"] == ["model.1", "model.3", "model.17"]
    assert audit["sppf_attention_path"] == "model.9"
    assert sum(isinstance(module, DistributionShiftConvBlock) for module in model.modules()) == 3
    assert sum(isinstance(module, HongSPPFAttention) for module in model.modules()) == 1
    assert sum(isinstance(module, PartialConvBlock) for module in model.modules()) == 24
    for name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        assert len(audit["pconv_paths"][name]) == 6
        for level in range(3):
            assert id(getattr(head, name)[level][2]) == terminal_ids[(name, level)]
    assert inject_hong_transfer(model) == audit


def test_hong_forward_keeps_end2end_contract_and_gradients() -> None:
    model = _model()
    inject_hong_transfer(model)
    inputs = torch.randn(2, 3, 64, 64)

    model.train()
    outputs = model(inputs)
    assert set(outputs) == {"one2many", "one2one"}
    scalar = sum(
        value.float().sum()
        for branch in outputs.values()
        for value in branch.values()
        if isinstance(value, torch.Tensor)
    )
    scalar.backward()
    assert model.model[1].conv.kds_scale.grad is not None
    assert model.model[-1].cv2[0][0].partial.conv.weight.grad is not None

    model.eval()
    with torch.no_grad():
        detections, raw = model(inputs)
    assert detections.shape[0] == 2
    assert detections.shape[-1] == 6
    assert 0 < detections.shape[1] <= 300
    assert set(raw) == {"one2many", "one2one"}


def test_hong_static_audit_passes_without_training(tmp_path: Path) -> None:
    result = static_architecture_audit(
        MODEL,
        tmp_path / "architecture.json",
        nc=21,
        image_size=64,
    )

    assert result["static_gate"] == "PASS"
    assert result["training_executed"] is False
    assert result["test_images_accessed"] is False
    assert result["finite_gradients"] is True
    assert result["state_reload_equal"] is True
    assert result["checkpoint_resume_equal"] is True
    assert result["module_counts"] == {
        "dsconv_blocks": 3,
        "sppf_attention": 1,
        "pconv_blocks": 24,
        "dsconv_blocks_after_fuse": 3,
    }


def test_hong_config_is_full_fifty_epoch_transfer() -> None:
    config = load_experiment(CONFIG)

    assert config["code"] == "HF"
    assert config["variant"] == "hong_transfer"
    assert config["hong_transfer"]["dsconv_layer_indices"] == [1, 3, 17]
    assert config["hong_transfer"]["pconv_ratio"] == 0.25
    assert config["train"]["epochs"] == 50
    assert config["train"]["max_det"] == 500


def test_hong_gate_requires_all_baseline_conditional_and_efficiency_rules() -> None:
    baseline_metrics = {
        "macro_map50_95": 0.40,
        "bottom3_class_map50_95": 0.10,
        "worst_class_map50_95": 0.05,
    }
    candidate_metrics = {
        "macro_map50_95": 0.41,
        "bottom3_class_map50_95": 0.095,
        "worst_class_map50_95": 0.04,
    }
    diagnostic_left = {
        "global": {
            "localization_conditioned_class_accuracy": 0.57,
            "proposal_accessibility": 0.96,
        }
    }
    diagnostic_right = {
        "global": {
            "localization_conditioned_class_accuracy": 0.60,
            "proposal_accessibility": 0.955,
        }
    }
    operational_left = {"result": {"correct_decision_f1": 0.56}}
    operational_right = {"result": {"correct_decision_f1": 0.57}}
    efficiency_left = {"latency_ms_per_image": 5.0}
    efficiency_right = {"latency_ms_per_image": 6.0}

    result = _decision(
        baseline_metrics,
        candidate_metrics,
        diagnostic_left,
        diagnostic_right,
        operational_left,
        operational_right,
        efficiency_left,
        efficiency_right,
    )

    assert result["decision"] == "PASS"
    assert all(result["criteria"].values())


def test_hong_runner_rejects_extra_seed_before_dataset_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="hanya pada seed 42"):
        run_hong_yolo26_transfer(
            tmp_path / "missing-data",
            tmp_path / "missing-summary.json",
            tmp_path / "missing.pt",
            tmp_path / "output",
            seed=123,
        )
