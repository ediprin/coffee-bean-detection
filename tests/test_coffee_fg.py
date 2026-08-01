import json
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

from coffee_detector.coffee_fg.model import (
    CoffeeFGConfig,
    CoffeeFGDetectHead,
    CoffeeFGDetectionModel,
    MultiLevelROIRefiner,
    inject_coffee_fg,
)
from coffee_detector.coffee_fg.loss import CoffeeFGLoss
from coffee_detector.analysis.coffee_fg_diagnostics import (
    _count_summary,
    _greedy_match,
    compare_p3_p2_diagnostics,
)
from coffee_detector.evaluate import _classwise_summary
from coffee_detector.experiments.run_coffee_fg_screening import (
    _cached_evaluation,
    run_coffee_fg_screening,
)
from coffee_detector.train import load_experiment


ROOT = Path(__file__).resolve().parents[1]
P2_MODEL = ROOT / "configs/coffee_fg/models/yolo26n-p2.yaml"


def test_cached_evaluation_requires_matching_complete_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    data_root = tmp_path / "data"
    data_root.mkdir()
    report = tmp_path / "report.json"
    payload = {
        "checkpoint": str(checkpoint),
        "data": str(data_root),
        "split": "val",
        "metrics": {
            "macro_map50_95": 0.4,
            "bottom3_class_map50_95": 0.1,
            "worst_class_map50_95": 0.0,
            "classes_without_ground_truth": [],
        },
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    assert _cached_evaluation(report, checkpoint, data_root, "val") == payload
    assert _cached_evaluation(report, checkpoint, data_root, "test") is None
    payload["metrics"]["classes_without_ground_truth"] = ["rare"]
    report.write_text(json.dumps(payload), encoding="utf-8")
    assert _cached_evaluation(report, checkpoint, data_root, "val") is None


def test_coffee_fg_loss_forwards_resumed_e2e_schedule_update() -> None:
    class BaseLoss:
        updates = 0

        def update(self) -> None:
            self.updates += 1

    loss = object.__new__(CoffeeFGLoss)
    loss.base = BaseLoss()
    loss.updates = 7

    loss.update()

    assert loss.base.updates == 8
    assert loss.updates == 8


def test_auxiliary_targets_follow_image_compute_device_not_stride_metadata() -> None:
    loss = object.__new__(CoffeeFGLoss)
    loss.head = SimpleNamespace(
        stride=torch.tensor([8.0, 16.0, 32.0]),
        config=SimpleNamespace(box_expand=1.1),
    )
    batch = {
        "img": torch.empty((1, 3, 32, 32), device="meta"),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[2.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }

    rois, labels, matching = loss._target_rois(batch)

    assert rois.device.type == "meta"
    assert labels.device.type == "meta"
    assert matching.device.type == "meta"


def test_first_order_and_bilinear_controls_are_capacity_matched() -> None:
    channels = (16, 32, 64)
    first = MultiLevelROIRefiner(
        channels,
        7,
        CoffeeFGConfig(mode="first_order", rank=8, roi_size=3),
    )
    bilinear = MultiLevelROIRefiner(
        channels,
        7,
        CoffeeFGConfig(mode="bilinear", rank=8, roi_size=3),
    )

    assert first.parameter_count() == bilinear.parameter_count()

    features = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    rois = torch.tensor(
        [[0.0, 2.0, 2.0, 20.0, 20.0], [1.0, 4.0, 5.0, 26.0, 24.0]]
    )
    assert first(features, rois, (4.0, 8.0, 16.0)).shape == (2, 7)
    assert bilinear(features, rois, (4.0, 8.0, 16.0)).shape == (2, 7)


def test_coffee_fg_wraps_yolo26_without_replacing_box_branches() -> None:
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(P2_MODEL), nc=5, verbose=False)
    original_head = model.model[-1]
    original_box_branch_ids = [id(branch) for branch in original_head.cv2]

    assert inject_coffee_fg(
        model,
        CoffeeFGConfig(mode="bilinear", rank=8, roi_size=3, topk=10),
    ) == 1
    head = model.model[-1]
    assert isinstance(head, CoffeeFGDetectHead)
    assert [id(branch) for branch in head.base_head.cv2] == original_box_branch_ids
    assert inject_coffee_fg(model, head.config) == 0


def test_coffee_fg_model_preserves_ultralytics_output_contract_and_gradients() -> None:
    model = CoffeeFGDetectionModel(
        str(P2_MODEL),
        nc=5,
        verbose=False,
        coffee_fg={
            "mode": "bilinear",
            "rank": 8,
            "roi_size": 3,
            "topk": 10,
        },
    )
    model.args = type(
        "Args",
        (),
        {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 2},
    )()
    batch = {
        "img": torch.randn(2, 3, 128, 128),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.3, 0.3], [0.4, 0.4, 0.2, 0.2]]
        ),
    }

    model.train()
    predictions = model(batch["img"])
    assert set(predictions) == {"one2many", "one2one"}
    loss, items = model.loss(batch, predictions)
    assert loss.shape == (4,)
    assert items.shape == (4,)
    loss.sum().backward()
    assert model.model[-1].refiner.classifier.weight.grad is not None

    model.eval()
    with torch.no_grad():
        detections, raw = model(batch["img"])
    assert detections.shape == (2, 300, 6)
    assert "coffee_fg_indices" in raw["one2one"]


def test_coffee_fg_configs_are_loadable() -> None:
    for prefix in ("R0_yolo26n_p3", "R2_yolo26n_p2"):
        first = load_experiment(
            ROOT / f"configs/coffee_fg/{prefix}_first_order.yaml"
        )
        bilinear_prefix = prefix.replace("R0_", "R1_").replace("R2_", "R3_")
        bilinear = load_experiment(
            ROOT / f"configs/coffee_fg/{bilinear_prefix}_bilinear.yaml"
        )
        assert first["variant"] == bilinear["variant"] == "coffee_fg"
        assert first["coffee_fg"]["mode"] == "first_order"
        assert bilinear["coffee_fg"]["mode"] == "bilinear"
        assert first["weights"] == bilinear["weights"] == "yolo26n.pt"
        assert first["coffee_fg"]["topk"] == bilinear["coffee_fg"]["topk"] == 500
        assert (
            first["coffee_fg"]["training_topk"]
            == bilinear["coffee_fg"]["training_topk"]
            == 500
        )
        assert first["train"]["max_det"] == bilinear["train"]["max_det"] == 500


def test_quick_p3_p2_controls_share_ten_epoch_schedule() -> None:
    p3 = load_experiment(ROOT / "configs/coffee_fg/D0Q_yolo26n_p3_quick10.yaml")
    p2 = load_experiment(ROOT / "configs/coffee_fg/D1Q_yolo26n_p2_quick10.yaml")
    assert p3["code"] == "D0Q"
    assert p2["code"] == "D1Q"
    assert p3["train"] == p2["train"]
    assert p3["train"]["epochs"] == 10
    assert p3["model"] != p2["model"]


def test_quick_refiners_are_capacity_matched_and_use_scaled_curriculum() -> None:
    first = load_experiment(
        ROOT / "configs/coffee_fg/R0Q_yolo26n_p3_first_order_quick10.yaml"
    )
    bilinear = load_experiment(
        ROOT / "configs/coffee_fg/R1Q_yolo26n_p3_bilinear_quick10.yaml"
    )
    assert first["code"] == "R0Q"
    assert bilinear["code"] == "R1Q"
    assert first["train"] == bilinear["train"]
    assert first["train"]["epochs"] == 10
    assert first["coffee_fg"]["predicted_start_epoch"] == 2
    assert first["coffee_fg"]["predicted_full_epoch"] == 5
    assert first["coffee_fg"] | {"mode": "bilinear"} == bilinear["coffee_fg"]


def test_coffee_fg_test_split_is_locked_before_dataset_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Test terkunci"):
        run_coffee_fg_screening(
            tmp_path / "missing-data",
            tmp_path / "output",
            models=("D0",),
            seeds=(42,),
            evaluation_split="test",
            open_test=False,
        )


def test_refiner_training_requires_diagnostic_before_dataset_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="diagnostic proposal/headroom"):
        run_coffee_fg_screening(
            tmp_path / "missing-data",
            tmp_path / "output",
            models=("R0", "R1"),
            seeds=(42,),
        )


def test_classwise_summary_does_not_invent_ap_for_missing_classes() -> None:
    metric = type(
        "Metric",
        (),
        {
            "ap_class_index": torch.tensor([0, 2]).numpy(),
            "ap": torch.tensor(
                [[0.2, 0.4, 0.6], [0.6, 0.8, 1.0]], dtype=torch.float64
            ).numpy(),
        },
    )()

    summary = _classwise_summary(metric, {0: "black", 1: "sour", 2: "insect"})

    assert summary["map50_95_by_class"] == {
        "black": pytest.approx(0.4),
        "insect": pytest.approx(0.8),
    }
    assert summary["classes_without_ground_truth"] == ["sour"]
    assert summary["macro_map50_95"] == pytest.approx(0.6)


def test_diagnostic_matching_and_counting() -> None:
    predictions = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    )
    targets = torch.tensor(
        [[1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]]
    )
    matches = _greedy_match(predictions, targets, 0.5)
    assert {(left, right) for left, right, _ in matches} == {(0, 0), (1, 1)}

    summary = _count_summary(
        [
            {"target": 300, "predicted": 300},
            {"target": 340, "predicted": 300},
        ]
    )
    assert summary["exact_count_accuracy"] == pytest.approx(0.5)
    assert summary["count_mae"] == pytest.approx(20.0)
    assert summary["signed_count_bias"] == pytest.approx(-20.0)
    assert summary["target_count_max"] == 340


def test_final_detection_matching_prioritizes_confidence() -> None:
    from coffee_detector.analysis.coffee_fg_diagnostics import (
        _confidence_ordered_match,
    )

    predictions = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.5, 0.5, 9.5, 9.5],
        ]
    )
    confidence = torch.tensor([0.2, 0.9])
    targets = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

    matches = _confidence_ordered_match(predictions, confidence, targets, 0.5)
    assert matches[0][0] == 1


def test_diagnostic_selects_pyramid_before_refiner() -> None:
    template = {
        "candidate_counts": [50, 500],
        "branches": {
            "one2one": {
                "500": {
                    "proposal_accessibility": 0.91,
                    "oracle_class_accuracy_headroom": 0.05,
                }
            }
        },
    }
    p3 = template
    p2 = {
        **template,
        "branches": {
            "one2one": {
                "500": {
                    "proposal_accessibility": 0.94,
                    "oracle_class_accuracy_headroom": 0.04,
                }
            }
        },
    }
    decision = compare_p3_p2_diagnostics(p3, p2)
    assert decision["recommended_foundation"] == "D1"
    assert decision["recommended_refiners"] == ["R2", "R3"]
    assert decision["classification_refinement_rational"] is True
