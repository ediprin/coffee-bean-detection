from pathlib import Path

import pytest
import torch

from coffee_detector.coffee_fg.model import (
    CoffeeFGConfig,
    CoffeeFGDetectHead,
    CoffeeFGDetectionModel,
    MultiLevelROIRefiner,
    inject_coffee_fg,
)
from coffee_detector.analysis.coffee_fg_diagnostics import (
    _count_summary,
    _greedy_match,
    compare_p3_p2_diagnostics,
)
from coffee_detector.evaluate import _classwise_summary
from coffee_detector.experiments.run_coffee_fg_screening import (
    run_coffee_fg_screening,
)
from coffee_detector.train import load_experiment


ROOT = Path(__file__).resolve().parents[1]
P2_MODEL = ROOT / "configs/coffee_fg/models/yolo26n-p2.yaml"


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
