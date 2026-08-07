from pathlib import Path

import pytest
import torch

from coffee_detector.experiments.run_faruq_v3_multilevel_head import (
    run_faruq_v3_multilevel_head,
)
from coffee_detector.multilevel_head.model import MultilevelHeadDetectionModel
from coffee_detector.train import load_experiment


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_multilevel_loss_preserves_native_loss_and_trains_refiner() -> None:
    model = MultilevelHeadDetectionModel(
        str(MODEL),
        nc=5,
        verbose=False,
        multilevel_head={
            "mode": "pyramid_fusion",
            "topk": 8,
            "training_topk": 8,
            "predicted_start_epoch": 1,
            "predicted_full_epoch": 2,
        },
    )
    model.args = type(
        "Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 2}
    )()
    model.train()
    batch = {
        "img": torch.randn(2, 3, 128, 128),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.3, 0.3], [0.45, 0.55, 0.25, 0.35]]
        ),
    }
    predictions = model(batch["img"])
    loss, items = model.loss(batch, predictions)
    assert loss.shape == (4,)
    assert items.shape == (4,)
    loss.sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.model[-1].refiner.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_multilevel_configs_are_capacity_matched() -> None:
    control = load_experiment(
        ROOT / "configs/multilevel_head/MHC0_yolo26n_p5_control.yaml"
    )
    fusion = load_experiment(
        ROOT / "configs/multilevel_head/MHF1_yolo26n_pyramid_fusion.yaml"
    )
    assert control["variant"] == fusion["variant"] == "multilevel_head"
    assert control["train"] == fusion["train"]
    left = dict(control["multilevel_head"])
    right = dict(fusion["multilevel_head"])
    assert left.pop("mode") == "p5_control"
    assert right.pop("mode") == "pyramid_fusion"
    assert left == right


def test_multilevel_runner_requires_explicit_one_seed_authorization(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="belum diotorisasi"):
        run_faruq_v3_multilevel_head(
            tmp_path / "data",
            tmp_path / "grouped.json",
            tmp_path / "baseline.json",
            tmp_path / "static.json",
            tmp_path / "output",
            authorize_training=False,
        )
    with pytest.raises(ValueError, match="seed 42"):
        run_faruq_v3_multilevel_head(
            tmp_path / "data",
            tmp_path / "grouped.json",
            tmp_path / "baseline.json",
            tmp_path / "static.json",
            tmp_path / "output",
            seed=123,
            authorize_training=True,
        )
