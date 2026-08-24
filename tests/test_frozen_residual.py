from pathlib import Path

import pytest
import torch

from coffee_detector.experiments.run_faruq_v3_frozen_residual import (
    run_faruq_v3_frozen_residual,
)
from coffee_detector.frozen_residual.model import (
    FrozenResidualConfig,
    FrozenResidualDetectionModel,
    freeze_native_detector,
    load_frozen_d0_weights,
)
from coffee_detector.multilevel_head.model import _expand_and_clip_boxes
from coffee_detector.train import load_experiment


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    native = DetectionModel(str(MODEL), nc=nc, verbose=False).eval()
    candidate = FrozenResidualDetectionModel(
        str(MODEL),
        nc=nc,
        verbose=False,
        frozen_residual={"topk": 8, "training_topk": 8},
    )
    load_frozen_d0_weights(candidate, native)
    return native, candidate


def test_zero_initialized_candidate_preserves_native_d0() -> None:
    native, candidate = _models()
    candidate.eval()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        native_output = native(image)
        candidate_output = candidate(image)
    assert torch.allclose(candidate_output[0], native_output[0], rtol=0.0, atol=1e-7)
    assert torch.equal(
        candidate_output[1]["one2one"]["boxes"],
        native_output[1]["one2one"]["boxes"],
    )
    assert torch.equal(
        candidate_output[1]["one2one"]["scores"],
        native_output[1]["one2one"]["scores"],
    )


def test_frozen_candidate_trains_only_refiner_and_gate_with_bn_frozen() -> None:
    _, candidate = _models()
    counts = freeze_native_detector(candidate)
    assert 0 < counts["trainable"] < counts["total"]
    candidate.train()
    trainable = [
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    ]
    assert trainable
    assert all(".refiner." in name or ".gate." in name for name in trainable)
    native_bn = [
        module
        for module in candidate.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and not (
            module in set(candidate.model[-1].refiner.modules())
            or module in set(candidate.model[-1].gate.modules())
        )
    ]
    assert native_bn and all(not module.training for module in native_bn)


def test_frozen_residual_loss_matches_predicted_box_and_has_no_native_gradients() -> None:
    _, candidate = _models()
    candidate.train()
    candidate.args = type(
        "Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 2}
    )()
    image = torch.randn(1, 3, 128, 128)
    predictions = candidate(image)
    source = predictions["one2one"]
    confidence = source["scores"].detach().sigmoid().amax(dim=1)[0]
    index = int(confidence.argmax())
    decoded = candidate.model[-1].base_head._get_decode_boxes(source).transpose(1, 2)
    box = _expand_and_clip_boxes(
        decoded[:, index : index + 1],
        image_height=128,
        image_width=128,
        factor=1.0,
    )[0, 0]
    centre = (box[:2] + box[2:]) * 0.5 / 128.0
    size = (box[2:] - box[:2]) / 128.0
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[2.0]]),
        "bboxes": torch.cat((centre, size)).reshape(1, 4),
    }
    loss, items = candidate.loss(batch, predictions)
    assert loss.shape == items.shape == (4,)
    assert candidate.criterion.last_matches == 1
    loss.sum().backward()
    head = candidate.model[-1]
    assert head.refiner.classifier.weight.grad is not None
    assert head.gate.linear.weight.grad is not None
    assert all(
        parameter.grad is None
        for parameter in head.base_head.parameters()
    )


def test_frozen_residual_config_and_runner_lock(tmp_path: Path) -> None:
    config = load_experiment(
        ROOT / "configs/frozen_residual/FRM1_yolo26n_frozen_multilevel.yaml"
    )
    assert config["variant"] == "frozen_residual"
    assert config["train"]["epochs"] == 10
    assert FrozenResidualConfig.from_mapping(config["frozen_residual"]).descriptor_dim == 512
    with pytest.raises(RuntimeError, match="belum diotorisasi"):
        run_faruq_v3_frozen_residual(
            tmp_path / "data",
            tmp_path / "grouped.json",
            tmp_path / "baseline.json",
            tmp_path / "d0.pt",
            tmp_path / "static.json",
            tmp_path / "output",
        )
    with pytest.raises(ValueError, match="seed 42"):
        run_faruq_v3_frozen_residual(
            tmp_path / "data",
            tmp_path / "grouped.json",
            tmp_path / "baseline.json",
            tmp_path / "d0.pt",
            tmp_path / "static.json",
            tmp_path / "output",
            seed=123,
            authorize_training=True,
        )
