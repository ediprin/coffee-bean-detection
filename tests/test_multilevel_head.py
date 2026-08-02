from pathlib import Path

import torch

from coffee_detector.multilevel_head.model import (
    CapacityMatchedROIClassifier,
    MultilevelHeadConfig,
    MultilevelResidualDetectHead,
    inject_multilevel_head,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def test_capacity_control_and_fusion_have_identical_state_schema() -> None:
    control = CapacityMatchedROIClassifier(
        (64, 128, 256),
        21,
        MultilevelHeadConfig(mode="p5_control", topk=8),
    )
    fusion = CapacityMatchedROIClassifier(
        (64, 128, 256),
        21,
        MultilevelHeadConfig(mode="pyramid_fusion", topk=8),
    )
    assert sum(parameter.numel() for parameter in control.parameters()) == sum(
        parameter.numel() for parameter in fusion.parameters()
    )
    assert {
        key: tuple(value.shape) for key, value in control.state_dict().items()
    } == {key: tuple(value.shape) for key, value in fusion.state_dict().items()}


def test_capacity_control_and_fusion_use_different_parameter_free_context() -> None:
    torch.manual_seed(42)
    control = CapacityMatchedROIClassifier(
        (64, 128, 256),
        5,
        MultilevelHeadConfig(mode="p5_control", topk=4),
    )
    torch.manual_seed(42)
    fusion = CapacityMatchedROIClassifier(
        (64, 128, 256),
        5,
        MultilevelHeadConfig(mode="pyramid_fusion", topk=4),
    )
    features = [
        torch.randn(1, 64, 16, 16),
        torch.randn(1, 128, 8, 8),
        torch.randn(1, 256, 4, 4),
    ]
    rois = torch.tensor([[0.0, 8.0, 8.0, 90.0, 100.0]])
    control_logits = control(features, rois, (8.0, 16.0, 32.0))
    fusion_logits = fusion(features, rois, (8.0, 16.0, 32.0))
    assert control_logits.shape == fusion_logits.shape == (1, 5)
    assert not torch.equal(control_logits, fusion_logits)


def test_injection_preserves_native_head_and_yolo_output_contract() -> None:
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel(str(MODEL_YAML), nc=5, verbose=False)
    native = model.model[-1]
    box_ids = [id(branch) for branch in native.cv2]
    class_ids = [id(branch) for branch in native.cv3]
    assert inject_multilevel_head(
        model,
        MultilevelHeadConfig(
            mode="pyramid_fusion", topk=4, inference_weight=0.0
        ),
    ) == 1
    head = model.model[-1]
    assert isinstance(head, MultilevelResidualDetectHead)
    assert [id(branch) for branch in head.base_head.cv2] == box_ids
    assert [id(branch) for branch in head.base_head.cv3] == class_ids
    assert inject_multilevel_head(model, head.config) == 0

    image = torch.randn(1, 3, 128, 128)
    model.train()
    train_output = model(image)
    assert set(train_output) == {"one2many", "one2one"}
    model.eval()
    with torch.inference_mode():
        final, raw = model(image)
    assert final.ndim == 3
    assert set(raw) == {"one2many", "one2one"}
    assert set(raw["one2one"]) == {"boxes", "scores", "feats"}
