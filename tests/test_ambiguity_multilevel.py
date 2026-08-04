import inspect
from pathlib import Path

import torch

from coffee_detector.ambiguity_multilevel.audit import (
    audit_ambiguity_multilevel_static,
    static_ambiguity_multilevel_audit,
)
from coffee_detector.ambiguity_multilevel.model import (
    AmbiguityMultilevelConfig,
    AmbiguityMultilevelDetectionModel,
    AmbiguityMultilevelDetectHead,
    load_ambiguity_multilevel_detector_weights,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"


def _models(nc: int = 5):
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=nc, verbose=False).eval()
    candidate = AmbiguityMultilevelDetectionModel(
        str(MODEL_YAML), nc=nc, verbose=False, ambiguity_multilevel=AmbiguityMultilevelConfig(hidden_dim=16)
    ).eval()
    load_ambiguity_multilevel_detector_weights(candidate, source)
    return source, candidate


def test_acmc_is_field_level_and_preserves_d0_before_learning() -> None:
    source, candidate = _models()
    image = torch.randn(1, 3, 128, 128)
    with torch.inference_mode():
        source_output = source(image)
        candidate_output = candidate(image)
    head = candidate.model[-1]
    assert isinstance(head, AmbiguityMultilevelDetectHead)
    assert torch.allclose(candidate_output[0], source_output[0], rtol=0.0, atol=1e-7)
    # Ultralytics validation runs loss computation while the model is in eval
    # mode, so an unfused ACMC head must retain both native branches here.
    assert set(candidate_output[1]) == {"one2many", "one2one"}
    assert torch.equal(candidate_output[1]["one2one"]["boxes"], source_output[1]["one2one"]["boxes"])
    assert torch.equal(candidate_output[1]["one2one"]["scores"], source_output[1]["one2one"]["scores"])
    source_code = inspect.getsource(type(head)) + inspect.getsource(type(head.correction))
    assert "roi_align" not in source_code
    assert ".topk(" not in source_code
    assert "_get_decode_boxes" not in source_code


def test_acmc_trains_correction_through_native_detection_scores() -> None:
    _, candidate = _models()
    candidate.train()
    output = candidate(torch.randn(1, 3, 128, 128))
    assert set(output) == {"one2many", "one2one"}
    loss = output["one2many"]["scores"].square().mean()
    loss.backward()
    head = candidate.model[-1]
    assert head.correction.class_corrections[0].weight.grad is not None
    assert head.base_head.cv2[0][-1].weight.grad is None
    assert head.base_head.cv3[0][-1].weight.grad is not None


def test_acmc_fused_inference_uses_only_one_to_one_branch() -> None:
    _, candidate = _models()
    head = candidate.model[-1]
    assert isinstance(head, AmbiguityMultilevelDetectHead)
    head.fuse()
    candidate.eval()
    with torch.inference_mode():
        detections, predictions = candidate(torch.randn(1, 3, 128, 128))
    assert detections.shape[0] == 1
    assert set(predictions) == {"one2one"}
    assert predictions["one2one"]["boxes"].shape[0] == 1


def test_static_audit_rejects_roi_style_execution() -> None:
    result = audit_ambiguity_multilevel_static(MODEL_YAML, num_classes=5, image_size=128, config={"hidden_dim": 16})
    assert result["training_executed"] is False
    assert result["test_images_accessed"] is False
    assert result["native_head_state_identical"]
    assert result["initial_detection_identical"]
    assert result["box_tensor_identical"]
    assert result["score_tensor_identical"]
    assert not result["has_roi_align"]
    assert not result["has_topk"]
    assert not result["has_box_decode"]


def test_checkpoint_static_audit_proves_score_only_active_correction(tmp_path: Path) -> None:
    from ultralytics.nn.tasks import DetectionModel

    source = DetectionModel(str(MODEL_YAML), nc=5, verbose=False)
    checkpoint = tmp_path / "d0.pt"
    torch.save({"model": source}, checkpoint)
    result = static_ambiguity_multilevel_audit(
        MODEL_YAML,
        checkpoint,
        tmp_path / "audit.json",
        nc=5,
        image_size=128,
        config={"hidden_dim": 16},
    )
    assert result["decision"] == "PASS"
    assert result["gates"]["active_correction_changes_scores"]
    assert result["gates"]["active_correction_preserves_boxes"]
