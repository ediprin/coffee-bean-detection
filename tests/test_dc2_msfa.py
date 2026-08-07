import json
from pathlib import Path

import numpy as np
import torch

from coffee_detector.dc2_crop.msfa import (
    DC2MSFAClassifier,
    MatchedCropGlobalDataset,
    record_signature,
)
from coffee_detector.dc2_crop.predicted import PredictedCropRecord
from coffee_detector.dataset import Box
from coffee_detector.experiments.run_faruq_v3_dc2_msfa_screening import decide_dc2_msfa


def _record(image_path: Path, class_id: int = 2) -> PredictedCropRecord:
    return PredictedCropRecord(
        image_path=image_path,
        class_id=class_id,
        gt_box=Box(class_id=class_id, x_center=0.5, y_center=0.5, width=0.4, height=0.4),
        predicted_xyxy=(10.0, 10.0, 30.0, 30.0),
        predicted_class_id=class_id,
        predicted_confidence=0.8,
        matched_iou=0.75,
    )


def test_zero_initialized_msfa_preserves_local_logits_bitwise() -> None:
    torch.manual_seed(1)
    model = DC2MSFAClassifier(21, global_dim=17, imagenet_pretrained=False).eval()
    image = torch.randn(2, 3, 64, 64)
    global_descriptor = torch.randn(2, 17)
    with torch.inference_mode():
        local = model(image, global_descriptor, enable_global=False)
        fused = model(image, global_descriptor, enable_global=True)
    assert torch.equal(local, fused)
    assert torch.count_nonzero(model.global_projection.weight) == 0
    assert torch.count_nonzero(model.global_projection.bias) == 0


def test_active_msfa_projection_can_change_classification_logits() -> None:
    torch.manual_seed(2)
    model = DC2MSFAClassifier(5, global_dim=7, imagenet_pretrained=False).eval()
    image = torch.randn(1, 3, 64, 64)
    global_descriptor = torch.randn(1, 7)
    with torch.inference_mode():
        baseline = model(image, global_descriptor, enable_global=False)
        model.global_projection.weight.fill_(0.05)
        changed = model(image, global_descriptor, enable_global=True)
    assert not torch.equal(baseline, changed)


def test_record_signature_changes_with_predicted_geometry(tmp_path: Path) -> None:
    first = _record(tmp_path / "a.jpg")
    second = PredictedCropRecord(
        image_path=first.image_path,
        class_id=first.class_id,
        gt_box=first.gt_box,
        predicted_xyxy=(11.0, 10.0, 30.0, 30.0),
        predicted_class_id=first.predicted_class_id,
        predicted_confidence=first.predicted_confidence,
        matched_iou=first.matched_iou,
    )
    assert record_signature([first]) != record_signature([second])


def test_matched_crop_global_dataset_requires_aligned_descriptor_count(tmp_path: Path) -> None:
    record = _record(tmp_path / "a.jpg")
    try:
        MatchedCropGlobalDataset([record], np.zeros((2, 4), dtype=np.float32), 64, training=False)
    except ValueError as error:
        assert "tidak sejajar" in str(error)
    else:
        raise AssertionError("Mismatch descriptor count harus ditolak")


def test_msfa_gate_compares_against_optimization_matched_local_control() -> None:
    local_ft = {"metrics": {"macro_f1": 0.80, "bottom3_f1": 0.60, "worst_f1": 0.50}}
    msfa = {"metrics": {"macro_f1": 0.81, "bottom3_f1": 0.61, "worst_f1": 0.50}}
    result = decide_dc2_msfa(local_ft, msfa)
    assert result["decision"] == "PASS"
    assert all(result["criteria"].values())

    fail = decide_dc2_msfa(
        local_ft,
        {"metrics": {"macro_f1": 0.803, "bottom3_f1": 0.61, "worst_f1": 0.50}},
    )
    assert fail["decision"] == "FAIL"


def test_msfa_notebook_is_val_only_and_points_to_branch() -> None:
    notebook = Path("notebooks/Faruq_V3_DC2_MSFA_Global_Local_Screening_Colab.ipynb")
    assert notebook.is_file()
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "agent/dc2-msfa-global-local-screening" in source
    assert "run_faruq_v3_dc2_msfa_screening" in source
    assert "--authorize-training" in source
    assert "split=test" not in source.lower()
