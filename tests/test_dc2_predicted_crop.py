import json
from pathlib import Path

from PIL import Image
import torch

from coffee_detector.dataset import Box
from coffee_detector.dc2_crop.predicted import (
    MatchedRawObjectCropDataset,
    PredictedCropRecord,
    greedy_match_xyxy,
    xyxy_iou,
)
from coffee_detector.experiments.run_faruq_v3_dc2_predicted_crop_screening import (
    decide_dc2_predicted,
)


def test_xyxy_iou_and_greedy_matching_are_class_agnostic_and_unique() -> None:
    assert xyxy_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    predictions = [(0, 0, 10, 10), (1, 1, 9, 9), (20, 20, 30, 30)]
    scores = [0.5, 0.9, 0.8]
    targets = [(0, 0, 10, 10), (20, 20, 30, 30)]
    matches = greedy_match_xyxy(predictions, scores, targets, iou_threshold=0.5)
    assert {(pred, target) for pred, target, _ in matches} == {(0, 0), (2, 1)}
    assert len({pred for pred, _, _ in matches}) == len(matches)
    assert len({target for _, target, _ in matches}) == len(matches)


def test_matched_raw_crop_dataset_switches_between_gt_and_predicted_pixels(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (100, 100), color=(0, 0, 0))
    for x in range(15, 35):
        for y in range(40, 60):
            image.putpixel((x, y), (255, 0, 0))
    for x in range(65, 85):
        for y in range(40, 60):
            image.putpixel((x, y), (0, 255, 0))
    image.save(image_path)
    record = PredictedCropRecord(
        image_path=image_path,
        class_id=4,
        gt_box=Box(class_id=4, x_center=0.25, y_center=0.50, width=0.20, height=0.20),
        predicted_xyxy=(65.0, 40.0, 85.0, 60.0),
        predicted_class_id=7,
        predicted_confidence=0.8,
        matched_iou=0.7,
    )
    gt_dataset = MatchedRawObjectCropDataset(
        [record], 32, training=False, source="gt", context=1.0
    )
    predicted_dataset = MatchedRawObjectCropDataset(
        [record], 32, training=False, source="predicted", context=1.0
    )
    gt_tensor, gt_label = gt_dataset[0]
    predicted_tensor, predicted_label = predicted_dataset[0]
    assert gt_label == predicted_label == 4
    assert gt_tensor.shape == predicted_tensor.shape == (3, 32, 32)
    gt_means = gt_tensor.mean(dim=(1, 2))
    predicted_means = predicted_tensor.mean(dim=(1, 2))
    assert gt_means[0] > gt_means[1]
    assert predicted_means[1] > predicted_means[0]


def test_dc2_predicted_gate_requires_coverage_gain_and_retention() -> None:
    passed = decide_dc2_predicted(
        train_coverage=0.95,
        val_coverage=0.94,
        detector_metrics={"macro_f1": 0.80, "bottom3_f1": 0.60, "worst_f1": 0.50},
        predicted_metrics={"macro_f1": 0.83, "bottom3_f1": 0.62, "worst_f1": 0.49},
        gt_matched_metrics={"macro_f1": 0.86, "bottom3_f1": 0.70, "worst_f1": 0.60},
    )
    assert passed["decision"] == "PASS"
    assert all(passed["criteria"].values())

    failed = decide_dc2_predicted(
        train_coverage=0.95,
        val_coverage=0.89,
        detector_metrics={"macro_f1": 0.80, "bottom3_f1": 0.60, "worst_f1": 0.50},
        predicted_metrics={"macro_f1": 0.805, "bottom3_f1": 0.61, "worst_f1": 0.49},
        gt_matched_metrics={"macro_f1": 0.86, "bottom3_f1": 0.70, "worst_f1": 0.60},
    )
    assert failed["decision"] == "FAIL"
    assert not failed["criteria"]["val_matched_coverage_at_least_90_percent"]
    assert not failed["criteria"]["predicted_local_macro_gain_vs_native_at_least_1_point"]


def test_dc2_predicted_notebook_is_val_only_and_points_to_branch() -> None:
    notebook = Path("notebooks/Faruq_V3_DC2_Predicted_Raw_Crop_Screening_Colab.ipynb")
    assert notebook.is_file()
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "agent/dc2-predicted-raw-crop-screening" in source
    assert "run_faruq_v3_dc2_predicted_crop_screening" in source
    assert "--authorize-training" in source
    assert "test/" not in source
    assert "split=test" not in source.lower()
