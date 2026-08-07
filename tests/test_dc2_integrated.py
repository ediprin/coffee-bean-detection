import inspect
import json
from pathlib import Path

from PIL import Image

from coffee_detector.dc2_crop.integrated import (
    GroundTruthRecord,
    IntegratedPredictedCropDataset,
    IntegratedPredictionRecord,
    collect_all_detector_predictions,
    decide_dc2_integrated,
    detection_map_summary,
)


def _prediction(image: Path, class_id: int, score: float = 0.9):
    return IntegratedPredictionRecord(
        image_path=image,
        predicted_xyxy=(10.0, 10.0, 30.0, 30.0),
        predicted_class_id=class_id,
        predicted_confidence=score,
    )


def _target(image: Path, class_id: int):
    return GroundTruthRecord(
        image_path=image,
        class_id=class_id,
        xyxy=(10.0, 10.0, 30.0, 30.0),
    )


def test_detection_map_is_one_for_perfect_two_class_predictions(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    predictions = [_prediction(image_a, 0), _prediction(image_b, 1)]
    targets = [_target(image_a, 0), _target(image_b, 1)]
    metrics = detection_map_summary(predictions, targets, 2)
    assert abs(metrics["map50_95"] - 1.0) < 1e-9
    assert abs(metrics["map50"] - 1.0) < 1e-9
    assert abs(metrics["worst_ap50_95"] - 1.0) < 1e-9


def test_detection_map_penalizes_wrong_fine_grained_class(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    predictions = [_prediction(image_a, 1), _prediction(image_b, 0)]
    targets = [_target(image_a, 0), _target(image_b, 1)]
    metrics = detection_map_summary(predictions, targets, 2)
    assert metrics["map50_95"] == 0.0
    assert metrics["map50"] == 0.0


def test_integrated_crop_reads_predicted_raw_rgb_region(tmp_path: Path) -> None:
    image_path = tmp_path / "bean.jpg"
    Image.new("RGB", (64, 64), color=(100, 120, 140)).save(image_path)
    dataset = IntegratedPredictedCropDataset([_prediction(image_path, 0)], 32)
    tensor = dataset[0]
    assert tuple(tensor.shape) == (3, 32, 32)


def test_integrated_gate_is_detection_level_and_paired() -> None:
    native = {"map50_95": 0.80, "bottom3_ap50_95": 0.60, "worst_ap50_95": 0.50}
    refined = {"map50_95": 0.81, "bottom3_ap50_95": 0.61, "worst_ap50_95": 0.50}
    result = decide_dc2_integrated(native, refined)
    assert result["decision"] == "PASS"
    assert all(result["criteria"].values())

    failed = decide_dc2_integrated(
        native,
        {"map50_95": 0.803, "bottom3_ap50_95": 0.61, "worst_ap50_95": 0.50},
    )
    assert failed["decision"] == "FAIL"


def test_all_prediction_collection_is_not_gt_matched_and_uses_parent_like_nms() -> None:
    source = inspect.getsource(collect_all_detector_predictions)
    assert "agnostic_nms" in source
    assert "greedy_match" not in source
    assert "match_iou" not in source


def test_integrated_notebook_is_val_only_and_points_to_branch() -> None:
    notebook = Path("notebooks/Faruq_V3_DC2_Integrated_Inference_Screening_Colab.ipynb")
    assert notebook.is_file()
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in payload.get("cells", [])
    )
    assert "agent/dc2-integrated-inference-screening" in source
    assert "run_faruq_v3_dc2_integrated_inference" in source
    assert "--authorize-evaluation" in source
    assert "split=test" not in source.lower()
    assert "test/" not in source.lower()
