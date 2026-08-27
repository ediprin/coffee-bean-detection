import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from coffee_detector.analysis.af2_box_score_factorial import (
    _decision,
    _match_predictions,
    _postprocess_branch,
    _validation_loader,
    combine_branch,
    summarize_detection_stats,
)


def test_combine_branch_preserves_sources_and_requires_aligned_anchors() -> None:
    left = {
        "boxes": torch.randn(1, 8, 20),
        "scores": torch.randn(1, 21, 20),
        "feats": [torch.randn(1, 8, 4, 5)],
    }
    right = {
        "boxes": torch.randn(1, 8, 20),
        "scores": torch.randn(1, 21, 20),
        "feats": [torch.randn(1, 8, 4, 5)],
    }
    hybrid = combine_branch(left, right)
    assert hybrid["boxes"] is left["boxes"]
    assert hybrid["scores"] is right["scores"]
    assert hybrid["feats"] is left["feats"]
    with pytest.raises(ValueError, match="anchor/grid"):
        combine_branch(
            left,
            {"boxes": right["boxes"], "scores": torch.randn(1, 21, 19), "feats": right["feats"]},
        )


def test_match_predictions_is_class_aware_and_one_to_one() -> None:
    predicted_boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=torch.float32)
    target_boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    matched = _match_predictions(
        torch.tensor([1, 0]),
        torch.tensor([1]),
        predicted_boxes,
        target_boxes,
        torch.tensor([0.5, 0.95]),
    )
    assert matched.tolist() == [[True, True], [False, False]]


def test_pure_branch_postprocess_is_bitwise_native_yolo26_endpoint() -> None:
    from ultralytics.nn.modules.head import Detect

    torch.manual_seed(42)
    head = Detect(nc=2, end2end=True, ch=(8, 16, 32)).eval()
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    features = [
        torch.rand(1, 8, 8, 8),
        torch.rand(1, 16, 4, 4),
        torch.rand(1, 32, 2, 2),
    ]
    with torch.inference_mode():
        native, raw = head(features)
        rebuilt = _postprocess_branch(head, raw["one2one"], max_det=500)
    assert torch.equal(native, rebuilt)


def test_detection_summary_has_exact_perfect_ap() -> None:
    summary = summarize_detection_stats(
        np.ones((2, 10), dtype=bool),
        np.asarray([0.9, 0.8]),
        np.asarray([0, 1]),
        np.asarray([0, 1]),
        {0: "a", 1: "b"},
    )
    assert summary["macro_map50_95"] == pytest.approx(0.995, abs=1e-9)
    assert summary["bottom3_class_map50_95"] == pytest.approx(0.995, abs=1e-9)
    assert summary["classes_without_ground_truth"] == []


def test_validation_loader_matches_rectangular_validator_contract(tmp_path: Path) -> None:
    import cv2

    image_root = tmp_path / "val/images"
    label_root = tmp_path / "val/labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    # OpenCV writes BGR; the Ultralytics Format transform must return RGB CHW.
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    cv2.imwrite(str(image_root / "sample.jpg"), image)
    (label_root / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {"path": str(tmp_path), "train": "val/images", "val": "val/images", "names": {0: "bean"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    class FakeYOLO:
        overrides = {"task": "detect"}

    dataset, loader = _validation_loader(
        FakeYOLO(), data_yaml, image_size=64, batch_size=1, workers=0
    )
    batch = next(iter(loader))
    assert len(dataset) == 1
    assert tuple(batch["img"].shape[:2]) == (1, 3)
    assert batch["img"][0, 0].float().mean() > batch["img"][0, 2].float().mean()
    assert batch["cls"].reshape(-1).tolist() == [0.0]


def test_decision_supports_raw_boxes_only_when_all_locked_tail_gates_pass() -> None:
    rows = {
        "DD": {metric: 0.80 for metric in ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")},
        "AD": {metric: 0.79 for metric in ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")},
        "AA": {"macro_map50_95": 0.88, "bottom3_class_map50_95": 0.80, "worst_class_map50_95": 0.78},
        "DA": {"macro_map50_95": 0.879, "bottom3_class_map50_95": 0.81, "worst_class_map50_95": 0.79},
    }
    comparison, decision, next_action = _decision(rows)
    assert decision == "SUPPORT_D0FT_BOX_AF2_SCORE_ARCHITECTURE"
    assert next_action == "IMPLEMENT_DECOUPLED_AF2_CLASSIFICATION_BRANCH"
    assert all(comparison["criteria"].values())


def test_factorial_notebook_is_validation_only_and_compiles() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks/Faruq_V3_AF2_Box_Score_Factorial_Colab.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
    assert "codex/af2-box-score-factorial" in source
    assert "D0FT_seed42/weights/best.pt" in source
    assert "AF2_seed42/weights/best.pt" in source
    assert "af2_igem_paired_confirmation.json" in source
    assert "--reference-summary" in source
    assert "--authorize-training" not in source
    assert "split=test" not in source.lower()
    assert "test tidak boleh tersedia" in source.lower()
