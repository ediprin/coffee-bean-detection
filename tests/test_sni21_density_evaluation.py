import numpy as np
import pytest

from coffee_detector.run_sni21_density_evaluation import (
    _pairwise_iou,
    diagnose_image,
)


def test_pairwise_iou_handles_empty_and_overlap() -> None:
    boxes = np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    target = np.asarray([[0, 0, 10, 10]], dtype=float)

    values = _pairwise_iou(boxes, target)

    assert values.shape == (2, 1)
    assert values[:, 0].tolist() == pytest.approx([1.0, 0.0])
    assert _pairwise_iou(np.empty((0, 4)), target).shape == (0, 1)


def test_diagnose_image_separates_proposal_and_class_failures() -> None:
    ground_truth_classes = np.asarray([0, 1, 2])
    ground_truth_boxes = np.asarray(
        [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ],
        dtype=float,
    )
    prediction_classes = np.asarray([0, 0, 3])
    prediction_boxes = np.asarray(
        [
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [70, 70, 80, 80],
        ],
        dtype=float,
    )

    result = diagnose_image(
        ground_truth_classes,
        ground_truth_boxes,
        prediction_classes,
        prediction_boxes,
        iou_threshold=0.5,
        max_det=3,
    )

    assert result["proposal_accessible"] == 2
    assert result["localized_correct"] == 1
    assert result["localized_wrong_class"] == 1
    assert result["proposal_miss"] == 1
    assert result["unlocalized_prediction_count"] == 1
    assert result["duplicate_candidate_count"] == 0
    assert result["saturated_max_det"] is True


def test_diagnose_image_counts_duplicate_candidates() -> None:
    result = diagnose_image(
        np.asarray([0]),
        np.asarray([[0, 0, 10, 10]], dtype=float),
        np.asarray([0, 0]),
        np.asarray(
            [[0, 0, 10, 10], [0.5, 0.5, 9.5, 9.5]], dtype=float
        ),
        iou_threshold=0.5,
        max_det=300,
    )

    assert result["proposal_accessible"] == 1
    assert result["duplicate_candidate_count"] == 1
    assert result["localized_correct"] == 1
