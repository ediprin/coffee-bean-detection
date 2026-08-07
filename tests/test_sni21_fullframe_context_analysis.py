import numpy as np

from coffee_detector.analyze_sni21_fullframe_context import (
    _bootstrap,
    _exact_mcnemar,
    corrected_record_diagnosis,
)


def test_corrected_record_uses_highest_confidence_localized_candidate() -> None:
    record = {
        "image": "sample.jpg",
        "ground_truth": [{"class_id": 0, "xyxy": [0, 0, 10, 10]}],
        "predictions": [
            {"class_id": 1, "confidence": 0.2, "xyxy": [0, 0, 10, 10]},
            {"class_id": 0, "confidence": 0.8, "xyxy": [0.2, 0.2, 9.8, 9.8]},
        ],
    }

    result = corrected_record_diagnosis(record, 0.5)

    assert result["state"] == "correct"
    assert result["predicted_class"] == 0
    assert result["confidence"] == 0.8


def test_bootstrap_and_mcnemar_capture_paired_harm() -> None:
    paired = []
    for class_id in range(3):
        for index in range(10):
            paired.append(
                {
                    "class_id": class_id,
                    "fc1_correct": 1,
                    "fc2_correct": int(index == 0),
                    "fc1_proposal": 1,
                    "fc2_proposal": 1,
                }
            )

    result = _bootstrap(paired, iterations=500, seed=42)
    mcnemar = _exact_mcnemar(harmed=27, rescued=0)

    assert np.isclose(result["macro_top1_accuracy_delta"]["point"], -0.9)
    assert result["macro_top1_accuracy_delta"]["ci95"][1] < 0
    assert mcnemar["p_two_sided"] < 0.05
