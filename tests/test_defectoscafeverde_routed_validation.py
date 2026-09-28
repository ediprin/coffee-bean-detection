import numpy as np
import torch
from pathlib import Path

from coffee_detector.analysis.defectoscafeverde_routed_validation import (
    decide_routed_screen,
    route_predictions,
)


def test_decision_requires_calibration_and_frozen_route():
    baseline = {"macro_map50_95": 0.90, "bottom3_class_map50_95": 0.80, "worst_class_map50_95": 0.75}
    results = {
        "D0DIRECT": {**baseline},
        "ROUTED": {"macro_map50_95": 0.906, "bottom3_class_map50_95": 0.801, "worst_class_map50_95": 0.748, "classes_without_ground_truth": []},
    }
    decision = decide_routed_screen(results, baseline)
    assert decision["decision"] == "RETAIN_ROUTED_CANDIDATE"
    bad = decide_routed_screen(results, {**baseline, "macro_map50_95": 0.89})
    assert bad["decision"] == "INVALID_ENDPOINT_CALIBRATION"


def test_route_predictions_can_select_associated_expert(monkeypatch):
    import coffee_detector.analysis.defectoscafeverde_routed_validation as module

    router = {"subset": ["D0DIRECT", "RAFC1"]}
    monkeypatch.setattr(module, "predict_final_router", lambda router, record: "RAFC1")
    image = torch.zeros(1, 3, 16, 16)
    predictions = {
        "D0DIRECT": (torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([0]), torch.tensor([0.8])),
        "RAFC1": (torch.tensor([[1.0, 1.0, 10.0, 10.0]]), torch.tensor([1]), torch.tensor([0.9])),
    }
    routed, choices = route_predictions(image, predictions, router)
    assert routed.shape == (1, 6)
    assert int(routed[0, 5]) == 1
    assert np.isclose(float(routed[0, 4]), 0.9)
    assert choices == {"RAFC1": 1}


def test_notebook_is_rgb_native_train_gated_validation_only():
    source = Path("notebooks/DefectosCafeVerde_Routed_Validation_Screen_Colab.ipynb").read_text(encoding="utf-8")
    assert "v2-rgb-native" in source
    assert "AUTHORIZE_SEPARATE_ROUTED_ARCHITECTURE_SCREEN" in source
    assert "first=='val'" in source
    assert "TRAIN/TEST TEREXPOSE" in source
    assert "authorize-training" not in source
    assert "TORCH PRE-INSTALL" in source

