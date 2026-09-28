import numpy as np
from pathlib import Path

from coffee_detector.analysis.defectoscafeverde_error_complementarity import MODEL_ORDER
from coffee_detector.analysis.defectoscafeverde_train_routability import (
    _native_rgb_letterbox_sample,
    build_router_features,
    fit_final_router,
    predict_final_router,
    select_minimal_subset,
    summarize_routability,
)


def _row(image, correct, classes=None, confidences=None):
    classes = classes or {}
    confidences = confidences or {}
    return {
        "image": image,
        "target_index": 0,
        "class_id": 0,
        "class_name": "agrio",
        "anchor_available": True,
        "appearance": [0.1] * 20,
        "models": {
            model: {
                "associated": True,
                "predicted_class_id": classes.get(model, 0),
                "confidence": confidences.get(model, 0.5),
                "anchor_iou": 1.0 if model == "D0DIRECT" else 0.8,
                "correct": model in correct,
            }
            for model in MODEL_ORDER
        },
    }


def test_features_do_not_depend_on_ground_truth_fields():
    row = _row("one.jpg", {"D0DIRECT"})
    first = build_router_features(row, ("D0DIRECT", "RAFC1"), 12)
    row["class_id"] = 11
    row["class_name"] = "triangulo"
    second = build_router_features(row, ("D0DIRECT", "RAFC1"), 12)
    assert np.array_equal(first, second)


def test_minimal_subset_prefers_smallest_pair_retaining_oracle_gain():
    rows = []
    for index in range(20):
        correct = {"D0DIRECT"} if index < 15 else set()
        if index in (15, 16, 17, 18):
            correct.add("RAFC1")
        if index == 19:
            correct.add("AF2DIRECT")
        rows.append(_row(f"{index}.jpg", correct))
    result = select_minimal_subset(rows)
    assert result["subset"] == ["D0DIRECT", "AF2DIRECT", "RAFC1"]
    assert result["full_oracle_gain_objects"] == 5
    assert result["retained_full_gain_fraction"] == 1.0


def test_summary_is_train_only_router_gate_shape():
    rows = []
    for index in range(100):
        correct = {"D0DIRECT"} if index % 10 else {"RAFC1"}
        classes = {model: (1 if model == "RAFC1" and index % 10 == 0 else 0) for model in MODEL_ORDER}
        rows.append(_row(f"{index}.jpg", correct, classes=classes))
    result = summarize_routability(rows, 12)
    assert set(result) == {"minimal_subset", "oof_router", "criteria", "decision", "next"}
    assert result["minimal_subset"]["subset"] == ["D0DIRECT", "RAFC1"]
    assert result["oof_router"]["groups"] == 100
    assert result["criteria"]["associated_full_oracle_gain_at_least_1_point"]


def test_colab_is_train_only_and_has_shared_drive_fallback():
    source = Path("notebooks/DefectosCafeVerde_Train_Routability_Audit_Colab.ipynb").read_text(encoding="utf-8")
    assert "prepare_defectoscafeverde_routability_colab" in source
    assert "run_train_routability_audit" in source
    assert "first=='train'" in source
    assert "VAL/TEST TEREXPOSE" in source
    assert "authorize-training" not in source
    assert "TORCH PRE-INSTALL" in source
    assert "--no-deps" in source
    assert "torch_version_before" in source
    assert "v2-rgb-native" in source
    helper = Path("src/coffee_detector/experiments/prepare_defectoscafeverde_routability_colab.py").read_text(encoding="utf-8")
    assert "1maWHfcfTkM7CB5V4NpBP9WlI-D74TuKB" in helper
    assert "CHECKPOINT_SHA" in helper
    assert "supportsAllDrives=True" in helper


def test_native_rgb_wrapper_reverses_diagnostic_bgr(monkeypatch):
    import torch
    import coffee_detector.analysis.defectoscafeverde_train_routability as module

    bgr = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    monkeypatch.setattr(module, "_letterbox_sample", lambda *args, **kwargs: (bgr, "boxes", "labels", "shape"))
    image, boxes, labels, shape = _native_rgb_letterbox_sample("unused")
    assert image.flatten().tolist() == [3.0, 2.0, 1.0]
    assert (boxes, labels, shape) == ("boxes", "labels", "shape")


def test_final_router_round_trip_uses_only_frozen_subset():
    rows = []
    for index in range(30):
        correct = {"RAFC1"} if index % 5 == 0 else {"D0DIRECT"}
        classes = {model: (1 if model == "RAFC1" and index % 5 == 0 else 0) for model in MODEL_ORDER}
        rows.append(_row(f"{index}.jpg", correct, classes=classes))
    subset = ("D0DIRECT", "RAFC1")
    router = fit_final_router(rows, subset, 12)
    choices = {predict_final_router(router, row) for row in rows}
    assert choices <= set(subset)
    assert router["training_targets"] == 30

