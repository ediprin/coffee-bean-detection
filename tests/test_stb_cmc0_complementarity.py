from coffee_detector.analysis.stb_cmc0_complementarity import _pair_summary


def _event(key, gt, *, matched, pred=None, correct=False, confidence=0.8, iou=0.7):
    return {
        "target_key": key,
        "image": "image.jpg",
        "target_index": int(key.rsplit("gt", 1)[-1]),
        "gt_class_id": gt,
        "gt_class_name": {0: "A", 1: "B"}[gt],
        "accessible": matched,
        "matched": matched,
        "pred_class_id": pred,
        "pred_class_name": None if pred is None else {0: "A", 1: "B"}[pred],
        "confidence": confidence if matched else None,
        "iou": iou if matched else None,
        "correct": correct,
    }


def test_pair_summary_tracks_directional_rescue_and_oracle():
    names = {0: "A", 1: "B"}
    cmc0 = {
        "image.jpg::gt0": _event("image.jpg::gt0", 0, matched=True, pred=0, correct=True),
        "image.jpg::gt1": _event("image.jpg::gt1", 1, matched=True, pred=0, correct=False),
        "image.jpg::gt2": _event("image.jpg::gt2", 0, matched=False),
        "image.jpg::gt3": _event("image.jpg::gt3", 1, matched=True, pred=1, correct=True),
    }
    stb = {
        "image.jpg::gt0": _event("image.jpg::gt0", 0, matched=True, pred=0, correct=True),
        "image.jpg::gt1": _event("image.jpg::gt1", 1, matched=True, pred=1, correct=True),
        "image.jpg::gt2": _event("image.jpg::gt2", 0, matched=True, pred=0, correct=True),
        "image.jpg::gt3": _event("image.jpg::gt3", 1, matched=True, pred=0, correct=False),
    }

    summary, rows = _pair_summary(cmc0, stb, names)

    assert len(rows) == 4
    assert summary["contingency"] == {
        "both_correct": 1,
        "cmc0_only_correct": 1,
        "stb_only_correct": 2,
        "neither_correct": 0,
    }
    assert summary["rescue"]["cmc0_to_stb_count"] == 2
    assert summary["rescue"]["stb_to_cmc0_count"] == 1
    assert summary["rescue"]["cmc0_to_stb_classification_rescue"] == 1
    assert summary["rescue"]["stb_to_cmc0_classification_rescue"] == 1
    assert summary["oracle"]["accuracy_iou50"] == 1.0
    assert summary["oracle"]["gain_over_best_model"] == 0.25
    assert summary["error_overlap"]["jaccard"] == 0.0
    assert summary["top_confusion_pair_rescues"]["cmc0_wrong_stb_correct"][0] == {
        "gt": "B",
        "cmc0_pred": "A",
        "count": 1,
    }


def test_pair_summary_detects_shared_errors():
    names = {0: "A", 1: "B"}
    cmc0 = {
        "image.jpg::gt0": _event("image.jpg::gt0", 0, matched=True, pred=1, correct=False),
        "image.jpg::gt1": _event("image.jpg::gt1", 1, matched=True, pred=1, correct=True),
    }
    stb = {
        "image.jpg::gt0": _event("image.jpg::gt0", 0, matched=True, pred=1, correct=False),
        "image.jpg::gt1": _event("image.jpg::gt1", 1, matched=True, pred=1, correct=True),
    }

    summary, _ = _pair_summary(cmc0, stb, names)

    assert summary["error_overlap"]["jaccard"] == 1.0
    assert summary["oracle"]["gain_over_best_model"] == 0.0
    assert summary["rescue"]["cmc0_to_stb_count"] == 0
    assert summary["rescue"]["stb_to_cmc0_count"] == 0
