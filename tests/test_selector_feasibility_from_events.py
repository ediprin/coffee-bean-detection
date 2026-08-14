from coffee_detector.analysis.selector_feasibility_from_events import compare_pair


def _payload(name, rows):
    return {
        "protocol": "faruq-v3-validation-object-events-v1",
        "model": name,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "events": rows,
    }


def _row(gt, pred, conf, correct=True, matched=True):
    return {
        "image": "x.jpg",
        "gt_class_id": 0,
        "gt_class_name": gt,
        "matched": matched,
        "pred_class_name": pred if matched else None,
        "confidence": conf if matched else None,
        "correct": correct if matched else False,
    }


def test_higher_confidence_selector_metrics():
    a = _payload(
        "IGEM1",
        {
            "a": _row("g", "g", 0.60, True),   # disagree; B higher and wrong
            "b": _row("g", "x", 0.40, False),  # disagree; B higher and correct
            "c": _row("g", "x", 0.30, False),  # disagree; both wrong
            "d": _row("g", "g", 0.80, True),   # agree and correct
        },
    )
    b = _payload(
        "AF2",
        {
            "a": _row("g", "x", 0.90, False),
            "b": _row("g", "g", 0.70, True),
            "c": _row("g", "y", 0.60, False),
            "d": _row("g", "g", 0.70, True),
        },
    )
    result = compare_pair(a, b)
    assert result["joint_matched_targets"] == 4
    assert result["disagreement_targets"] == 3
    assert result["resolvable_disagreements_exactly_one_correct"] == 2
    assert result["both_wrong_disagreements"] == 1
    assert result["higher_conf_correct_expert_rate_when_exactly_one_correct"] == 0.5
    assert result["higher_conf_accuracy_on_all_disagreements"] == 1 / 3
    assert result["oracle_accuracy_on_disagreements"] == 2 / 3


def test_threshold_policy_defaults_to_primary_outside_disagreement():
    a = _payload(
        "IGEM1",
        {
            "a": _row("g", "g", 0.60, True),
            "b": _row("g", "g", 0.80, True),
        },
    )
    b = _payload(
        "SAF1",
        {
            "a": _row("g", "x", 0.90, False),
            "b": _row("g", "g", 0.95, True),
        },
    )
    result = compare_pair(a, b)
    sweep = result["candidate_switch_threshold_sweep"]
    at_zero = next(row for row in sweep if row["threshold"] == 0.0)
    assert at_zero["switches"] == 1
    assert at_zero["harmful_switches"] == 1
    assert at_zero["beneficial_switches"] == 0
    assert at_zero["gt_aligned_object_accuracy"] == 0.5
