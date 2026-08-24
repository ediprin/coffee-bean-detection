from coffee_detector.analysis.faruq_v3_fusion_cm512 import decide_fusion_cm512


def _metrics(
    macro: float,
    bottom3: float,
    worst: float,
    top3: float,
) -> dict:
    return {
        "macro_f1": macro,
        "bottom3_f1": bottom3,
        "worst_class_f1": worst,
        "top3_accuracy": top3,
    }


def test_cm512_gate_passes_material_lower_tail_safe_fusion() -> None:
    decision = decide_fusion_cm512(
        _metrics(0.73, 0.56, 0.52, 0.90),
        _metrics(0.78, 0.64, 0.58, 0.93),
    )
    assert decision["decision"] == "PASS"
    assert decision["next_action"] == "AUTHORIZE_MULTILEVEL_HEAD_PROTOCOL"
    assert decision["deltas"]["macro_f1"] > 0.02
    assert decision["detector_training_authorized"] is False


def test_cm512_gate_rejects_nonmaterial_fusion() -> None:
    decision = decide_fusion_cm512(
        _metrics(0.76, 0.58, 0.54, 0.92),
        _metrics(0.77, 0.59, 0.54, 0.93),
    )
    assert decision["decision"] == "FAIL"
    assert decision["next_action"] == "STOP_MULTILEVEL_HEAD_CAPACITY_CONTROL"


def test_cm512_gate_rejects_lower_tail_damage() -> None:
    decision = decide_fusion_cm512(
        _metrics(0.73, 0.58, 0.55, 0.91),
        _metrics(0.78, 0.57, 0.53, 0.92),
    )
    assert decision["criteria"]["macro_gain_at_least_2_points"] is True
    assert decision["criteria"]["bottom3_preserved"] is False
    assert decision["criteria"]["worst_drop_no_more_than_1_point"] is False
    assert decision["decision"] == "FAIL"
