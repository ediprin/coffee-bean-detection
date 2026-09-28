from coffee_detector.analysis.defectoscafeverde_lif_rpf_gate_audit import (
    classify_attribution,
)


def _values(macro, bottom3, worst):
    return {
        "macro_map50_95": macro,
        "bottom3_class_map50_95": bottom3,
        "worst_class_map50_95": worst,
    }


def test_gate_audit_attributes_negligible_gate_and_lower_zero_endpoint_to_training_path():
    baseline = _values(0.92, 0.87, 0.85)
    active = _values(0.91, 0.86, 0.84)
    zero = _values(0.9102, 0.8601, 0.8402)
    assert classify_attribution(baseline, active, zero) == (
        "TRAINING_PATH_DOMINANT_CUE_NEARLY_IGNORED"
    )


def test_gate_audit_can_identify_active_inference_harm():
    baseline = _values(0.92, 0.87, 0.85)
    active = _values(0.90, 0.83, 0.80)
    zero = _values(0.91, 0.85, 0.82)
    assert classify_attribution(baseline, active, zero) == (
        "ACTIVE_CUE_HARMS_INFERENCE"
    )
