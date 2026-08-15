from coffee_detector.experiments.run_faruq_v3_geometry_conditioning_paired_confirmation import (
    ALL_SEEDS,
    _aggregate,
)


def _record(macro, bottom3, worst, size, control="PASS"):
    return {
        "control_validity": {"decision": control},
        "geo_minus_geoc0": {
            "macro_map50_95": macro,
            "bottom3_class_map50_95": bottom3,
            "worst_class_map50_95": worst,
            "size_class_mean_map50_95": size,
        },
    }


def test_confirmation_gate_passes_reproducible_geometry_signal():
    assert ALL_SEEDS == (42, 123, 2026)
    per_seed = {
        "42": _record(0.0022, 0.0510, 0.0835, 0.0161),
        "123": _record(0.0040, 0.0100, 0.0120, 0.0080),
        "2026": _record(0.0030, 0.0060, 0.0040, 0.0070),
    }
    aggregate, criteria = _aggregate(per_seed)
    assert aggregate["macro_map50_95"]["improved_seeds"] == 3
    assert all(criteria.values())


def test_confirmation_gate_rejects_seed42_only_effect():
    per_seed = {
        "42": _record(0.0022, 0.0510, 0.0835, 0.0161),
        "123": _record(-0.0040, -0.0100, -0.0120, -0.0080),
        "2026": _record(-0.0030, -0.0060, -0.0040, -0.0070),
    }
    _, criteria = _aggregate(per_seed)
    assert not criteria["macro_improved_at_least_2_of_3"]
    assert not criteria["size_mean_improved_at_least_2_of_3"]
    assert not all(criteria.values())


def test_confirmation_gate_rejects_invalid_zero_information_control():
    per_seed = {
        "42": _record(0.0040, 0.0100, 0.0100, 0.0100),
        "123": _record(0.0040, 0.0100, 0.0100, 0.0100, control="FAIL"),
        "2026": _record(0.0040, 0.0100, 0.0100, 0.0100),
    }
    _, criteria = _aggregate(per_seed)
    assert criteria["all_three_geoc0_validity_pass"] is False
    assert not all(criteria.values())
