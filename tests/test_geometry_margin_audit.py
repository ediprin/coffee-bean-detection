from coffee_detector.analysis.geometry_margin_audit import _band, _cue_supports_gt, _margin, _robust_scale


def test_margin_and_bands():
    assert _margin(3.0, 2.0, 2.0) == 0.5
    assert _band(0.1) == "boundary_le_0p25_iqr"
    assert _band(0.4) == "near_0p25_to_0p5_iqr"
    assert _band(0.8) == "mid_0p5_to_1_iqr"
    assert _band(1.2) == "far_gt_1_iqr"


def test_cue_supports_expected_direction():
    assert _cue_supports_gt(0.3, 0.5, "low", "low", "high") is True
    assert _cue_supports_gt(0.7, 0.5, "high", "low", "high") is True
    assert _cue_supports_gt(0.7, 0.5, "low", "low", "high") is False


def test_robust_scale_uses_pooled_iqr():
    low = {"q25": 1.0, "q75": 2.0, "std": 0.5}
    high = {"q25": 3.0, "q75": 5.0, "std": 0.5}
    assert _robust_scale(low, high) == 4.0
