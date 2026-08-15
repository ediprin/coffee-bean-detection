import numpy as np

from coffee_detector.analysis.scale_identifiability_audit import (
    _best_balanced_accuracy,
    _expected_auc,
    _iqr_overlap,
    _parse_size_class,
    _summary,
)


def test_parse_size_class_variants():
    assert _parse_size_class("kulit_kopi_ukuran_besar") == ("kulit_kopi", "besar", 2)
    assert _parse_size_class("tanah_batu_ranting_sedang") == ("tanah_batu_ranting", "sedang", 1)
    assert _parse_size_class("biji_normal") is None


def test_expected_auc_perfect_order():
    low = np.asarray([1.0, 2.0, 3.0])
    high = np.asarray([4.0, 5.0, 6.0])
    assert _expected_auc(low, high) == 1.0
    score, threshold = _best_balanced_accuracy(low, high)
    assert score == 1.0
    assert 3.0 < threshold < 4.0


def test_iqr_overlap_positive_case():
    a = _summary(np.asarray([1.0, 2.0, 3.0, 4.0]))
    b = _summary(np.asarray([2.0, 3.0, 4.0, 5.0]))
    overlap = _iqr_overlap(a, b)
    assert overlap["exists"] is True
    assert overlap["width"] > 0.0
    assert 0.0 < overlap["fraction_of_iqr_union"] <= 1.0


def test_iqr_overlap_disjoint_case():
    a = _summary(np.asarray([1.0, 2.0, 3.0, 4.0]))
    b = _summary(np.asarray([5.0, 6.0, 7.0, 8.0]))
    overlap = _iqr_overlap(a, b)
    assert overlap["exists"] is False
    assert overlap["width"] == 0.0
    assert overlap["fraction_of_iqr_union"] == 0.0
