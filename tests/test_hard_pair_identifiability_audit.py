from coffee_detector.analysis.hard_pair_identifiability_audit import _is_pair_error, _pair_name


def test_pair_name_is_undirected_and_sorted():
    assert _pair_name("b", "a") == "a <-> b"
    assert _pair_name("a", "b") == "a <-> b"


def test_is_pair_error_requires_matched_wrong_and_exact_family():
    row = {
        "matched": True,
        "correct": False,
        "gt_class_name": "biji_muda",
        "pred_class_name": "biji_berlubang_satu",
    }
    assert _is_pair_error(row, "biji_berlubang_satu <-> biji_muda")
    assert not _is_pair_error(row, "biji_muda <-> biji_normal")
    row["correct"] = True
    assert not _is_pair_error(row, "biji_berlubang_satu <-> biji_muda")
    row["correct"] = False
    row["matched"] = False
    assert not _is_pair_error(row, "biji_berlubang_satu <-> biji_muda")
