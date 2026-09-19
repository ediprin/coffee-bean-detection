import ast
import json
from pathlib import Path

from coffee_detector.analysis.coffee_standard_j25_black_broken_audit import (
    TARGET_CLASS,
    _target_view,
    attribute_target,
)


ROOT = Path(__file__).resolve().parents[1]


def _stage(*, targets=4, accessible=4, matched=4, correct=4):
    return {
        "targets": targets,
        "accessible": accessible,
        "matched": matched,
        "correct_class": correct,
        "proposal_accessibility": accessible / max(targets, 1),
        "matched_recall": matched / max(targets, 1),
        "localization_conditioned_class_accuracy": correct / max(matched, 1),
    }


def _view(raw, final):
    return {"raw_top500": raw, "final_conf0001": final}


def test_target_attribution_separates_localization_classification_and_selection():
    assert attribute_target(_view(_stage(accessible=0, matched=0, correct=0), _stage(accessible=0, matched=0, correct=0))) == "NO_RAW_LOCALIZATION"
    assert attribute_target(_view(_stage(correct=0), _stage(correct=0))) == "RAW_LOCALIZED_BUT_WRONG_CLASS"
    assert attribute_target(_view(_stage(), _stage(matched=2, correct=2))) == "FINAL_SELECTION_OR_RANKING_LOSS"
    assert attribute_target(_view(_stage(), _stage(correct=0))) == "FINAL_LOCALIZED_BUT_WRONG_CLASS"
    assert attribute_target(_view(_stage(), _stage())) == "CORRECT_DETECTIONS_EXIST_BUT_AP_RANKING_FAILS"


def test_target_view_preserves_counts_and_wrong_destination_names():
    diagnostic = {
        "branches": {
            "one2one": {
                "500": {
                    "per_class": {TARGET_CLASS: _stage(correct=1)},
                    "confusion": {TARGET_CLASS: {TARGET_CLASS: 1, "Biji Hitam Penuh": 3}},
                }
            }
        },
        "final_detections": {
            "per_class": {TARGET_CLASS: _stage(matched=2, correct=0)},
            "confusion": {TARGET_CLASS: {"Biji Hitam Penuh": 2}},
        },
    }
    view = _target_view(diagnostic)
    assert view["raw_top500"]["wrong_class"] == 3
    assert view["raw_top500"]["wrong_destinations"] == {"Biji Hitam Penuh": 3}
    assert view["final_conf0001"]["wrong_class"] == 2


def test_protocol_and_notebook_are_validation_only_and_parseable():
    protocol = (
        ROOT / "docs/COFFEE_STANDARD_J25_BLACK_BROKEN_ROOT_CAUSE_PROTOCOL_2026-09-19.md"
    ).read_text(encoding="utf-8")
    assert "Status: **frozen before diagnostic evaluation**" in protocol
    assert "no training" in protocol.lower()
    assert "locked test" in protocol

    notebook_path = ROOT / "notebooks/Coffee_Standard_J25_Black_Broken_Root_Cause_Colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    ast.parse(code)
    assert "BRANCH='codex/j25-black-broken-audit'" in code
    assert "coffee_standard_j25_black_broken_audit" in code
    assert "--authorize-diagnostic" in code
    assert "--authorize-training" not in code
    assert "--authorize-test" not in code
    assert "test/images" not in code
