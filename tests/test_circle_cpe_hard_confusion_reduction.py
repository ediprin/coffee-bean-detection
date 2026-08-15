import json

from coffee_detector.analysis.circle_cpe_hard_confusion_reduction import run


def _event(model, rows):
    return {
        "protocol": "faruq-v3-validation-object-events-v1",
        "model": model,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "matching": {"iou_threshold": 0.5},
        "events": rows,
    }


def test_hard_family_reduction_counts_paired_rescue(tmp_path):
    families = [f"a{i} <-> b{i}" for i in range(17)]
    consensus = {
        "protocol": "faruq-v3-cross-model-hard-confusion-consensus-v1",
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "undirected_families": [
            {"family": family, "frozen_consensus_hard_family": True}
            for family in families
        ],
    }
    cpe_rows = {
        "x::gt0": {
            "matched": True, "correct": False,
            "gt_class_name": "a0", "pred_class_name": "b0",
        },
        "x::gt1": {
            "matched": True, "correct": True,
            "gt_class_name": "a1", "pred_class_name": "a1",
        },
    }
    cir_rows = {
        "x::gt0": {
            "matched": True, "correct": True,
            "gt_class_name": "a0", "pred_class_name": "a0",
        },
        "x::gt1": {
            "matched": True, "correct": True,
            "gt_class_name": "a1", "pred_class_name": "a1",
        },
    }

    cpe = tmp_path / "cpe.json"
    cir = tmp_path / "cir.json"
    con = tmp_path / "consensus.json"
    out = tmp_path / "out.json"
    cpe.write_text(json.dumps(_event("CPE0", cpe_rows)))
    cir.write_text(json.dumps(_event("CIR0", cir_rows)))
    con.write_text(json.dumps(consensus))

    result = run(cpe, cir, con, out)
    assert result["classification_errors_iou50"] == {
        "CPE0": 1, "CIR0": 0, "delta_CIR0_minus_CPE0": -1,
    }
    assert result["frozen_hard_family_errors"]["absolute_reduction"] == 1
    assert result["paired_transitions"]["cpe_hard_to_cir_correct"] == 1
    assert result["family_summary"] == {"improved": 1, "unchanged": 16, "worsened": 0}
    assert result["screening_decision_remains"] == "STOP_CIRCLE_CPE"
