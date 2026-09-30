import json
from pathlib import Path

from coffee_detector.analysis.defectoscafeverde_dual_view_audit import (
    _eligible_pairs,
    _source_order,
    _validation_samples,
    summarize_pairs,
)


class Annotation:
    def __init__(self, class_id: int):
        self.class_id = class_id


def _row(name: str, class_id: int = 0):
    return (
        Path(name),
        (Annotation(class_id),),
        {"source_name": name, "physical_group_id": "A0"},
    )


def test_source_order_uses_numeric_sequence_not_lexical_order():
    assert _source_order("A2.png") < _source_order("A10.png")


def test_eligibility_requires_two_single_object_views_with_same_label():
    good, rejected = _eligible_pairs({"A0": [_row("A0.png"), _row("A1.png")]})
    assert len(good) == 1
    assert not rejected

    disagreement, rejected = _eligible_pairs(
        {"A0": [_row("A0.png", 0), _row("A1.png", 1)]}
    )
    assert not disagreement
    assert rejected["paired_labels_disagree"] == 1


def test_validation_loader_does_not_require_train_split(tmp_path: Path):
    (tmp_path / "val/images").mkdir(parents=True)
    (tmp_path / "val/labels").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text(
        "path: ignored\nval: val/images\nnames:\n  0: agrio\n",
        encoding="utf-8",
    )
    (tmp_path / "val/images/A0.jpg").write_bytes(b"not-decoded-by-loader")
    (tmp_path / "val/labels/A0.txt").write_text(
        "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
    )

    names, samples = _validation_samples(tmp_path)

    assert names == {0: "agrio"}
    assert len(samples) == 1
    assert samples[0][1][0].class_id == 0
    assert not (tmp_path / "train").exists()


def test_pair_summary_separates_paper_rule_from_oracle():
    records = [
        {
            "class_id": 0,
            "views": [
                {"correct": True, "confidence": 0.60},
                {"correct": False, "confidence": 0.90},
            ],
            "paper_max_confidence_correct": False,
            "pair_oracle_correct": True,
        },
        {
            "class_id": 0,
            "views": [
                {"correct": False, "confidence": 0.40},
                {"correct": True, "confidence": 0.80},
            ],
            "paper_max_confidence_correct": True,
            "pair_oracle_correct": True,
        },
    ]
    summary = summarize_pairs(records, {0: "agrio"})
    assert summary["best_single_side_accuracy"] == 0.5
    assert summary["paper_max_confidence_accuracy"] == 0.5
    assert summary["pair_oracle_accuracy"] == 1.0
    assert summary["learned_fusion_headroom_over_paper_rule"] == 0.5
    assert summary["exactly_one_side_correct"] == 2


def test_protocol_freezes_no_training_and_test_lock():
    text = Path(
        "docs/DEFECTOSCAFEVERDE_DUAL_VIEW_AUDIT_PROTOCOL_2026-09-30.md"
    ).read_text(encoding="utf-8")
    assert "No detector, classifier, router, or fusion model is trained" in text
    assert "Test is not extracted or accessed" in text
    assert "AUTHORIZE_ONE_LEARNED_DUAL_VIEW_ARCHITECTURE_SCREEN" in text


def test_colab_notebook_compiles_and_extracts_validation_only():
    notebook = json.loads(
        Path("notebooks/DefectosCafeVerde_Dual_View_Audit_Colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    compile(source, "dual_view_audit_colab", "exec")
    assert "codex/defectoscafeverde-dual-view-fusion" in source
    assert "defectoscafeverde_dual_view_audit" in source
    assert "first=='val'" in source
    assert "first in {'train','val'}" not in source
    assert "if (DATA/'test').exists() or (DATA/'train').exists()" in source
    assert "--authorize-training" not in source
