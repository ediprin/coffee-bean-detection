import json
from pathlib import Path

import pytest

from coffee_detector.analysis.defectoscafeverde_error_complementarity import (
    MODEL_ORDER,
    summarize_records,
    validate_result_contracts,
)


def _stage(correct=False, matched=None, accessible=None):
    matched = correct if matched is None else matched
    accessible = matched if accessible is None else accessible
    return {
        "accessible": accessible,
        "matched": matched,
        "correct": correct,
        "predicted_class_id": 0 if matched else None,
        "confidence": 0.9 if matched else None,
        "iou": 0.8 if matched else None,
    }


def _record(class_id, raw_correct, final_correct):
    return {
        "image": f"image-{class_id}.jpg",
        "target_index": 0,
        "class_id": class_id,
        "class_name": {0: "negro", 1: "agrio"}[class_id],
        "models": {
            model: {
                "raw_top500": _stage(raw_correct.get(model, False)),
                "final_conf0001": _stage(final_correct.get(model, False)),
            }
            for model in MODEL_ORDER
        },
    }


def test_summary_exposes_rescues_regressions_and_union_headroom():
    records = [
        _record(
            0,
            {"D0DIRECT": True},
            {"D0DIRECT": True, "AF2DIRECT": True},
        ),
        _record(
            0,
            {"AF2DIRECT": True},
            {"AF2DIRECT": True},
        ),
        _record(
            1,
            {},
            {"D0DIRECT": True, "RAFC1": True},
        ),
        _record(1, {}, {}),
    ]
    result = summarize_records(records, {0: "negro", 1: "agrio"})
    final = result["stages"]["final_conf0001"]
    assert final["models"]["D0DIRECT"]["correct_decision_recall"] == pytest.approx(0.5)
    assert final["union"]["correct_decision_recall"] == pytest.approx(0.75)
    assert final["union"]["gain_over_reference"] == pytest.approx(0.25)
    assert final["candidate_vs_reference"]["AF2DIRECT"][
        "rescued_reference_errors"
    ] == 1
    assert final["candidate_vs_reference"]["AF2DIRECT"][
        "lost_reference_correct"
    ] == 1
    assert final["per_class"]["negro"]["union_gain_over_reference"] == pytest.approx(
        0.5
    )


def _write_result(path: Path, arm: str, checkpoint_sha: str, dataset_sha="same"):
    payload = {
        "arm": arm,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "completed_epochs": 50,
        "checkpoint_sha256": checkpoint_sha,
        "run_contract": {
            "dataset_audit_sha256": dataset_sha,
            "protocol": "protocol",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_contract_rejects_mixed_dataset_audit(tmp_path):
    import hashlib

    results, checkpoints = {}, {}
    for index, arm in enumerate(MODEL_ORDER):
        checkpoint = tmp_path / f"{arm}.pt"
        checkpoint.write_bytes(arm.encode())
        sha = hashlib.sha256(arm.encode()).hexdigest()
        result = tmp_path / f"{arm}.json"
        _write_result(result, arm, sha, "other" if index == 4 else "same")
        results[arm] = result
        checkpoints[arm] = checkpoint
    with pytest.raises(RuntimeError, match="Dataset audit SHA tidak seragam"):
        validate_result_contracts(results, checkpoints)


def test_notebook_is_validation_only_and_names_exact_checkpoints():
    source = Path(
        "notebooks/DefectosCafeVerde_Error_Complementarity_Audit_Colab.ipynb"
    ).read_text(encoding="utf-8")
    assert "codex/defectoscafeverde-error-complementarity" in source
    assert "run_error_complementarity_audit" in source
    assert "D0DIRECT_seed42/weights/best.pt" in source
    assert "AF2DIRECT_seed42/weights/best.pt" in source
    assert "DCWCF1_seed42/weights/best.pt" in source
    assert "LIFRPF1_seed42/weights/best.pt" in source
    assert "RAFC1_seed42/weights/best.pt" in source
    assert "Drive API fallback" in source
    assert "14syRwc1tGNs2Lq27q-1X6B0VQwv88qWQ" in source
    assert "1maWHfcfTkM7CB5V4NpBP9WlI-D74TuKB" in source
    assert "CHECKPOINT_SHA" in source
    assert "PROJECT=Path('/content/drive/MyDrive/Coffee_Bean_Detection')" in source
    assert "Harus ada tepat satu D0 result" not in source
    assert "authorize-training" not in source
    assert "'test'" in source
