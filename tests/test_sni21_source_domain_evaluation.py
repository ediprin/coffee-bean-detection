import json
from pathlib import Path

import pytest

import coffee_detector.evaluate_sni21_source_domains as runner


def _write_separation(root: Path) -> None:
    rows = []
    for source, images, boxes in (
        ("adrian_detection", 2, 20),
        ("faruq_segmentation", 3, 6),
    ):
        source_root = root / source
        (source_root / "val/images").mkdir(parents=True)
        rows.append(
            {
                "source_dataset": source,
                "val_images": images,
                "val_boxes": boxes,
            }
        )
    (root / "source_separation_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "test_images_accessed": False,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )


def test_evaluates_each_source_without_training_or_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    separated = tmp_path / "separated"
    output = tmp_path / "reports"
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    _write_separation(separated)
    calls = []

    def fake_evaluate(checkpoint, data_root, output, split, device):
        calls.append((Path(data_root).name, split, device))
        return {
            "checkpoint": str(checkpoint),
            "data": str(Path(data_root).resolve()),
            "split": split,
            "metrics": {
                "metrics/mAP50-95(B)": 0.4,
                "metrics/mAP50(B)": 0.6,
                "metrics/precision(B)": 0.5,
                "metrics/recall(B)": 0.55,
                "macro_map50_95": 0.4,
                "bottom3_class_map50_95": 0.1,
                "worst_class_map50_95": 0.0,
                "classes_without_ground_truth": [],
            },
        }

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    summary = runner.evaluate_sni21_source_domains(
        checkpoint, separated, output, device="cpu"
    )

    assert calls == [
        ("adrian_detection", "val", "cpu"),
        ("faruq_segmentation", "val", "cpu"),
    ]
    assert summary["training_executed"] is False
    assert summary["test_images_accessed"] is False
    assert summary["rows"][0]["boxes_per_image"] == 10
    assert summary["rows"][1]["boxes_per_image"] == 2


def test_rejects_source_dataset_with_test_materialized(tmp_path: Path) -> None:
    separated = tmp_path / "separated"
    output = tmp_path / "reports"
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    _write_separation(separated)
    (separated / "adrian_detection/test").mkdir()
    with pytest.raises(RuntimeError, match="Test tidak boleh tersedia"):
        runner.evaluate_sni21_source_domains(checkpoint, separated, output)
