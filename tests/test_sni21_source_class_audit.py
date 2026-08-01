import json
from pathlib import Path

import yaml
from PIL import Image

from coffee_detector.analyze_sni21_source_classes import (
    analyze_sni21_source_classes,
)
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


def _write_source(root: Path, source: str, present: tuple[int, ...]) -> None:
    source_root = root / source
    (source_root / "val/images").mkdir(parents=True)
    (source_root / "val/labels").mkdir(parents=True)
    for index, class_id in enumerate(present):
        name = f"{source}_{index}.jpg"
        Image.new("RGB", (8, 8), (80, 40, 20)).save(
            source_root / "val/images" / name
        )
        (source_root / "val/labels" / Path(name).with_suffix(".txt")).write_text(
            f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    (source_root / "train/images").mkdir(parents=True)
    (source_root / "train/labels").mkdir(parents=True)
    (source_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(source_root),
                "train": "train/images",
                "val": "val/images",
                "names": {i: name for i, name in enumerate(SNI21_CLASSES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_class_audit_preserves_missing_gt_and_never_reads_test(tmp_path: Path) -> None:
    separated = tmp_path / "separated"
    reports = tmp_path / "reports"
    output = tmp_path / "output"
    reports.mkdir()
    report_paths = {}
    for source, present in (
        ("adrian_detection", (0, 1)),
        ("faruq_segmentation", (0, 2)),
    ):
        _write_source(separated, source, present)
        path = reports / f"{source}.json"
        path.write_text(
            json.dumps(
                {
                    "split": "val",
                    "test_images_accessed": False,
                    "metrics": {
                        "map50_95_by_class": {
                            SNI21_CLASSES[class_id]: 0.1 + class_id / 10
                            for class_id in present
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        report_paths[source] = str(path)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "training_executed": False,
                "test_images_accessed": False,
                "reports": report_paths,
            }
        ),
        encoding="utf-8",
    )

    summary = analyze_sni21_source_classes(separated, evaluation, output)

    assert summary["training_executed"] is False
    assert summary["test_images_accessed"] is False
    assert len(summary["rows"]) == 42
    adrian_missing = summary["sources"]["adrian_detection"][
        "classes_without_ground_truth"
    ]
    assert SNI21_CLASSES[2] in adrian_missing
    missing_row = next(
        row
        for row in summary["rows"]
        if row["source_dataset"] == "adrian_detection" and row["class_id"] == 2
    )
    assert missing_row["map50_95"] is None
    assert missing_row["val_instances"] == 0
