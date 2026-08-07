import json
from pathlib import Path

import yaml
from PIL import Image

from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES
from coffee_detector.run_sni21_hard_class_visual_audit import (
    run_sni21_hard_class_visual_audit,
)


def _write_dataset(root: Path, source: str) -> None:
    source_root = root / source
    for split in ("train", "val"):
        (source_root / split / "images").mkdir(parents=True)
        (source_root / split / "labels").mkdir(parents=True)
        name = f"{source}_{split}.jpg"
        Image.new("RGB", (64, 48), (120, 90, 50)).save(
            source_root / split / "images" / name
        )
        (source_root / split / "labels" / Path(name).with_suffix(".txt")).write_text(
            "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
        )
    (source_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(source_root),
                "train": "train/images",
                "val": "val/images",
                "names": {index: name for index, name in enumerate(SNI21_CLASSES)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_visual_audit_is_data_only_and_test_locked(tmp_path: Path) -> None:
    separated = tmp_path / "separated"
    output = tmp_path / "output"
    for source in ("adrian_detection", "faruq_segmentation"):
        _write_dataset(separated, source)
    rows = []
    sources = {}
    for source in ("adrian_detection", "faruq_segmentation"):
        row = {
            "source_dataset": source,
            "class_id": 0,
            "class_name": SNI21_CLASSES[0],
            "train_instances": 60,
            "val_instances": 12,
            "map50_95": 0.1,
            "val_to_train_prevalence_ratio": 1.0,
        }
        rows.append(row)
        sources[source] = {"largest_train_val_prevalence_shifts": [row]}
    class_audit = tmp_path / "class_audit.json"
    class_audit.write_text(
        json.dumps(
            {
                "training_executed": False,
                "test_images_accessed": False,
                "rows": rows,
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )

    summary = run_sni21_hard_class_visual_audit(
        separated, class_audit, output, samples_per_split=1
    )

    assert summary["training_executed"] is False
    assert summary["inference_executed"] is False
    assert summary["test_images_accessed"] is False
    assert len(summary["sheets"]) == 2
    assert all(Path(row["contact_sheet"]).is_file() for row in summary["sheets"])
