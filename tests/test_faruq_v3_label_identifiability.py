import json
from pathlib import Path

import pytest
from PIL import Image

from coffee_detector.analysis.faruq_v3_label_identifiability import (
    _auc_for_order,
    _confusion_kind,
    audit_faruq_v3_label_identifiability,
)


NAMES = [
    "kulit_kopi_ukuran_kecil",
    "kulit_kopi_ukuran_sedang",
    "kulit_kopi_ukuran_besar",
    "kulit_tanduk_ukuran_kecil",
    "kulit_tanduk_ukuran_sedang",
    "kulit_tanduk_ukuran_besar",
    "tanah_batu_ranting_kecil",
    "tanah_batu_ranting_sedang",
    "tanah_batu_ranting_besar",
]


def _dataset(root: Path) -> None:
    (root / "data.yaml").write_text(
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(NAMES)),
        encoding="utf-8",
    )
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for index, class_id in enumerate(range(len(NAMES))):
            image = root / split / "images" / f"{index}.jpg"
            Image.new("RGB", (100, 100), "white").save(image)
            size = (0.10, 0.20, 0.40)[class_id % 3]
            (root / split / "labels" / f"{index}.txt").write_text(
                f"{class_id} 0.5 0.5 {size} {size}\n", encoding="utf-8"
            )


def test_order_auc_and_confusion_taxonomy() -> None:
    assert _auc_for_order([0.1, 0.2], [0.3, 0.4]) == 1.0
    assert _confusion_kind(
        "kulit_kopi_ukuran_kecil", "kulit_kopi_ukuran_besar"
    ) == "within_family_size"
    assert _confusion_kind("biji_muda", "biji_bertutul_tutul") == (
        "local_defect_similarity"
    )


def test_identifiability_audit_is_training_free_and_test_locked(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _dataset(data)
    diagnostic = tmp_path / "diagnostic.json"
    diagnostic.write_text(
        json.dumps({
            "top_directional_confusions": [{
                "expected": "kulit_kopi_ukuran_kecil",
                "predicted": "kulit_kopi_ukuran_besar",
                "count": 3,
            }]
        }),
        encoding="utf-8",
    )
    result = audit_faruq_v3_label_identifiability(
        data, tmp_path / "result.json", diagnostic=diagnostic
    )
    assert result["decision"] == "GEOMETRY_HEAD_JUSTIFIED"
    assert result["training_executed"] is False
    assert result["test_images_accessed"] is False
    assert result["confusion_taxonomy"]["counts_by_kind"] == {
        "within_family_size": 3
    }

    (data / "test").mkdir()
    with pytest.raises(RuntimeError, match="tidak boleh menyediakan test"):
        audit_faruq_v3_label_identifiability(data, tmp_path / "blocked.json")
