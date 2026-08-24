import json
from pathlib import Path

import pytest
from PIL import Image

from coffee_detector.analysis.faruq_v3_label_visual_audit import (
    _quantile_select,
    audit_faruq_v3_label_visuals,
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
    "biji_muda",
    "biji_bertutul_tutul",
]


def _dataset(root: Path) -> None:
    (root / "data.yaml").write_text(
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(NAMES)),
        encoding="utf-8",
    )
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for class_id, _ in enumerate(NAMES):
            for sample in range(2):
                stem = f"{class_id}_{sample}"
                Image.new("RGB", (160, 120), (180, 150, 110)).save(
                    root / split / "images" / f"{stem}.jpg"
                )
                size = 0.10 + 0.02 * (class_id % 3) + 0.01 * sample
                (root / split / "labels" / f"{stem}.txt").write_text(
                    f"{class_id} 0.5 0.5 {size} {size}\n", encoding="utf-8"
                )


def _diagnostic(path: Path) -> None:
    path.write_text(
        json.dumps({
            "training_executed": False,
            "test_images_accessed": False,
            "top_directional_confusions": [{
                "expected": "biji_muda",
                "predicted": "biji_bertutul_tutul",
                "count": 5,
            }],
        }),
        encoding="utf-8",
    )


def test_visual_audit_is_deterministic_training_free_and_test_locked(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _dataset(data)
    diagnostic = tmp_path / "diagnostic.json"
    _diagnostic(diagnostic)

    result = audit_faruq_v3_label_visuals(
        data,
        diagnostic,
        tmp_path / "output",
        samples_per_class=2,
        max_local_pairs=1,
    )
    assert result["decision"] == "PENDING_HUMAN_VISUAL_REVIEW"
    assert result["training_executed"] is False
    assert result["inference_executed"] is False
    assert result["test_images_accessed"] is False
    assert len(result["size_sheets"]) == 6
    assert len(result["local_pair_sheets"]) == 2
    assert all(Path(row["contact_sheet"]).is_file() for row in result["size_sheets"])
    assert all(Path(row["contact_sheet"]).is_file() for row in result["local_pair_sheets"])

    (data / "test").mkdir()
    with pytest.raises(RuntimeError, match="tidak boleh menyediakan test"):
        audit_faruq_v3_label_visuals(data, diagnostic, tmp_path / "blocked")


def test_quantile_selection_covers_extremes() -> None:
    class Item:
        def __init__(self, area: float) -> None:
            self.normalized_area = area
            self.image_path = Path(f"{area}.jpg")

    selected = _quantile_select([Item(value) for value in (0.5, 0.1, 0.4, 0.2, 0.3)], 3)
    assert [item.normalized_area for item in selected] == [0.1, 0.3, 0.5]
