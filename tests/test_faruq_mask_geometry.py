import json
from pathlib import Path

from PIL import Image, ImageDraw

from coffee_detector.audit_faruq_mask_geometry import audit_faruq_mask_geometry


def _write_fixture(root: Path) -> None:
    split = root / "train"
    split.mkdir(parents=True)
    expected = Image.new("RGB", (60, 100), "white")
    ImageDraw.Draw(expected).rectangle((8, 12, 27, 38), fill=(80, 40, 20))
    # Stored image needs a CCW transform to match the declared portrait COCO frame.
    raw = expected.transpose(Image.Transpose.ROTATE_270)
    raw.save(split / "bean.jpg")
    payload = {
        "images": [{"id": 1, "file_name": "bean.jpg", "width": 60, "height": 100}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 12, 19, 26],
                "segmentation": [[8, 12, 27, 12, 27, 38, 8, 38]],
            }
        ],
        "categories": [{"id": 1, "name": "biji_normal"}],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")


def test_mask_audit_detects_wrong_fixed_clockwise_rule(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    output = tmp_path / "output"
    _write_fixture(raw)

    summary = audit_faruq_mask_geometry(
        raw, output, score_long_side=100, min_improvement=0.01, contact_sheet_limit=2
    )

    records = json.loads(Path(summary["records"]).read_text(encoding="utf-8"))
    assert records[0]["current_transform"] == "rotate_cw"
    assert records[0]["best_transform"] == "rotate_ccw"
    assert records[0]["flagged_orientation"] is True
    assert summary["flagged_fraction"] == 1.0
    assert summary["test_images_accessed"] is False
    assert summary["training_executed"] is False
    assert summary["inference_executed"] is False
    assert Path(summary["contact_sheet"]).is_file()
