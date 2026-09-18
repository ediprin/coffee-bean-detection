import io
import zipfile

import numpy as np
from PIL import Image

from coffee_detector.data.prepare_coffee_standard_j25_source_split import (
    _parent,
    _representative,
)


def _jpeg(value: int) -> bytes:
    buffer = io.BytesIO()
    pattern = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) / (32 * 32 - 1)
    gray = np.clip(value + 80 * pattern, 0, 255).astype(np.uint8)
    pixels = np.repeat(gray[..., None], 3, axis=2)
    Image.fromarray(pixels).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_parent_rule_requires_label_content_to_become_identity() -> None:
    assert _parent("same_name.rf.0123456789abcdef") == "same_name"
    assert _parent("same_name") == "same_name"


def test_visual_medoid_accepts_three_structurally_identical_siblings(tmp_path) -> None:
    path = tmp_path / "siblings.zip"
    members = []
    with zipfile.ZipFile(path, "w") as archive:
        for index, value in enumerate((80, 100, 120)):
            name = f"train/images/source.rf.{index:032x}.jpg"
            archive.writestr(name, _jpeg(value))
            members.append({"image_member": name})
    with zipfile.ZipFile(path) as archive:
        selected, minimum = _representative(archive, members)
    assert selected in members
    assert minimum > 0.99
