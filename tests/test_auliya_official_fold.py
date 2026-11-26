import hashlib
import json

import pytest
import yaml

from coffee_detector.experiments import prepare_auliya_official_fold as fold


@pytest.fixture
def development(tmp_path, monkeypatch):
    # Historical Roboflow `test` path is unrelated to held-out group A.
    examples = {"train": "test/a", "val": "valid/b"}
    for role, item in examples.items():
        source_role, name = item.split("/")
        for kind, suffix in (("images", ".jpg"), ("labels", ".txt")):
            path = tmp_path / "source" / source_role / kind / (name + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        (tmp_path / f"{role}.txt").write_bytes(f"./source/{source_role}/images/{name}.jpg\n".encode())
    hashes = {
        r: hashlib.sha256((tmp_path / f"{r}.txt").read_bytes()).hexdigest() for r in examples
    }
    monkeypatch.setattr(fold, "COUNTS", {"train": 1, "val": 1})
    monkeypatch.setattr(fold, "SPLIT_SHA256", hashes)
    names = [f"class{i}" for i in range(21)]
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(fold._dataset(names)), encoding="utf-8")
    (tmp_path / "official_fold_contract.json").write_text(json.dumps({
        "protocol": fold.PROTOCOL, "held_out_group": "A", "held_out_count": fold.GROUP_A_COUNT,
        "held_out_contents_read": False, "images": fold.COUNTS, "names": names,
        "metadata_sha256": fold.METADATA_SHA256, "pilot_sha256": fold.PILOT_SHA256, "split_sha256": hashes,
    }), encoding="utf-8")
    return tmp_path


def test_fold_preserves_original_archive_names(development):
    assert fold.validate_official_fold(development) == development / "data.yaml"
    assert fold._archive_paths("test/images/a.jpg", "B") == (
        "development/train/images/test__a.jpg", "development/train/labels/test__a.txt",
    )


def test_fold_rejects_changed_split(development):
    (development / "val.txt").write_bytes(b"./source/test/images/a.jpg\n")
    with pytest.raises(ValueError, match="val list changed"):
        fold.validate_official_fold(development)


def test_fold_rejects_test_reference_and_extra_payload(development):
    data = development / "data.yaml"
    original = data.read_text(encoding="utf-8")
    data.write_text(original + "test: test.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train/val-only"):
        fold.validate_official_fold(development)
    data.write_text(original, encoding="utf-8")
    (development / "source/test/images/not_in_development.jpg").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="inventory changed"):
        fold.validate_official_fold(development)
