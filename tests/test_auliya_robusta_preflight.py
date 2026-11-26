from io import BytesIO
from zipfile import ZipFile

import pytest
from PIL import Image

from coffee_detector.analysis import auliya_robusta_preflight as preflight


def _fixtures(tmp_path, monkeypatch, *, leak_group=False, labels=True):
    metadata_path = tmp_path / "metadata.zip"
    images_path = tmp_path / "images.zip"
    source_paths = {
        "A": "train/images/bean-a.jpg",
        "B": "test/images/bean-b.jpg",
        "C": "valid/images/bean-c.jpg",
    }
    with ZipFile(metadata_path, "w") as bundle:
        bundle.writestr(
            preflight.MANIFEST,
            "split,file,var,kelompok,n\n"
            "train,bean-a.txt,V1,A,1\n"
            "test,bean-b.txt,V3,B,1\n"
            "valid,bean-c.txt,V5,C,1\n",
        )
        for group in "ABC":
            train, val = [key for key in "ABC" if key != group]
            if group == "A" and leak_group:
                group, train = train, group
                folder_group = "A"
            else:
                folder_group = group
            for role, key in (("train", train), ("val", val), ("test", group)):
                bundle.writestr(f"splits/v4_uji{folder_group}_{role}.txt", f"./{source_paths[key]}\n")
    monkeypatch.setattr(preflight, "METADATA_SHA256", preflight._sha256(metadata_path))
    image = BytesIO()
    with Image.new("RGB", (3000, 4000)) as frame:
        frame.save(image, format="JPEG")
    with ZipFile(images_path, "w") as bundle:
        for group, path in source_paths.items():
            # The held-out entries are indexed only. Reading them would fail.
            bundle.writestr(f"export/{path}", b"invalid JPEG" if group == "A" else image.getvalue())
            if labels:
                polygon = "invalid held-out polygon" if group == "A" else "3 .1 .1 .5 .1 .5 .5 .1 .5"
                bundle.writestr(f"export/{preflight._label_name(path)}", polygon)
    return metadata_path, images_path


def test_development_contents_follow_official_fold_not_archive_folder(tmp_path, monkeypatch):
    metadata, images = _fixtures(tmp_path, monkeypatch)

    report = preflight.run_preflight(metadata, images)

    assert report["status"] == "DEVELOPMENT_ARCHIVE_AUDITED"
    assert report["held_out_contents_read"] is False
    assert report["development_polygon_instances_by_class"]["train"][3] == 1
    assert report["development_polygon_instances_by_class"]["val"][3] == 1
    assert 3 not in report["missing_class_ids_by_role"]["val"]
    assert preflight.run_preflight(metadata)["status"] == "METADATA_ONLY"


def test_image_only_export_fails_before_decoding(tmp_path, monkeypatch):
    metadata, images = _fixtures(tmp_path, monkeypatch, labels=False)

    with pytest.raises(ValueError, match="missing labels=3"):
        preflight.run_preflight(metadata, images)


def test_acquisition_group_leakage_stops_preflight(tmp_path, monkeypatch):
    metadata, images = _fixtures(tmp_path, monkeypatch, leak_group=True)

    with pytest.raises(ValueError, match="crosses the held-out boundary"):
        preflight.run_preflight(metadata, images)
