import json
from pathlib import Path

from coffee_detector.archive_vadcp import pack_training_arm, restore_training_arm


def test_pack_and_restore_only_training_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for relative, content in {
        "data.yaml": "names: [bean]\n",
        "train/images/a.jpg": "image",
        "train/labels/a.txt": "0 0.5 0.5 1 1\n",
        "metadata/generation_manifest.json": json.dumps(
            {"include_real_train": False}
        ),
        "val/images/val.jpg": "do not archive",
        "test/images/test.jpg": "do not archive",
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    archive = pack_training_arm(source, tmp_path / "arm.tar", progress_every=1)
    restored = restore_training_arm(
        archive, tmp_path / "restored", progress_every=1
    )

    assert (restored / "train" / "images" / "a.jpg").read_text() == "image"
    assert (restored / "metadata" / "generation_manifest.json").is_file()
    assert not (restored / "val").exists()
    assert not (restored / "test").exists()
    assert archive.with_suffix(".tar.json").is_file()
