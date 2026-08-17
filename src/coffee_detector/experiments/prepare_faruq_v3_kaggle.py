from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import yaml


ARCHIVE_NAME = "faruq-development-v3-grouped.tar.bin"
ARCHIVE_BYTES = 1_178_275_840
ARCHIVE_SHA256 = "357b23d03058581af087d308584a967a0c386608deb011689dbb0773104224ec"
DATASET_DIRNAME = "faruq-development-v3-grouped"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_IMAGES = {"train": 1665, "val": 294}
EXPECTED_ANNOTATIONS = {"train": 2986, "val": 526}
EXPECTED_CLASSES = set(range(21))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_file(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Harus ada tepat satu {name} di Kaggle Input; ditemukan {matches}"
        )
    return matches[0]


def _label_inventory(label_root: Path) -> tuple[int, Counter[int]]:
    files = sorted(label_root.glob("*.txt"))
    counts: Counter[int] = Counter()
    for path in files:
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            fields = raw.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise RuntimeError(
                    f"Label YOLO harus memiliki 5 kolom: {path}:{line_number}"
                )
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
            if class_id not in range(21):
                raise RuntimeError(f"Class id di luar SNI-21: {path}:{line_number}")
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                raise RuntimeError(f"Koordinat di luar [0,1]: {path}:{line_number}")
            if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                raise RuntimeError(f"Box kosong: {path}:{line_number}")
            counts[class_id] += 1
    return len(files), counts


def _image_count(image_root: Path) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in image_root.iterdir()
    )


def prepare_faruq_v3_kaggle_input(
    input_root: str | Path,
    work_root: str | Path,
) -> tuple[Path, dict]:
    """Extract and validate the immutable Faruq-v3 Kaggle input before GPU work."""

    input_root = Path(input_root).expanduser().resolve()
    work_root = Path(work_root).expanduser().resolve()
    if not input_root.is_dir() or not work_root.is_dir():
        raise FileNotFoundError(f"Input/work root tidak tersedia: {input_root}, {work_root}")

    archive = _unique_file(input_root, ARCHIVE_NAME)
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise RuntimeError(
            f"Ukuran arsip salah: {archive.stat().st_size}, diharapkan {ARCHIVE_BYTES}"
        )
    archive_sha256 = _sha256(archive)
    if archive_sha256 != ARCHIVE_SHA256:
        raise RuntimeError(
            f"SHA256 arsip salah: {archive_sha256}, diharapkan {ARCHIVE_SHA256}"
        )

    data_root = work_root / DATASET_DIRNAME
    if data_root.exists():
        resolved = data_root.resolve()
        if resolved.parent != work_root:
            raise RuntimeError(f"Menolak membersihkan path di luar work root: {resolved}")
        shutil.rmtree(data_root)

    with tarfile.open(archive, "r:*") as handle:
        handle.extractall(work_root, filter="data")

    yaml_path = data_root / "data.yaml"
    summary_path = data_root / "faruq_grouped_summary.json"
    if not yaml_path.is_file() or not summary_path.is_file():
        raise RuntimeError(f"Arsip tidak menghasilkan dataset Faruq-v3 lengkap: {data_root}")
    if (data_root / "test").exists():
        raise RuntimeError("Development archive tidak boleh mengekspos test")

    dataset_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    dataset_yaml["path"] = str(data_root)
    yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_contract: dict[str, dict] = {}
    for split in ("train", "val"):
        image_root = data_root / split / "images"
        label_root = data_root / split / "labels"
        images = _image_count(image_root)
        label_files, class_counts = _label_inventory(label_root)
        if images != EXPECTED_IMAGES[split] or label_files != EXPECTED_IMAGES[split]:
            raise RuntimeError(
                f"Jumlah {split} salah: images={images}, labels={label_files}, "
                f"expected={EXPECTED_IMAGES[split]}"
            )
        if set(class_counts) != EXPECTED_CLASSES:
            raise RuntimeError(
                f"Cakupan kelas {split} salah: {sorted(class_counts)}"
            )
        if sum(class_counts.values()) != EXPECTED_ANNOTATIONS[split]:
            raise RuntimeError(
                f"Jumlah anotasi {split} salah: {sum(class_counts.values())}, "
                f"expected={EXPECTED_ANNOTATIONS[split]}"
            )
        split_contract[split] = {
            "images": images,
            "labels": label_files,
            "annotations": sum(class_counts.values()),
            "classes": sorted(class_counts),
        }

    if summary.get("format") != "coffee_detector.faruq_grouped_development.v1":
        raise RuntimeError("Format grouped summary tidak dikenal")
    if summary.get("images_by_split") != EXPECTED_IMAGES:
        raise RuntimeError("Grouped summary tidak cocok dengan kontrak image split")
    if summary.get("annotations_by_split") != EXPECTED_ANNOTATIONS:
        raise RuntimeError("Grouped summary tidak cocok dengan kontrak anotasi")
    if not summary.get("training_ready") or not summary.get("test_locked"):
        raise RuntimeError("Grouped summary tidak mengizinkan development training")
    if not all(bool(value) for value in summary.get("gates", {}).values()):
        raise RuntimeError("Gate grouped dataset tidak seluruhnya PASS")

    contract = {
        "format": "coffee_detector.faruq_v3_kaggle_input_contract.v1",
        "archive": str(archive),
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": archive_sha256,
        "data_root": str(data_root),
        "yaml_path": str(yaml_path),
        "yaml_resolved_path": dataset_yaml["path"],
        "splits": split_contract,
        "test_images_accessed": False,
        "decision": "PASS",
    }
    (data_root / "kaggle_input_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return data_root, contract
