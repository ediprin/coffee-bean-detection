from __future__ import annotations

import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import yaml

from coffee_detector.af2_rn.audit import sha256
from coffee_detector.experiments.prepare_af2_spectral_kaggle import (
    MANIFEST_FORMAT,
    MANIFEST_NAME,
)
from coffee_detector.experiments.prepare_faruq_v3_kaggle import (
    ARCHIVE_BYTES,
    ARCHIVE_NAME,
    ARCHIVE_SHA256,
    DATASET_DIRNAME,
    EXPECTED_ANNOTATIONS,
    EXPECTED_CLASSES,
    EXPECTED_IMAGES,
    IMAGE_SUFFIXES,
)


D0_NAME = "D0_seed42_best.pt"


def _unique_file(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Harus ada tepat satu {name} di Kaggle Input; ditemukan {matches}"
        )
    return matches[0]


def _train_inventory(data_root: Path) -> dict:
    image_root = data_root / "train/images"
    label_root = data_root / "train/labels"
    images = sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in image_root.iterdir()
    )
    label_files = sorted(label_root.glob("*.txt"))
    counts: Counter[int] = Counter()
    for path in label_files:
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            fields = raw.split()
            if len(fields) != 5:
                raise RuntimeError(
                    f"Label YOLO train harus memiliki 5 kolom: {path}:{line_number}"
                )
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
            if class_id not in EXPECTED_CLASSES:
                raise RuntimeError(f"Class id train di luar SNI-21: {path}:{line_number}")
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                raise RuntimeError(f"Koordinat train di luar [0,1]: {path}:{line_number}")
            if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                raise RuntimeError(f"Box train kosong: {path}:{line_number}")
            counts[class_id] += 1
    result = {
        "images": images,
        "labels": len(label_files),
        "annotations": sum(counts.values()),
        "classes": sorted(counts),
    }
    expected = {
        "images": EXPECTED_IMAGES["train"],
        "labels": EXPECTED_IMAGES["train"],
        "annotations": EXPECTED_ANNOTATIONS["train"],
        "classes": sorted(EXPECTED_CLASSES),
    }
    if result != expected:
        raise RuntimeError(f"Kontrak train Faruq-v3 gagal: {result} != {expected}")
    return result


def prepare_af2rn_kaggle_input(
    input_root: str | Path,
    work_root: str | Path,
) -> tuple[Path, Path, dict]:
    """Validate spectral-v2 input and prepare Faruq-v3 without reading validation.

    The immutable archive SHA proves the complete archive identity. This helper
    deliberately inventories only ``train`` so the AF2RN observability stage can
    truthfully report that no validation file was read.
    """

    input_root = Path(input_root).expanduser().resolve()
    work_root = Path(work_root).expanduser().resolve()
    if not input_root.is_dir() or not work_root.is_dir():
        raise FileNotFoundError(f"Input/work root tidak tersedia: {input_root}, {work_root}")

    manifest_path = _unique_file(input_root, MANIFEST_NAME)
    archive = _unique_file(input_root, ARCHIVE_NAME)
    d0_checkpoint = _unique_file(input_root, D0_NAME)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise RuntimeError(
            "Kaggle Input bukan spectral-v2. Refresh private dataset "
            "faruq-v3-experiment-core-v1 ke versi terbaru."
        )
    if manifest.get("test_images_included") is not False:
        raise RuntimeError("Bundle Kaggle tidak mempertahankan test lock")

    artifacts = manifest.get("artifacts", {})
    proofs = manifest.get("checkpoint_validation", {})
    resolved_hashes = {
        ARCHIVE_NAME: sha256(archive),
        D0_NAME: sha256(d0_checkpoint),
    }
    for name, path in ((ARCHIVE_NAME, archive), (D0_NAME, d0_checkpoint)):
        contract = artifacts.get(name)
        if not isinstance(contract, dict):
            raise RuntimeError(f"Manifest tidak memiliki kontrak {name}")
        if path.stat().st_size != int(contract.get("bytes", -1)):
            raise RuntimeError(f"Ukuran {name} tidak cocok manifest")
        if resolved_hashes[name] != contract.get("sha256"):
            raise RuntimeError(f"SHA256 {name} tidak cocok manifest")
    if archive.stat().st_size != ARCHIVE_BYTES or resolved_hashes[ARCHIVE_NAME] != ARCHIVE_SHA256:
        raise RuntimeError("Arsip Faruq-v3 tidak cocok kontrak immutable repository")
    d0_proof = proofs.get(D0_NAME, {})
    if (
        d0_proof.get("loadable_by_ultralytics") is not True
        or int(d0_proof.get("nc", -1)) != 21
        or int(d0_proof.get("bytes", -1)) != d0_checkpoint.stat().st_size
        or d0_proof.get("sha256") != resolved_hashes[D0_NAME]
    ):
        raise RuntimeError("D0 seed-42 tidak cocok bukti load-test SNI-21 di manifest")

    data_root = work_root / DATASET_DIRNAME
    if data_root.exists():
        resolved = data_root.resolve()
        if resolved.parent != work_root:
            raise RuntimeError(f"Menolak membersihkan path di luar work root: {resolved}")
        shutil.rmtree(data_root)
    with tarfile.open(archive, "r:*") as stream:
        stream.extractall(work_root, filter="data")

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
    if summary.get("format") != "coffee_detector.faruq_grouped_development.v1":
        raise RuntimeError("Format grouped summary tidak dikenal")
    if summary.get("images_by_split") != EXPECTED_IMAGES:
        raise RuntimeError("Grouped summary tidak cocok dengan image contract")
    if summary.get("annotations_by_split") != EXPECTED_ANNOTATIONS:
        raise RuntimeError("Grouped summary tidak cocok dengan annotation contract")
    if not summary.get("training_ready") or not summary.get("test_locked"):
        raise RuntimeError("Grouped summary tidak mempertahankan development/test gate")
    if not all(bool(value) for value in summary.get("gates", {}).values()):
        raise RuntimeError("Grouped dataset gate tidak seluruhnya PASS")

    train_contract = _train_inventory(data_root)
    # The validation contract comes from the immutable archive SHA and frozen
    # grouped summary. No validation image or label is opened by this helper.
    val_contract = {
        "images": EXPECTED_IMAGES["val"],
        "labels": EXPECTED_IMAGES["val"],
        "annotations": EXPECTED_ANNOTATIONS["val"],
        "classes": sorted(EXPECTED_CLASSES),
        "source": "immutable_archive_sha_and_grouped_summary",
    }
    contract = {
        "format": "coffee_detector.faruq_v3_kaggle_input_contract.v1",
        "protocol": "faruq-v3-af2rn-train-only-preparation-v1",
        "archive": str(archive),
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "d0_checkpoint": str(d0_checkpoint),
        "d0_checkpoint_sha256": resolved_hashes[D0_NAME],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "data_root": str(data_root),
        "yaml_path": str(yaml_path),
        "yaml_resolved_path": dataset_yaml["path"],
        "splits": {"train": train_contract, "val": val_contract},
        "validation_files_read": False,
        "test_images_accessed": False,
        "decision": "PASS",
    }
    (data_root / "kaggle_input_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    return data_root, d0_checkpoint, contract
