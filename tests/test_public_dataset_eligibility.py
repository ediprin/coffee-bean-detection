import hashlib
import tarfile
from pathlib import Path

import yaml
from PIL import Image

from coffee_detector.analysis.public_dataset_eligibility import (
    REGISTRY_FORMAT,
    audit_public_dataset_registry,
    extract_audit_archive,
)


def _write_dataset(root: Path, offset: int, *, train_only: bool = False) -> None:
    splits = ("train",) if train_only else ("train", "valid", "test")
    names = {0: "normal", 1: "defect"}
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": names,
            }
        ),
        encoding="utf-8",
    )
    for split_index, split in enumerate(splits):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for image_index in range(2):
            stem = f"{split}-{image_index}-{offset}"
            pixels = [
                ((x * 13 + offset) % 255, (y * 17 + split_index) % 255, (x * y + image_index) % 255)
                for y in range(16)
                for x in range(16)
            ]
            image = Image.new("RGB", (16, 16))
            image.putdata(pixels)
            image.save(root / split / "images" / f"{stem}.png")
            (root / split / "labels" / f"{stem}.txt").write_text(
                f"{image_index % 2} 0.5 0.5 0.4 0.4\n", encoding="utf-8"
            )


def _archive(tmp_path: Path, code: str) -> tuple[Path, str]:
    path = tmp_path / f"{code}.tar"
    path.write_bytes(f"frozen-{code}".encode())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(tmp_path: Path, code: str, root: Path, *, ambiguous: list[str] | None = None) -> dict:
    archive, sha256 = _archive(tmp_path, code)
    return {
        "code": code,
        "task": "object_detection",
        "owner": f"owner-{code}",
        "project": f"project-{code}",
        "version": 1,
        "source_url": f"https://example.test/{code}/1",
        "license": "CC BY 4.0",
        "archive_path": str(archive),
        "archive_sha256": sha256,
        "dataset_root": str(root),
        "declared_augmentation": "none",
        "ambiguous_classes": ambiguous or [],
    }


def _run(
    tmp_path: Path,
    specs: list[dict],
    *,
    minimum: int = 3,
    minimum_source_images: int = 1,
) -> dict:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "format": REGISTRY_FORMAT,
                "minimum_independent_lineages": minimum,
                "minimum_source_images": minimum_source_images,
                "datasets": specs,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return audit_public_dataset_registry(
        registry, tmp_path / "reports", near_threshold=-1, progress=False
    )


def test_three_clean_datasets_pass_v2_gate(tmp_path: Path) -> None:
    specs = []
    for index, code in enumerate(("a", "b", "c"), start=1):
        root = tmp_path / code
        _write_dataset(root, index * 31)
        specs.append(_spec(tmp_path, code, root))

    report = _run(tmp_path, specs)

    assert report["decision"] == "PASS_V2_DATASET_GATE"
    assert report["eligible_lineage_count"] == 3
    assert report["training_authorized"] is False
    assert (tmp_path / "reports" / "public_dataset_eligibility_summary.md").is_file()


def test_exact_cross_dataset_copy_reduces_independent_lineages(tmp_path: Path) -> None:
    specs = []
    roots = {}
    for index, code in enumerate(("a", "b", "c"), start=1):
        root = tmp_path / code
        _write_dataset(root, index * 37)
        roots[code] = root
        specs.append(_spec(tmp_path, code, root))
    source = roots["a"] / "train" / "images" / "train-0-37.png"
    target = roots["b"] / "train" / "images" / "shared.png"
    target.write_bytes(source.read_bytes())
    (roots["b"] / "train" / "labels" / "shared.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )

    report = _run(tmp_path, specs)

    assert report["cross_dataset"]["exact_cross_dataset_groups"] == 1
    assert report["eligible_lineage_count"] == 2
    assert report["decision"] == "FAIL_MINIMUM_INDEPENDENT_LINEAGES"


def test_train_only_dataset_requires_grouped_rebuild(tmp_path: Path) -> None:
    root = tmp_path / "train-only"
    _write_dataset(root, 19, train_only=True)

    report = _run(tmp_path, [_spec(tmp_path, "train_only", root)], minimum=1)

    assert report["datasets"][0]["status"] == "REBUILD_GROUPED_SPLIT"
    assert "missing_validation_or_test_split" in report["datasets"][0]["reasons"]
    assert report["decision"] == "PASS_V2_DATASET_GATE"


def test_ambiguous_class_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ambiguous"
    _write_dataset(root, 23)

    report = _run(
        tmp_path,
        [_spec(tmp_path, "ambiguous", root, ambiguous=["objects"])],
        minimum=1,
    )

    assert report["datasets"][0]["status"] == "REJECT"
    assert report["decision"] == "FAIL_DATASET_ELIGIBILITY"


def test_wrong_archive_sha_is_held(tmp_path: Path) -> None:
    root = tmp_path / "wrong-sha"
    _write_dataset(root, 29)
    spec = _spec(tmp_path, "wrong_sha", root)
    spec["archive_sha256"] = "0" * 64

    report = _run(tmp_path, [spec], minimum=1)

    assert report["datasets"][0]["status"] == "HOLD_METADATA"
    assert report["datasets"][0]["metadata_gates"]["archive_sha256_verified"] is False
    assert report["decision"] == "HOLD_METADATA_VERIFICATION"


def test_tiny_version_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "tiny"
    _write_dataset(root, 41)

    report = _run(
        tmp_path,
        [_spec(tmp_path, "tiny", root)],
        minimum=1,
        minimum_source_images=100,
    )

    assert report["datasets"][0]["status"] == "REJECT"
    assert "insufficient_independent_source_images" in report["datasets"][0]["reasons"]


def test_archive_extraction_is_hash_bound_and_nested_root_is_discovered(tmp_path: Path) -> None:
    source = tmp_path / "source" / "nested"
    _write_dataset(source, 47)
    archive = tmp_path / "dataset.tar"
    with tarfile.open(archive, "w") as bundle:
        bundle.add(source, arcname="export/nested")
    target = tmp_path / "extracted"

    assert extract_audit_archive(archive, target) == target
    assert extract_audit_archive(archive, target) == target
    spec = _spec(tmp_path, "nested", target)
    spec["archive_path"] = str(archive)
    spec["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = _run(tmp_path, [spec], minimum=1)

    assert report["datasets"][0]["status"] == "PASS_AS_IS"
    assert report["datasets"][0]["dataset_root"].endswith("export\\nested") or report["datasets"][0]["dataset_root"].endswith("export/nested")
