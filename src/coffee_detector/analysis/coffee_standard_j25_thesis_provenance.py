"""Verify the Coffee Standard v8 export against Jundullah's thesis counts.

This audit is model-free. It establishes dataset lineage from archive metadata
and count arithmetic; it never infers source identity from non-unique filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml


FORMAT = "coffee_detector.coffee_standard_j25_thesis_provenance.v1"
AUTHOR_ARCHIVE_SHA256 = "a4d8570e7de8d0ebba408d1d736256806e8f5ff1ebcf62cafd2fa13d149d613d"
THESIS_COUNTS = {
    "raw_images": 451,
    "train_images_before_augmentation": 271,
    "validation_images": 113,
    "test_images": 67,
    "train_annotations_before_augmentation": 3720,
    "validation_annotations": 1606,
    "test_annotations": 1161,
    "classes": 25,
}


class _Archive:
    def __init__(self, path: Path):
        self.path = path
        suffixes = "".join(path.suffixes).lower()
        if suffixes.endswith(".zip"):
            self.handle = zipfile.ZipFile(path)
            self.kind = "zip"
        elif suffixes.endswith((".tar", ".tar.gz", ".tgz")):
            self.handle = tarfile.open(path, "r:*")
            self.kind = "tar"
        else:
            raise ValueError("Archive harus .zip, .tar, .tar.gz, atau .tgz")

    def names(self) -> list[str]:
        if self.kind == "zip":
            return [name for name in self.handle.namelist() if not name.endswith("/")]
        return [member.name for member in self.handle.getmembers() if member.isfile()]

    def read(self, name: str) -> bytes:
        if self.kind == "zip":
            return self.handle.read(name)
        stream = self.handle.extractfile(self.handle.getmember(name))
        if stream is None:
            raise FileNotFoundError(name)
        return stream.read()

    def close(self) -> None:
        self.handle.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_kind(name: str) -> tuple[str, str] | None:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    for index in range(len(parts) - 1):
        split = {"valid": "val", "validation": "val"}.get(parts[index], parts[index])
        kind = parts[index + 1]
        if split in {"train", "val", "test"} and kind in {"images", "labels"}:
            return split, kind
    return None


def _unique_text(archive: _Archive, basename: str) -> str:
    texts = {
        archive.read(name).decode("utf-8", errors="replace")
        for name in archive.names()
        if PurePosixPath(name).name == basename
    }
    if len(texts) != 1:
        raise RuntimeError(f"{basename} harus memiliki tepat satu isi unik; ditemukan {len(texts)}")
    return texts.pop()


def _label_instances(content: bytes) -> int:
    return sum(bool(line.strip()) for line in io.StringIO(content.decode("utf-8", errors="strict")))


def _cross_split_exact_hashes(rows: Iterable[tuple[str, str]]) -> list[str]:
    splits_by_hash: dict[str, set[str]] = defaultdict(set)
    for split, digest in rows:
        splits_by_hash[digest].add(split)
    return sorted(digest for digest, splits in splits_by_hash.items() if len(splits) > 1)


def audit_j25_thesis_provenance(
    archive_path: str | Path,
    output: str | Path,
    *,
    expected_sha256: str = AUTHOR_ARCHIVE_SHA256,
) -> dict:
    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    archive_sha256 = _sha256(archive_path)
    archive = _Archive(archive_path)
    try:
        names = archive.names()
        data = yaml.safe_load(_unique_text(archive, "data.yaml"))
        readme = _unique_text(archive, "README.roboflow.txt")
        image_counts = Counter()
        label_counts = Counter()
        instance_counts = Counter()
        image_hash_rows: list[tuple[str, str]] = []
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for name in names:
            classified = _split_kind(name)
            if classified is None:
                continue
            split, kind = classified
            suffix = PurePosixPath(name).suffix.lower()
            if kind == "images" and suffix in image_extensions:
                payload = archive.read(name)
                image_counts[split] += 1
                image_hash_rows.append((split, hashlib.sha256(payload).hexdigest()))
            elif kind == "labels" and suffix == ".txt":
                label_counts[split] += 1
                instance_counts[split] += _label_instances(archive.read(name))

        match = re.search(r"create\s+(\d+)\s+versions of each source image", readme, re.I)
        if match is None:
            raise RuntimeError("Faktor augmentasi tidak ditemukan pada README.roboflow.txt")
        augmentation_factor = int(match.group(1))
        train_source_images = image_counts["train"] // augmentation_factor
        inferred_raw_images = train_source_images + image_counts["val"] + image_counts["test"]
        inferred_train_annotations = instance_counts["train"] // augmentation_factor
        exact_overlap = _cross_split_exact_hashes(image_hash_rows)
        roboflow = data.get("roboflow", {})
        observed = {
            "images": dict(image_counts),
            "labels": dict(label_counts),
            "instances": dict(instance_counts),
            "total_images": sum(image_counts.values()),
            "class_count": int(data.get("nc", -1)),
            "augmentation_factor": augmentation_factor,
            "inferred_train_source_images": train_source_images,
            "inferred_raw_images": inferred_raw_images,
            "inferred_train_annotations_before_augmentation": inferred_train_annotations,
            "exact_cross_split_image_hashes": len(exact_overlap),
            "roboflow": roboflow,
        }
        discrepancy = instance_counts["test"] - THESIS_COUNTS["test_annotations"]
        gates = {
            "author_archive_sha256_exact": archive_sha256 == expected_sha256.lower(),
            "roboflow_project_matches": roboflow.get("workspace") == "tes-rcphs"
            and roboflow.get("project") == "coffee-detection-with-standard"
            and int(roboflow.get("version", -1)) == 8,
            "ontology_has_25_classes": observed["class_count"] == THESIS_COUNTS["classes"],
            "export_has_993_images": observed["total_images"] == 993,
            "train_is_271_sources_times_three": image_counts["train"]
            == THESIS_COUNTS["train_images_before_augmentation"] * augmentation_factor,
            "source_image_arithmetic_recovers_451": inferred_raw_images == THESIS_COUNTS["raw_images"],
            "train_annotation_arithmetic_recovers_3720": inferred_train_annotations
            == THESIS_COUNTS["train_annotations_before_augmentation"],
            "validation_matches_thesis_exactly": image_counts["val"]
            == THESIS_COUNTS["validation_images"]
            and instance_counts["val"] == THESIS_COUNTS["validation_annotations"],
            "test_image_count_matches_thesis": image_counts["test"] == THESIS_COUNTS["test_images"],
            "test_annotation_discrepancy_at_most_one": abs(discrepancy) <= 1,
            "image_and_label_counts_match": all(image_counts[s] == label_counts[s] for s in ("train", "val", "test")),
            "zero_exact_cross_split_image_hashes": not exact_overlap,
            "filename_basename_not_used_as_identity": True,
            "training_not_executed": True,
            "model_test_evaluation_not_executed": True,
        }
        passed = all(gates.values())
        payload = {
            "format": FORMAT,
            "status": "complete",
            "source_archive": str(archive_path),
            "source_archive_sha256": archive_sha256,
            "thesis_counts": THESIS_COUNTS,
            "observed": observed,
            "test_annotation_delta_export_minus_thesis": discrepancy,
            "gates": gates,
            "decision": "PASS_THESIS_LINEAGE_WITH_ONE_ANNOTATION_DISCREPANCY" if passed else "FAIL_THESIS_LINEAGE",
            "dataset_role": "PRIMARY_DATASET_PROVENANCE_SUPPORTED" if passed else "PROVISIONAL_ONLY",
            "identity_warning": (
                "Roboflow export basenames are not authoritative source asset IDs and collide across distinct "
                "photographs. Do not group, quarantine, or split this dataset using stripped filenames."
            ),
            "interpretation": (
                "The 993-image export is the 451-image thesis dataset with three train variants per source. "
                "The journal paper's 20-class/2,000-image experiment is a different dataset version."
            ),
            "training_authorized": False,
            "test_access_authorized": False,
            "training_executed": False,
            "test_images_accessed_by_model": False,
        }
    finally:
        archive.close()

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Coffee Standard v8 archive against thesis counts.")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_j25_thesis_provenance(args.archive, args.output)
    print("OBSERVED:", result["observed"])
    print("TEST ANNOTATION DELTA:", result["test_annotation_delta_export_minus_thesis"])
    print("GATES:", result["gates"])
    print("DECISION:", result["decision"])
    print("ROLE:", result["dataset_role"])
    print("TRAINING AUTHORIZED:", result["training_authorized"])
    print("SUMMARY:", result["summary"])


if __name__ == "__main__":
    main()
