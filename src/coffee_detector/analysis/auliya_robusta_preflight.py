"""Audit the Auliya Robusta SNI v2 inputs without training a model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image

from coffee_detector.analysis.public_dataset_eligibility import _sha256


METADATA_SHA256 = "798bb53b0cfad35dfcfbcc463fbcb79ff4af61f9807803d6bcc65d37f7c8feaa"
MANIFEST = "metadata/split_manifest_v4.csv"
CLASS_COUNT = 21
REUSED_OBJECT_CLASSES = (18, 19, 20)
GROUP_BY_VARIATION = {"V1": "A", "V2": "A", "V3": "B", "V4": "B", "V5": "C"}
ROLES = ("train", "val", "test")
ORIGINAL_SPLITS = {"train", "valid", "test"}


def _read_text(bundle: ZipFile, member: str) -> str:
    return bundle.read(member).decode("utf-8-sig")


def _load_manifest(bundle: ZipFile) -> dict[str, dict]:
    manifest = {}
    for row in csv.DictReader(_read_text(bundle, MANIFEST).splitlines()):
        label = PurePosixPath(row["file"])
        original_split = row["split"]
        group = row["kelompok"]
        objects = int(row["n"])
        if original_split not in ORIGINAL_SPLITS or label.name != str(label) or label.suffix != ".txt":
            raise ValueError(f"Invalid manifest path: {original_split}/{label}")
        if group == "-":
            if row["var"] != "kosong" or objects != 0:
                raise ValueError(f"Invalid excluded image: {label}")
        elif GROUP_BY_VARIATION.get(row["var"]) != group or objects < 1:
            raise ValueError(f"Invalid variation or object count: {label}")
        image = f"{original_split}/images/{label.with_suffix('.jpg')}"
        if image in manifest:
            raise ValueError(f"Duplicate manifest image: {image}")
        manifest[image] = {"group": group, "objects": objects}
    return manifest


def _load_folds(bundle: ZipFile, manifest: dict[str, dict]) -> dict:
    folds = {}
    included = {image for image, row in manifest.items() if row["group"] != "-"}
    for group in "ABC":
        roles = {}
        seen = set()
        for role in ROLES:
            lines = _read_text(bundle, f"splits/v4_uji{group}_{role}.txt").splitlines()
            images = [line.strip().removeprefix("./") for line in lines if line.strip()]
            unique = set(images)
            if not images or len(unique) != len(images) or seen & unique or not unique <= included:
                raise ValueError(f"Empty, duplicate, overlapping, or unknown image in fold {group}/{role}")
            if any((manifest[image]["group"] == group) != (role == "test") for image in images):
                raise ValueError(f"Acquisition group crosses the held-out boundary in fold {group}/{role}")
            roles[role] = images
            seen.update(unique)
        if seen != included:
            raise ValueError(f"Fold {group} does not cover the eligible manifest")
        folds[group] = roles
    return folds


def _archive_members(bundle: ZipFile) -> dict:
    members = {}
    for entry in bundle.infolist():
        if entry.is_dir():
            continue
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
            raise ValueError(f"Invalid ZIP path: {entry.filename}")
        if len(path.parts) < 3 or path.parts[-3] not in ORIGINAL_SPLITS:
            continue
        kind = path.parts[-2]
        if (kind, path.suffix.lower()) not in {("images", ".jpg"), ("labels", ".txt")}:
            continue
        canonical = "/".join(path.parts[-3:])
        if canonical in members:
            raise ValueError(f"Duplicate dataset member: {canonical}")
        members[canonical] = entry
    return members


def _label_name(image: str) -> str:
    split, _, file = image.split("/")
    return f"{split}/labels/{PurePosixPath(file).with_suffix('.txt')}"


def _polygon_counts(text: str, path: str) -> Counter:
    counts = Counter()
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
            points = [float(value) for value in fields[1:]]
            if (
                not 0 <= class_id < CLASS_COUNT
                or len(points) < 6
                or len(points) % 2
                or not all(math.isfinite(value) and 0 <= value <= 1 for value in points)
                or len(set(points[::2])) < 2
                or len(set(points[1::2])) < 2
            ):
                raise ValueError
        except ValueError as error:
            raise ValueError(f"Invalid YOLO polygon at {path}:{line_number}") from error
        counts[class_id] += 1
    return counts


def _audit_archive(bundle: ZipFile, manifest: dict, roles: dict) -> dict:
    entries = _archive_members(bundle)
    images = {path for path in entries if "/images/" in path}
    labels = {path for path in entries if "/labels/" in path}
    required_labels = {_label_name(path) for path, row in manifest.items() if row["objects"]}
    missing_images = set(manifest) - images
    extra_images = images - set(manifest)
    missing_labels = required_labels - labels
    unexpected_labels = labels - {_label_name(path) for path in manifest}
    empty_entries = [path for path in (images | required_labels) & entries.keys() if entries[path].file_size == 0]
    if missing_images or extra_images or missing_labels or unexpected_labels or empty_entries:
        raise ValueError(
            "Image/label ZIP differs from the v2 manifest "
            f"(missing images={len(missing_images)}, extra images={len(extra_images)}, "
            f"missing labels={len(missing_labels)}, extra labels={len(unexpected_labels)}, "
            f"empty files={len(empty_entries)}). Obtain the original annotated v2 export."
        )
    counts_by_role = {}
    for role in ("train", "val"):
        counts = Counter()
        for image_path in roles[role]:
            with bundle.open(entries[image_path]) as stream, Image.open(stream) as image:
                if image.format != "JPEG" or tuple(sorted(image.size)) != (3000, 4000):
                    raise ValueError(f"Expected original 3000×4000 JPEG: {image_path}")
            label = _label_name(image_path)
            instances = _polygon_counts(bundle.read(entries[label]).decode("utf-8-sig"), label)
            if instances.total() != manifest[image_path]["objects"]:
                raise ValueError(f"Polygon count differs from manifest: {label}")
            counts.update(instances)
        counts_by_role[role] = {class_id: counts[class_id] for class_id in range(CLASS_COUNT)}
    return counts_by_role


def run_preflight(
    metadata_zip: Path, images_zip: Path | None = None, *, held_out_group: str = "A"
) -> dict:
    if _sha256(metadata_zip) != METADATA_SHA256:
        raise ValueError("Metadata ZIP does not match Zenodo 24346584 v2")
    with ZipFile(metadata_zip) as metadata:
        manifest = _load_manifest(metadata)
        folds = _load_folds(metadata, manifest)
    if held_out_group not in folds:
        raise ValueError(f"Unknown held-out group: {held_out_group}")
    counts = None
    if images_zip is not None:
        with ZipFile(images_zip) as images:
            counts = _audit_archive(images, manifest, folds[held_out_group])
    return {
        "dataset_doi": "10.5281/zenodo.24346584",
        "manifest_images": len(manifest),
        "excluded_unannotated": sum(row["group"] == "-" for row in manifest.values()),
        "official_fold_counts": {
            group: {role: len(paths) for role, paths in roles.items()}
            for group, roles in folds.items()
        },
        "held_out_group": held_out_group,
        "reused_object_class_ids": list(REUSED_OBJECT_CLASSES),
        "train_validation_are_image_level_split_within_development_groups": True,
        "development_polygon_instances_by_class": counts,
        "missing_class_ids_by_role": (
            None if counts is None else {
                role: [class_id for class_id, count in values.items() if count == 0]
                for role, values in counts.items()
            }
        ),
        "held_out_contents_read": False,
        "status": "METADATA_ONLY" if counts is None else "DEVELOPMENT_ARCHIVE_AUDITED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-zip", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path)
    parser.add_argument("--held-out-group", choices=("A", "B", "C"), default="A")
    args = parser.parse_args()
    print(json.dumps(run_preflight(
        args.metadata_zip, args.images_zip, held_out_group=args.held_out_group
    ), indent=2))


if __name__ == "__main__":
    main()
