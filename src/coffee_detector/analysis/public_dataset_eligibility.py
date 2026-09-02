"""Leakage-safe eligibility audit for public YOLO detection datasets.

The audit is intentionally model-free.  It reads metadata, images, and labels,
but never imports a detector, trains a model, or evaluates a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image

from coffee_detector.dataset import (
    IMAGE_SUFFIXES,
    ImageRecord,
    collect_records,
    discover_layout,
)


REPORT_FORMAT = "coffee_detector.public_dataset_eligibility.v1"
REGISTRY_FORMAT = "coffee_detector.public_dataset_registry.v1"
ELIGIBLE_AFTER_REBUILD = {"PASS_AS_IS", "REBUILD_GROUPED_SPLIT"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PublicRecord:
    dataset: str
    record: ImageRecord


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_optional_path(value: object, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def extract_audit_archive(archive: str | Path, target: str | Path) -> Path:
    """Safely extract an immutable dataset archive into a hash-bound directory."""
    archive = Path(archive).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    digest = _sha256(archive)
    marker = target / ".audit_archive_sha256"
    if target.exists() and any(target.iterdir()):
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
            return target
        raise FileExistsError(f"Target ekstraksi tidak kosong/tidak cocok: {target}")
    target.mkdir(parents=True, exist_ok=True)

    def safe_destination(name: str) -> Path:
        destination = (target / name).resolve()
        try:
            destination.relative_to(target)
        except ValueError as error:
            raise RuntimeError(f"Path traversal di arsip: {name}") from error
        return destination

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                safe_destination(member.filename)
            bundle.extractall(target)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as bundle:
            for member in bundle.getmembers():
                safe_destination(member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Link tidak diizinkan dalam arsip: {member.name}")
            try:
                bundle.extractall(target, filter="data")
            except TypeError:  # Python 3.10/3.11 compatibility.
                bundle.extractall(target)
    else:
        raise ValueError(f"Format arsip tidak didukung: {archive}")
    marker.write_text(digest + "\n", encoding="utf-8")
    return target


def _resolve_nested_yolo_root(root: Path) -> Path:
    candidates = [root]
    candidates.extend(sorted({path.parent for path in root.rglob("data.yaml")}))
    candidates.extend(sorted({path.parent for path in root.rglob("dataset.yaml")}))
    valid = []
    for candidate in candidates:
        try:
            discover_layout(candidate)
        except (FileNotFoundError, ValueError):
            continue
        valid.append(candidate.resolve())
    valid = sorted(set(valid))
    if len(valid) != 1:
        raise RuntimeError(f"Harus ada tepat satu root YOLO; ditemukan: {valid}")
    return valid[0]


def _metadata_gates(
    spec: dict, archive: Path | None, archive_actual_sha256: str | None
) -> dict[str, bool]:
    expected_sha = str(spec.get("archive_sha256") or "").lower()
    gates = {
        "source_url_present": str(spec.get("source_url") or "").startswith(("http://", "https://")),
        "owner_present": bool(str(spec.get("owner") or "").strip()),
        "project_present": bool(str(spec.get("project") or "").strip()),
        "version_frozen": spec.get("version") not in (None, "", "latest"),
        "license_present": bool(str(spec.get("license") or "").strip()),
        "archive_sha256_declared": bool(SHA256_PATTERN.fullmatch(expected_sha)),
        "archive_present": archive is not None and archive.is_file(),
        "archive_sha256_verified": False,
        "object_detection_task": spec.get("task") == "object_detection",
    }
    if gates["archive_present"] and gates["archive_sha256_declared"]:
        gates["archive_sha256_verified"] = archive_actual_sha256 == expected_sha
    return gates


def _group_cross_split(records: list[ImageRecord], attribute: str) -> list[dict]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[str(getattr(record, attribute))].append(record)
    result = []
    for key, members in groups.items():
        splits = sorted({member.split for member in members})
        if len(splits) > 1:
            result.append(
                {
                    "key": key,
                    "splits": splits,
                    "count": len(members),
                    "files": [str(member.image_path) for member in members[:12]],
                }
            )
    return sorted(result, key=lambda row: (-row["count"], row["key"]))


def _near_pairs(
    records: list[PublicRecord],
    threshold: int,
    *,
    cross_dataset_only: bool = False,
    cross_split_only: bool = False,
) -> list[tuple[int, int, int, float]]:
    """Return dHash candidates; these remain review warnings, never proof."""
    if threshold < 0:
        return []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    pairs: list[tuple[int, int, int, float]] = []
    for index, wrapped in enumerate(records):
        record = wrapped.record
        candidates: set[int] = set()
        for band in range(8):
            candidates.update(buckets[(band, (record.dhash >> (band * 8)) & 0xFF)])
        for other_index in candidates:
            other = records[other_index]
            if cross_dataset_only and other.dataset == wrapped.dataset:
                continue
            if cross_split_only and (
                other.dataset != wrapped.dataset or other.record.split == record.split
            ):
                continue
            if other.record.sha256 == record.sha256:
                continue
            distance = (record.dhash ^ other.record.dhash).bit_count()
            color_distance = max(
                abs(left - right)
                for left, right in zip(record.mean_rgb, other.record.mean_rgb)
            )
            if distance <= threshold and color_distance <= 12.0:
                pairs.append((other_index, index, distance, color_distance))
        for band in range(8):
            buckets[(band, (record.dhash >> (band * 8)) & 0xFF)].append(index)
    return pairs


def _split_rows(records: list[ImageRecord], names: dict[int, str]) -> tuple[dict, list[dict]]:
    splits: dict[str, dict] = {}
    class_boxes: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    by_split: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_split[record.split].append(record)
        class_boxes.update(record.class_counts)
        class_images.update(record.class_counts.keys())
    for split in ("train", "val", "test"):
        members = by_split.get(split, [])
        splits[split] = {
            "images": len(members),
            "boxes": sum(len(member.boxes) for member in members),
            "empty_images": sum(not member.boxes for member in members),
            "parents": len({member.parent_id for member in members}),
        }
    classes = [
        {
            "class_id": class_id,
            "class_name": names[class_id],
            "boxes": class_boxes[class_id],
            "images": class_images[class_id],
        }
        for class_id in sorted(names)
    ]
    return splits, classes


def _image_geometry(records: list[ImageRecord]) -> dict:
    sizes: Counter[str] = Counter()
    failures = []
    for record in records:
        try:
            with Image.open(record.image_path) as image:
                sizes[f"{image.width}x{image.height}"] += 1
        except OSError as error:
            failures.append(f"{record.image_path}: {error}")
    return {
        "unique_resolutions": len(sizes),
        "most_common_resolutions": [
            {"resolution": resolution, "images": count}
            for resolution, count in sizes.most_common(12)
        ],
        "decode_errors": failures[:25],
    }


def _audit_one(
    spec: dict,
    registry_base: Path,
    root_override: Path | None,
    archive_override: Path | None,
    near_threshold: int,
    minimum_source_images: int,
    progress: bool,
) -> tuple[dict, list[ImageRecord]]:
    code = str(spec["code"])
    root = root_override or _resolve_optional_path(spec.get("dataset_root"), registry_base)
    archive = archive_override or _resolve_optional_path(spec.get("archive_path"), registry_base)
    archive_actual_sha256 = _sha256(archive) if archive is not None and archive.is_file() else None
    metadata = _metadata_gates(spec, archive, archive_actual_sha256)
    base = {
        "code": code,
        "owner": spec.get("owner"),
        "project": spec.get("project"),
        "version": spec.get("version"),
        "source_url": spec.get("source_url"),
        "license": spec.get("license"),
        "declared_augmentation": spec.get("declared_augmentation", "unknown"),
        "metadata_gates": metadata,
        "dataset_root": str(root) if root else None,
        "archive_path": str(archive) if archive else None,
        "archive_sha256": spec.get("archive_sha256"),
        "archive_actual_sha256": archive_actual_sha256,
        "ambiguous_classes_declared": list(spec.get("ambiguous_classes") or []),
        "notes": list(spec.get("notes") or []),
        "training_executed": False,
        "model_test_evaluation_executed": False,
    }
    if root is None or not root.is_dir():
        return {**base, "status": "NOT_ACQUIRED", "reasons": ["dataset_root_missing"]}, []
    try:
        root = _resolve_nested_yolo_root(root)
        base["dataset_root"] = str(root)
        layout = discover_layout(root)
    except (FileNotFoundError, ValueError) as error:
        return {**base, "status": "REJECT", "reasons": [str(error)]}, []

    records, errors = collect_records(layout, compute_visual_features=True, progress=progress)
    exact_cross = _group_cross_split(records, "sha256")
    parent_cross = _group_cross_split(records, "parent_id")
    wrapped = [PublicRecord(code, record) for record in records]
    near_cross = _near_pairs(wrapped, near_threshold, cross_split_only=True)
    splits, classes = _split_rows(records, layout.names)
    geometry = _image_geometry(records)
    absent_classes = [row["class_name"] for row in classes if row["boxes"] == 0]
    source_parents = len({record.parent_id for record in records})
    reasons: list[str] = []
    essential_metadata = (
        "source_url_present",
        "owner_present",
        "project_present",
        "version_frozen",
        "license_present",
        "archive_sha256_declared",
        "archive_present",
        "archive_sha256_verified",
        "object_detection_task",
    )
    if errors or geometry["decode_errors"]:
        status = "REJECT"
        reasons.append("invalid_images_or_labels")
    elif absent_classes:
        status = "REJECT"
        reasons.append("classes_without_instances")
    elif source_parents < minimum_source_images:
        status = "REJECT"
        reasons.append("insufficient_independent_source_images")
    elif base["ambiguous_classes_declared"]:
        status = "REJECT"
        reasons.append("ambiguous_or_placeholder_classes")
    elif not all(metadata[key] for key in essential_metadata):
        status = "HOLD_METADATA"
        reasons.extend(key for key in essential_metadata if not metadata[key])
    elif exact_cross or parent_cross or not splits["val"]["images"] or not splits["test"]["images"]:
        status = "REBUILD_GROUPED_SPLIT"
        if exact_cross:
            reasons.append("exact_cross_split_duplicates")
        if parent_cross:
            reasons.append("roboflow_parent_cross_split")
        if not splits["val"]["images"] or not splits["test"]["images"]:
            reasons.append("missing_validation_or_test_split")
    elif near_cross:
        status = "REVIEW_NEAR_DUPLICATES"
        reasons.append("perceptual_cross_split_candidates")
    else:
        status = "PASS_AS_IS"

    near_examples = [
        {
            "left": str(wrapped[left].record.image_path),
            "right": str(wrapped[right].record.image_path),
            "dhash_distance": distance,
            "max_mean_rgb_distance": color_distance,
        }
        for left, right, distance, color_distance in near_cross[:50]
    ]
    return {
        **base,
        "status": status,
        "reasons": reasons,
        "yaml": str(layout.yaml_path),
        "class_count": len(layout.names),
        "images": len(records),
        "boxes": sum(len(record.boxes) for record in records),
        "estimated_source_parents": source_parents,
        "splits": splits,
        "class_distribution": classes,
        "classes_without_boxes": absent_classes,
        "label_or_image_errors": errors[:50],
        "geometry": geometry,
        "exact_cross_split_groups": len(exact_cross),
        "parent_cross_split_groups": len(parent_cross),
        "near_cross_split_candidates": len(near_cross),
        "exact_cross_split_examples": exact_cross[:25],
        "parent_cross_split_examples": parent_cross[:25],
        "near_cross_split_examples": near_examples,
    }, records


def _cross_dataset_analysis(
    records_by_dataset: dict[str, list[ImageRecord]], near_threshold: int
) -> dict:
    wrapped = [
        PublicRecord(code, record)
        for code in sorted(records_by_dataset)
        for record in records_by_dataset[code]
    ]
    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(wrapped):
        exact_groups[item.record.sha256].append(index)
    exact = []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for sha256, indices in exact_groups.items():
        datasets = sorted({wrapped[index].dataset for index in indices})
        if len(datasets) < 2:
            continue
        exact.append(
            {
                "sha256": sha256,
                "datasets": datasets,
                "files": [str(wrapped[index].record.image_path) for index in indices[:12]],
            }
        )
        for left in datasets:
            adjacency[left].update(dataset for dataset in datasets if dataset != left)
    near = _near_pairs(wrapped, near_threshold, cross_dataset_only=True)
    near_examples = []
    near_dataset_pairs: Counter[tuple[str, str]] = Counter()
    for left, right, distance, color_distance in near:
        left_item, right_item = wrapped[left], wrapped[right]
        pair = tuple(sorted((left_item.dataset, right_item.dataset)))
        near_dataset_pairs[pair] += 1
        if len(near_examples) < 100:
            near_examples.append(
                {
                    "left_dataset": left_item.dataset,
                    "right_dataset": right_item.dataset,
                    "left": str(left_item.record.image_path),
                    "right": str(right_item.record.image_path),
                    "dhash_distance": distance,
                    "max_mean_rgb_distance": color_distance,
                }
            )

    candidates = sorted(records_by_dataset)
    lineage_components: list[list[str]] = []
    seen: set[str] = set()
    for code in candidates:
        if code in seen:
            continue
        stack, component = [code], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - seen))
        lineage_components.append(sorted(component))
    return {
        "exact_cross_dataset_groups": len(exact),
        "exact_cross_dataset_examples": exact[:100],
        "near_cross_dataset_candidates": len(near),
        "near_cross_dataset_examples": near_examples,
        "near_candidates_by_dataset_pair": [
            {"datasets": list(pair), "pairs": count}
            for pair, count in sorted(near_dataset_pairs.items())
        ],
        "exact_hash_lineage_components": lineage_components,
    }


def _write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Public Dataset Eligibility Audit",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "No training or model test evaluation was executed.",
        "",
        "| Dataset | Status | Images | Boxes | Classes | Reasons |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["datasets"]:
        lines.append(
            "| {code} | {status} | {images} | {boxes} | {classes} | {reasons} |".format(
                code=row["code"],
                status=row["status"],
                images=row.get("images", 0),
                boxes=row.get("boxes", 0),
                classes=row.get("class_count", 0),
                reasons=", ".join(row.get("reasons", [])) or "-",
            )
        )
    lines.extend(
        [
            "",
            f"Audited datasets: {report['audited_dataset_count']}",
            f"Eligible independent exact-hash lineages: {report['eligible_lineage_count']}",
            f"Cross-dataset exact groups: {report['cross_dataset']['exact_cross_dataset_groups']}",
            f"Cross-dataset perceptual candidates: {report['cross_dataset']['near_cross_dataset_candidates']}",
            "",
            f"Next: **{report['next_action']}**",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_public_dataset_registry(
    registry: str | Path,
    output_root: str | Path,
    *,
    root_overrides: dict[str, str | Path] | None = None,
    archive_overrides: dict[str, str | Path] | None = None,
    near_threshold: int = 4,
    progress: bool = True,
) -> dict:
    registry_path = Path(registry).expanduser().resolve()
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if payload.get("format") != REGISTRY_FORMAT:
        raise ValueError(f"Registry format harus {REGISTRY_FORMAT}")
    specs = payload.get("datasets")
    if not isinstance(specs, list) or not specs:
        raise ValueError("Registry harus memuat list datasets")
    codes = [str(spec.get("code") or "") for spec in specs]
    if any(not code for code in codes) or len(codes) != len(set(codes)):
        raise ValueError("Setiap dataset harus memiliki code unik")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    root_overrides = {key: Path(value) for key, value in (root_overrides or {}).items()}
    archive_overrides = {key: Path(value) for key, value in (archive_overrides or {}).items()}

    dataset_reports, records_by_dataset = [], {}
    minimum_source_images = int(payload.get("minimum_source_images", 100))
    if minimum_source_images < 1:
        raise ValueError("minimum_source_images harus positif")
    for spec in specs:
        code = str(spec["code"])
        report, records = _audit_one(
            spec,
            registry_path.parent,
            root_overrides.get(code),
            archive_overrides.get(code),
            near_threshold,
            minimum_source_images,
            progress,
        )
        dataset_reports.append(report)
        if records:
            records_by_dataset[code] = records
        dataset_path = output_root / "datasets" / f"{code}.json"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    cross = _cross_dataset_analysis(records_by_dataset, near_threshold)
    eligible_codes = {
        row["code"] for row in dataset_reports if row["status"] in ELIGIBLE_AFTER_REBUILD
    }
    eligible_components = [
        sorted(code for code in component if code in eligible_codes)
        for component in cross["exact_hash_lineage_components"]
    ]
    eligible_components = [component for component in eligible_components if component]
    acquired = len(records_by_dataset)
    pending_near = cross["near_cross_dataset_candidates"] > 0
    if acquired < len(specs):
        decision = "INCOMPLETE_ACQUISITION"
        next_action = "acquire_and_sha256_freeze_missing_dataset_versions"
    elif any(row["status"] == "REJECT" for row in dataset_reports):
        decision = "FAIL_DATASET_ELIGIBILITY"
        next_action = "remove_or_resolve_rejected_datasets"
    elif any(row["status"] == "HOLD_METADATA" for row in dataset_reports):
        decision = "HOLD_METADATA_VERIFICATION"
        next_action = "freeze_version_license_and_verified_archive_sha256"
    elif pending_near or any(row["status"] == "REVIEW_NEAR_DUPLICATES" for row in dataset_reports):
        decision = "REVIEW_DUPLICATE_LINEAGE"
        next_action = "review_perceptual_candidates_and_freeze_lineage_clusters"
    elif len(eligible_components) < int(payload.get("minimum_independent_lineages", 3)):
        decision = "FAIL_MINIMUM_INDEPENDENT_LINEAGES"
        next_action = "retain_formal_primary_dataset_variant"
    else:
        decision = "PASS_V2_DATASET_GATE"
        next_action = "rebuild_required_splits_then_freeze_dataset_manifests"

    report = {
        "format": REPORT_FORMAT,
        "registry": str(registry_path),
        "minimum_independent_lineages": int(payload.get("minimum_independent_lineages", 3)),
        "minimum_source_images": minimum_source_images,
        "dataset_count": len(specs),
        "audited_dataset_count": acquired,
        "eligible_dataset_codes": sorted(eligible_codes),
        "eligible_lineage_count": len(eligible_components),
        "eligible_exact_hash_lineages": eligible_components,
        "datasets": dataset_reports,
        "cross_dataset": cross,
        "decision": decision,
        "next_action": next_action,
        "training_authorized": False,
        "training_executed": False,
        "model_test_evaluation_executed": False,
        "near_duplicates_are_review_candidates_not_proof": True,
    }
    summary = output_root / "public_dataset_eligibility_summary.json"
    summary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(cross["near_cross_dataset_examples"], output_root / "cross_dataset_near_candidates.csv")
    _write_csv(cross["exact_cross_dataset_examples"], output_root / "cross_dataset_exact_groups.csv")
    _write_markdown(report, output_root / "public_dataset_eligibility_summary.md")
    return report


def _parse_overrides(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Override harus CODE=PATH: {value}")
        code, raw_path = value.split("=", 1)
        if not code or not raw_path or code in result:
            raise ValueError(f"Override tidak valid/duplikat: {value}")
        result[code] = Path(raw_path).expanduser().resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit eligibility beberapa dataset publik YOLO tanpa training.")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", action="append", default=[], metavar="CODE=PATH")
    parser.add_argument("--archive", action="append", default=[], metavar="CODE=PATH")
    parser.add_argument("--near-threshold", type=int, default=4)
    args = parser.parse_args()
    report = audit_public_dataset_registry(
        args.registry,
        args.output_root,
        root_overrides=_parse_overrides(args.dataset_root),
        archive_overrides=_parse_overrides(args.archive),
        near_threshold=args.near_threshold,
    )
    print("DECISION:", report["decision"])
    for row in report["datasets"]:
        print(
            f"{row['code']}: {row['status']} | images={row.get('images', 0)} "
            f"boxes={row.get('boxes', 0)} reasons={row.get('reasons', [])}"
        )
    print("INDEPENDENT LINEAGES:", report["eligible_lineage_count"])
    print("TRAINING AUTHORIZED:", report["training_authorized"])
    print("SUMMARY:", Path(args.output_root).resolve() / "public_dataset_eligibility_summary.json")


if __name__ == "__main__":
    main()
