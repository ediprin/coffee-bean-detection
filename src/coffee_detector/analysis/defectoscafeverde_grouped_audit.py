"""Fail-fast audit for the rebuilt DefectosCafeVerde grouped dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from coffee_detector.data.prepare_defectoscafeverde_grouped import NAMES, infer_physical_group


def _parse_label(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: label YOLO harus memiliki 5 kolom")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if class_id not in range(len(NAMES)):
            raise ValueError(f"{path}:{line_number}: class_id tidak valid")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{path}:{line_number}: koordinat di luar [0,1]")
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"{path}:{line_number}: ukuran box harus positif")
        counts[class_id] += 1
    if not counts:
        raise ValueError(f"{path}: label kosong")
    return counts


def audit_grouped_dataset(dataset_root: str | Path, output: str | Path) -> dict:
    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / "grouped_manifest.json"
    summary_path = root / "grouped_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Manifest atau summary grouped tidak ditemukan")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    ids, names = set(), set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    sha_splits: dict[str, set[str]] = defaultdict(set)
    split_images: Counter[str] = Counter()
    split_instances = {split: Counter() for split in ("train", "val", "test")}
    group_sizes: Counter[str] = Counter()
    for index, row in enumerate(rows):
        try:
            image_id = str(row["image_id"])
            source_name = str(row["source_name"])
            split = str(row["output_split"])
            group_id = str(row["physical_group_id"])
            if split not in split_instances:
                raise ValueError(f"split tidak valid: {split}")
            if group_id != infer_physical_group(source_name):
                raise ValueError("physical_group_id tidak cocok dengan nama sumber")
            if image_id in ids or source_name in names:
                raise ValueError("ID atau nama sumber berulang")
            ids.add(image_id)
            names.add(source_name)
            image = root / row["image"]
            label = root / row["label"]
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError("citra atau label hilang")
            sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
            if sha256 != row["sha256"]:
                raise ValueError("SHA citra berubah")
            with Image.open(image) as decoded:
                decoded.verify()
            counts = _parse_label(label)
            split_instances[split].update(counts)
            split_images[split] += 1
            group_splits[group_id].add(split)
            group_sizes[group_id] += 1
            sha_splits[sha256].add(split)
        except Exception as error:  # collect a bounded diagnostic instead of hiding later failures
            if len(errors) < 100:
                errors.append(f"row {index}: {error}")

    cross_group = sorted(group_id for group_id, splits in group_splits.items() if len(splits) > 1)
    cross_exact = sorted(sha256 for sha256, splits in sha_splits.items() if len(splits) > 1)
    missing = {
        split: [NAMES[class_id] for class_id in range(len(NAMES)) if split_instances[split][class_id] == 0]
        for split in split_instances
    }
    gates = {
        "exactly_4038_original_images": len(rows) == 4038 and len(ids) == 4038,
        "exactly_2069_inferred_physical_groups": len(group_splits) == 2069,
        "exactly_1969_two_view_groups": sum(size == 2 for size in group_sizes.values()) == 1969,
        "exactly_100_single_view_groups": sum(size == 1 for size in group_sizes.values()) == 100,
        "zero_invalid_files_or_labels": not errors,
        "zero_cross_split_physical_groups": not cross_group,
        "zero_exact_cross_split_duplicates": not cross_exact,
        "all_12_classes_present_each_split": all(not values for values in missing.values()),
        "no_augmentation_in_rebuild": build_summary.get("augmentation_applied") is False,
        "test_not_accessed_for_model_selection": build_summary.get("test_accessed") is False,
    }
    decision = "PASS_GROUPED_DATASET_GATE" if all(gates.values()) else "FAIL_GROUPED_DATASET_GATE"
    report = {
        "format": "coffee_detector.defectoscafeverde.grouped_audit.v1",
        "decision": decision,
        "gates": gates,
        "images_by_split": dict(split_images),
        "groups_by_split": {
            split: sum(splits == {split} for splits in group_splits.values())
            for split in split_instances
        },
        "class_instances_by_split": {
            split: {NAMES[class_id]: counts[class_id] for class_id in range(len(NAMES))}
            for split, counts in split_instances.items()
        },
        "missing_classes": missing,
        "cross_split_physical_groups": len(cross_group),
        "exact_cross_split_duplicates": len(cross_exact),
        "errors": errors,
        "identity_limitation": (
            "Physical identity is inferred from the paper's dual-sided acquisition and "
            "the even/odd filename sequence; public metadata has no explicit bean ID."
        ),
        "training_authorized": False,
        "test_accessed": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_grouped_dataset(args.dataset_root, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
