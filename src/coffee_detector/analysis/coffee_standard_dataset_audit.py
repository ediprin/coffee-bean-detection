"""Audit the external Coffee Detection with Standard Roboflow export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Only semantically direct mappings are proposed. Ambiguous contaminant classes
# deliberately stay unmapped until a research ontology is frozen.
SNI21_DIRECT_MAPPING = {
    "Biji Berkulit Tanduk": "biji_berkulit_tanduk",
    "Biji Berlubang Lebih dari satu": "biji_berlubang_lebih_satu",
    "Biji Berlubang Satu": "biji_berlubang_satu",
    "Biji Bertutul": "biji_bertutul_tutul",
    "Biji Cokelat": "biji_coklat",
    "Biji Hitam Pecah": "biji_hitam_pecah",
    "Biji Hitam Penuh": "biji_hitam",
    "Biji Hitam Sebagian": "biji_hitam_sebagian",
    "Biji Muda": "biji_muda",
    "Biji Normal": "biji_normal",
    "Biji Pecah": "biji_pecah",
    "Kopi Gelondong": "kopi_gelondong",
    "Kulit Kopi Ukuran Besar": "kulit_kopi_ukuran_besar",
    "Kulit Kopi Ukuran Kecil": "kulit_kopi_ukuran_kecil",
    "Kulit Kopi Ukuran Sedang": "kulit_kopi_ukuran_sedang",
    "Kulit Tanduk Ukuran Besar": "kulit_tanduk_ukuran_besar",
    "Kulit Tanduk Ukuran Kecil": "kulit_tanduk_ukuran_kecil",
    "Kulit Tanduk Ukuran Sedang": "kulit_tanduk_ukuran_sedang",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parent_key(path: Path) -> str:
    # Roboflow exports augmented siblings as <source>_jpg.rf.<hash>.<ext>.
    stem = re.split(r"\.rf\.[0-9a-fA-F]+", path.name, maxsplit=1)[0]
    return re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE).casefold()


def _resolve_root(root: Path) -> Path:
    candidates = [root, *[path.parent for path in root.rglob("data.yaml")]]
    valid = [path for path in candidates if (path / "data.yaml").is_file()]
    if len(set(valid)) != 1:
        raise RuntimeError(f"Dataset root ambigu/tidak ditemukan: {valid}")
    return valid[0]


def _load_names(data_yaml: Path) -> list[str]:
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload.get("names", [])
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda value: int(value))]
    return list(names)


def audit_dataset(dataset_root: str | Path, output: str | Path) -> dict:
    root = _resolve_root(Path(dataset_root).expanduser().resolve())
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    names = _load_names(root / "data.yaml")

    hashes: dict[str, list[dict]] = defaultdict(list)
    parents: dict[str, list[dict]] = defaultdict(list)
    split_rows = {}
    class_by_split: dict[str, Counter] = {}
    invalid_labels = []
    all_box_areas = []

    for split in SPLITS:
        image_dir, label_dir = root / split / "images", root / split / "labels"
        images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        counts = Counter()
        densities, areas = [], []
        missing_labels = 0
        for index, image_path in enumerate(images, start=1):
            record = {"split": split, "file": image_path.name}
            hashes[_sha256(image_path)].append(record)
            parents[_parent_key(image_path)].append(record)
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                missing_labels += 1
                densities.append(0)
                continue
            rows = [row for row in label_path.read_text(errors="replace").splitlines() if row.strip()]
            densities.append(len(rows))
            for line_number, row in enumerate(rows, start=1):
                fields = row.split()
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:5]]
                    if len(fields) != 5 or class_id < 0 or class_id >= len(names):
                        raise ValueError
                    _, _, width, height = values
                    if not all(0 <= value <= 1 for value in values) or width <= 0 or height <= 0:
                        raise ValueError
                except (ValueError, IndexError):
                    invalid_labels.append({"split": split, "file": label_path.name, "line": line_number, "row": row})
                    continue
                counts[class_id] += 1
                areas.append(width * height)
                all_box_areas.append(width * height)
            if index % 250 == 0 or index == len(images):
                print(f"AUDIT {split}: {index}/{len(images)}", flush=True)
        class_by_split[split] = counts
        split_rows[split] = {
            "images": len(images),
            "instances": sum(counts.values()),
            "missing_label_files": missing_labels,
            "density_median": statistics.median(densities) if densities else None,
            "density_q95": sorted(densities)[min(len(densities) - 1, int(0.95 * len(densities)))] if densities else None,
            "box_area_median": statistics.median(areas) if areas else None,
        }

    def cross_split_groups(groups: dict[str, list[dict]]) -> list[dict]:
        result = []
        for key, members in groups.items():
            member_splits = sorted({member["split"] for member in members})
            if len(member_splits) > 1:
                result.append({"key": key, "splits": member_splits, "count": len(members), "members": members})
        return sorted(result, key=lambda item: (-item["count"], item["key"]))

    exact_cross_split = cross_split_groups(hashes)
    parent_cross_split = cross_split_groups(parents)
    class_table = []
    for class_id, class_name in enumerate(names):
        row = {"class_id": class_id, "class_name": class_name}
        for split in SPLITS:
            row[split] = class_by_split[split][class_id]
        row["total"] = sum(row[split] for split in SPLITS)
        row["sni21_direct_mapping"] = SNI21_DIRECT_MAPPING.get(class_name)
        class_table.append(row)

    ambiguous = [name for name in names if name not in SNI21_DIRECT_MAPPING]
    decision = "FAIL_REBUILD_GROUPED_SPLIT" if exact_cross_split or parent_cross_split else "PASS_SPLIT_IDENTITY_GATE"
    summary = {
        "status": "audit_complete",
        "decision": decision,
        "training_authorized": False,
        "dataset_root": str(root),
        "class_count": len(names),
        "splits": split_rows,
        "total_images": sum(row["images"] for row in split_rows.values()),
        "total_instances": sum(row["instances"] for row in split_rows.values()),
        "exact_cross_split_groups": len(exact_cross_split),
        "roboflow_parent_cross_split_groups": len(parent_cross_split),
        "invalid_label_rows": len(invalid_labels),
        "global_box_area_median": statistics.median(all_box_areas) if all_box_areas else None,
        "direct_sni21_mapping_count": len(SNI21_DIRECT_MAPPING),
        "ambiguous_or_external_classes": ambiguous,
        "class_distribution": class_table,
        "exact_cross_split_examples": exact_cross_split[:25],
        "parent_cross_split_examples": parent_cross_split[:25],
        "invalid_label_examples": invalid_labels[:25],
        "next_action": "rebuild_grouped_development_split_before_training" if decision.startswith("FAIL") else "perform_visual_and_ontology_audit",
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_dataset(args.dataset_root, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
