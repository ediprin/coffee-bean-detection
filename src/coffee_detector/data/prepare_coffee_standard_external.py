"""Prepare an identity-independent SNI-21 external benchmark from Roboflow v8."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


SOURCE_SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SNI21_NAMES = [
    "biji_berkulit_tanduk", "biji_berlubang_lebih_satu", "biji_berlubang_satu",
    "biji_bertutul_tutul", "biji_coklat", "biji_hitam", "biji_hitam_pecah",
    "biji_hitam_sebagian", "biji_muda", "biji_normal", "biji_pecah",
    "kopi_gelondong", "kulit_kopi_ukuran_besar", "kulit_kopi_ukuran_kecil",
    "kulit_kopi_ukuran_sedang", "kulit_tanduk_ukuran_besar",
    "kulit_tanduk_ukuran_kecil", "kulit_tanduk_ukuran_sedang",
    "tanah_batu_ranting_besar", "tanah_batu_ranting_kecil", "tanah_batu_ranting_sedang",
]
SOURCE_TO_SNI21 = {
    2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 6, 8: 5, 9: 7, 10: 8,
    11: 9, 12: 10, 15: 11, 16: 12, 17: 13, 18: 14, 19: 15, 20: 16, 21: 17,
}


def _parent_key(path: Path) -> str:
    stem = re.split(r"\.rf\.[0-9a-fA-F]+", path.name, maxsplit=1)[0]
    return re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE).casefold()


def _dataset_root(root: Path) -> Path:
    matches = sorted(set(path.parent for path in root.rglob("data.yaml")))
    if len(matches) != 1:
        raise RuntimeError(f"Dataset root ambigu/tidak ditemukan: {matches}")
    return matches[0]


def _read_rows(label: Path) -> list[str]:
    return [row for row in label.read_text(errors="replace").splitlines() if row.strip()]


def _mapped_rows(label: Path) -> list[str]:
    output = []
    for row in _read_rows(label):
        fields = row.split()
        source_id = int(fields[0])
        if source_id in SOURCE_TO_SNI21:
            output.append(" ".join([str(SOURCE_TO_SNI21[source_id]), *fields[1:]]))
    return output


def _representative(records: list[dict]) -> dict:
    # Prefer the sibling retaining the most shared-SNI objects; ties are stable.
    return min(records, key=lambda record: (-record["mapped_instances"], record["image"].name))


def prepare_external(source_root: str | Path, output_root: str | Path) -> dict:
    source = _dataset_root(Path(source_root).expanduser().resolve())
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    image_out, label_out = output / "external" / "images", output / "external" / "labels"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    groups = defaultdict(list)
    for split in SOURCE_SPLITS:
        for image in sorted((source / split / "images").glob("*")):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = source / split / "labels" / f"{image.stem}.txt"
            mapped = _mapped_rows(label)
            groups[_parent_key(image)].append({
                "image": image, "label": label, "source_split": split,
                "mapped_rows": mapped, "mapped_instances": len(mapped),
            })
    manifest, class_counts = [], Counter()
    skipped_empty = []
    for parent_id, records in sorted(groups.items()):
        selected = _representative(records)
        if not selected["mapped_rows"]:
            skipped_empty.append(parent_id)
            continue
        suffix = selected["image"].suffix.lower()
        name = f"{parent_id}{suffix}"
        shutil.copy2(selected["image"], image_out / name)
        (label_out / f"{Path(name).stem}.txt").write_text("\n".join(selected["mapped_rows"]) + "\n", encoding="utf-8")
        for row in selected["mapped_rows"]:
            class_counts[int(row.split()[0])] += 1
        manifest.append({
            "parent_id": parent_id, "source_image": selected["image"].name,
            "source_split": selected["source_split"], "available_siblings": len(records),
            "mapped_instances": selected["mapped_instances"],
        })
    yaml_payload = {
        "path": str(output), "train": "external/images", "val": "external/images",
        "names": SNI21_NAMES, "nc": len(SNI21_NAMES),
    }
    (output / "data.yaml").write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")
    (output / "external_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing = [SNI21_NAMES[index] for index in range(len(SNI21_NAMES)) if class_counts[index] == 0]
    summary = {
        "status": "complete", "role": "external_posthoc_diagnostic_only",
        "training_authorized": False, "test_claim_authorized": False,
        "source_images": sum(len(records) for records in groups.values()),
        "source_parent_identities": len(groups), "selected_parent_identities": len(manifest),
        "skipped_parents_without_directly_mapped_boxes": len(skipped_empty),
        "images": len(manifest), "instances": sum(class_counts.values()),
        "class_distribution": [
            {"class_id": index, "class_name": name, "instances": class_counts[index]}
            for index, name in enumerate(SNI21_NAMES)
        ],
        "classes_without_ground_truth": missing,
        "direct_mapping_source_class_ids": SOURCE_TO_SNI21,
        "limitations": [
            "One deterministic representative is selected per Roboflow parent identity.",
            "Seven non-equivalent source classes are excluded rather than force-mapped.",
            "Results measure cross-dataset transfer, not in-domain detector quality.",
        ],
    }
    (output / "external_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_external(args.source_root, args.output_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
