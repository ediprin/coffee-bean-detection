from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

from .dataset import UnionFind, image_sha256, write_json


SNI21_CLASSES = (
    "biji_berkulit_tanduk",
    "biji_berlubang_lebih_satu",
    "biji_berlubang_satu",
    "biji_bertutul_tutul",
    "biji_coklat",
    "biji_hitam",
    "biji_hitam_pecah",
    "biji_hitam_sebagian",
    "biji_muda",
    "biji_normal",
    "biji_pecah",
    "kopi_gelondong",
    "kulit_kopi_ukuran_besar",
    "kulit_kopi_ukuran_kecil",
    "kulit_kopi_ukuran_sedang",
    "kulit_tanduk_ukuran_besar",
    "kulit_tanduk_ukuran_kecil",
    "kulit_tanduk_ukuran_sedang",
    "tanah_batu_ranting_besar",
    "tanah_batu_ranting_kecil",
    "tanah_batu_ranting_sedang",
)
SNI21_IDS = {name: index for index, name in enumerate(SNI21_CLASSES)}

ADRIAN_CLASS_MAP = {
    "Batu berukuran besar": "tanah_batu_ranting_besar",
    "Batu berukuran kecil": "tanah_batu_ranting_kecil",
    "Batu berukuran sedang": "tanah_batu_ranting_sedang",
    "Biji berkulit tanduk": "biji_berkulit_tanduk",
    "Biji berlubang lebih dari satu": "biji_berlubang_lebih_satu",
    "Biji berlubang satu": "biji_berlubang_satu",
    "Biji bertutul-tutul": "biji_bertutul_tutul",
    "Biji cokelat": "biji_coklat",
    "Biji hitam": "biji_hitam",
    "Biji hitam pecah": "biji_hitam_pecah",
    "Biji hitam sebagian": "biji_hitam_sebagian",
    "Biji muda": "biji_muda",
    "Biji pecah": "biji_pecah",
    "Biji tanpa cacat": "biji_normal",
    "Kopi gelondong": "kopi_gelondong",
    "Kulit kopi ukuran besar": "kulit_kopi_ukuran_besar",
    "Kulit kopi ukuran kecil": "kulit_kopi_ukuran_kecil",
    "Kulit kopi ukuran sedang": "kulit_kopi_ukuran_sedang",
    "Kulit tanduk ukuran besar": "kulit_tanduk_ukuran_besar",
    "Kulit tanduk ukuran kecil": "kulit_tanduk_ukuran_kecil",
    "Kulit tanduk ukuran sedang": "kulit_tanduk_ukuran_sedang",
    "Ranting berukuran besar": "tanah_batu_ranting_besar",
    "Ranting berukuran kecil": "tanah_batu_ranting_kecil",
    "Tanah berukuran besar": "tanah_batu_ranting_besar",
    "Tanah berukuran kecil": "tanah_batu_ranting_kecil",
    "Tanah berukuran sedang": "tanah_batu_ranting_sedang",
    "ranting berukuran sedang": "tanah_batu_ranting_sedang",
}

FARUQ_CLASS_MAP = {
    "biji_berlubang_lebih_satu": "biji_berlubang_lebih_satu",
    "biji_berlubang_satu": "biji_berlubang_satu",
    "biji_bertutul_tutul": "biji_bertutul_tutul",
    "biji_coklat": "biji_coklat",
    "biji_gelondong": "kopi_gelondong",
    "biji_hitam": "biji_hitam",
    "biji_hitam_pecah": "biji_hitam_pecah",
    "biji_hitam_sebagian": "biji_hitam_sebagian",
    "biji_kulit_tanduk": "biji_berkulit_tanduk",
    "biji_muda": "biji_muda",
    "biji_normal": "biji_normal",
    "biji_pecah": "biji_pecah",
    "kulit_kopi_ukuran_besar": "kulit_kopi_ukuran_besar",
    "kulit_kopi_ukuran_kecil": "kulit_kopi_ukuran_kecil",
    "kulit_kopi_ukuran_sedang": "kulit_kopi_ukuran_sedang",
    "kulit_tanduk_ukuran_besar": "kulit_tanduk_ukuran_besar",
    "kulit_tanduk_ukuran_kecil": "kulit_tanduk_ukuran_kecil",
    "kulit_tanduk_ukuran_sedang": "kulit_tanduk_ukuran_sedang",
    "tanah_batu_ranting_besar": "tanah_batu_ranting_besar",
    "tanah_batu_ranting_kecil": "tanah_batu_ranting_kecil",
    "tanah_batu_ranting_sedang": "tanah_batu_ranting_sedang",
}

SOURCE_CLASS_MAPS = {
    "adrian_detection": ADRIAN_CLASS_MAP,
    "faruq_segmentation": FARUQ_CLASS_MAP,
}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Authority:
    split: str
    group_id: str
    kept_annotations: frozenset[str]
    annotation_classes: dict[str, str]


@dataclass
class SourceRecord:
    dataset: str
    archive_split: str
    image_id: str
    image_path: Path
    width: int
    height: int
    parent_id: str
    sha256: str
    annotations: list[dict]
    requested_split: str | None
    authority_group_id: str | None
    orientation_action: str

    @property
    def annotation_signature(self) -> tuple:
        return tuple(
            sorted(
                (
                    item["class_id"],
                    round(item["x"], 4),
                    round(item["y"], 4),
                    round(item["width"], 4),
                    round(item["height"], 4),
                )
                for item in self.annotations
            )
        )


def canonical_source_identity(file_name: str) -> str:
    stem = Path(file_name).stem
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def canonical_archive_split(value: str) -> str:
    key = str(value).strip().lower()
    if key not in SPLIT_ALIASES:
        raise ValueError(f"Split sumber tidak dikenal: {value}")
    return SPLIT_ALIASES[key]


def _stable_split(key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "val"
    return "test"


def _load_authorities(
    manifest_path: Path,
) -> tuple[
    dict[tuple[str, str, str], Authority],
    dict[tuple[str, str], tuple[str, str]],
    dict,
]:
    required = {
        "dataset",
        "archive_split",
        "generated_split",
        "group_id",
        "source_identity",
        "image_id",
        "annotation_id",
        "canonical_class",
    }
    rows: dict[tuple[str, str, str], dict] = {}
    parents: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                "Manifest crop belum lengkap: " + ", ".join(missing)
            )
        for row in reader:
            dataset = str(row["dataset"]).strip()
            archive_split = canonical_archive_split(row["archive_split"])
            generated_split = canonical_archive_split(row["generated_split"])
            image_id = str(row["image_id"]).strip()
            annotation_id = str(row["annotation_id"]).strip()
            group_id = str(row["group_id"]).strip()
            source_identity = str(row["source_identity"]).strip()
            class_name = str(row["canonical_class"]).strip()
            if dataset not in SOURCE_CLASS_MAPS:
                raise ValueError(f"Dataset manifest tidak dikenal: {dataset}")
            if class_name not in SNI21_IDS:
                raise ValueError(f"Kelas kanonis manifest tidak dikenal: {class_name}")
            key = (dataset, archive_split, image_id)
            item = rows.setdefault(
                key,
                {
                    "splits": set(),
                    "groups": set(),
                    "annotations": set(),
                    "classes": {},
                },
            )
            item["splits"].add(generated_split)
            item["groups"].add(group_id)
            item["annotations"].add(annotation_id)
            previous = item["classes"].get(annotation_id)
            if previous is not None and previous != class_name:
                raise ValueError(
                    f"Annotation {key}/{annotation_id} memiliki mapping bertentangan"
                )
            item["classes"][annotation_id] = class_name
            parents[(dataset, source_identity)].add((generated_split, group_id))

    authorities = {}
    for key, item in rows.items():
        if len(item["splits"]) != 1 or len(item["groups"]) != 1:
            raise ValueError(f"Authority split/group bertentangan: {key}")
        authorities[key] = Authority(
            split=next(iter(item["splits"])),
            group_id=next(iter(item["groups"])),
            kept_annotations=frozenset(item["annotations"]),
            annotation_classes=dict(item["classes"]),
        )

    parent_authorities = {}
    conflicting_parents = 0
    for key, values in parents.items():
        if len(values) == 1:
            parent_authorities[key] = next(iter(values))
        else:
            conflicting_parents += 1
    return authorities, parent_authorities, {
        "rows": sum(len(item.kept_annotations) for item in authorities.values()),
        "images": len(authorities),
        "parents_with_single_authority": len(parent_authorities),
        "parents_with_conflicting_authority": conflicting_parents,
    }


def _find_annotation_files(root: Path) -> list[Path]:
    paths = sorted(root.rglob("_annotations.coco.json"))
    if len(paths) != 3:
        raise FileNotFoundError(
            f"Diharapkan tiga file COCO train/valid/test di {root}; ditemukan {len(paths)}"
        )
    return paths


def _clamp_bbox(
    bbox: list[float], width: int, height: int
) -> tuple[dict | None, bool]:
    if len(bbox) != 4:
        return None, False
    x, y, box_width, box_height = (float(value) for value in bbox)
    if not all(
        value == value and abs(value) != float("inf")
        for value in (x, y, box_width, box_height)
    ):
        return None, False
    left = max(0.0, min(float(width), x))
    top = max(0.0, min(float(height), y))
    right = max(0.0, min(float(width), x + box_width))
    bottom = max(0.0, min(float(height), y + box_height))
    if right <= left or bottom <= top:
        return None, False
    clamped = (
        abs(left - x) > 1e-9
        or abs(top - y) > 1e-9
        or abs(right - (x + box_width)) > 1e-9
        or abs(bottom - (y + box_height)) > 1e-9
    )
    return {
        "x": left,
        "y": top,
        "width": right - left,
        "height": bottom - top,
    }, clamped


def _read_sources(
    source_roots: dict[str, Path],
    authorities: dict[tuple[str, str, str], Authority],
    parent_authorities: dict[tuple[str, str], tuple[str, str]],
) -> tuple[list[SourceRecord], list[dict], dict]:
    records = []
    quarantine = []
    counters = Counter()
    for dataset, root in source_roots.items():
        class_map = SOURCE_CLASS_MAPS[dataset]
        for annotation_path in _find_annotation_files(root):
            archive_split = canonical_archive_split(annotation_path.parent.name)
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            categories = {
                int(item["id"]): str(item["name"])
                for item in payload.get("categories", [])
            }
            annotations_by_image: dict[int | str, list[dict]] = defaultdict(list)
            for annotation in payload.get("annotations", []):
                annotations_by_image[annotation["image_id"]].append(annotation)
            for image_index, image in enumerate(payload.get("images", []), 1):
                image_id = str(image["id"])
                image_path = annotation_path.parent / str(image["file_name"])
                source_annotations = annotations_by_image.get(image["id"], [])
                key = (dataset, archive_split, image_id)
                authority = authorities.get(key)
                parent_id = canonical_source_identity(str(image["file_name"]))
                parent_authority = parent_authorities.get((dataset, parent_id))
                reasons = []
                if not image_path.is_file():
                    reasons.append("missing_image")
                    raw_width, raw_height = 0, 0
                else:
                    with Image.open(image_path) as source_image:
                        raw_width, raw_height = source_image.size
                width, height = int(image["width"]), int(image["height"])
                if width <= 0 or height <= 0:
                    reasons.append("invalid_declared_dimensions")
                if (raw_width, raw_height) == (width, height):
                    orientation_action = "none"
                elif (raw_width, raw_height) == (height, width):
                    orientation_action = "rotate_clockwise"
                    counters["rotated_clockwise"] += 1
                else:
                    orientation_action = "invalid"
                    reasons.append("image_dimensions_do_not_match_coco")

                source_ids = {str(item["id"]) for item in source_annotations}
                if source_annotations and authority is None:
                    reasons.append("all_annotations_excluded_by_crop_audit")
                if authority is not None:
                    omitted = sorted(source_ids - authority.kept_annotations)
                    unknown = sorted(authority.kept_annotations - source_ids)
                    if omitted:
                        reasons.append("annotation_excluded_by_crop_audit")
                    if unknown:
                        reasons.append("manifest_annotation_missing_from_coco")

                converted = []
                for annotation in source_annotations:
                    source_name = categories.get(int(annotation["category_id"]))
                    canonical_name = class_map.get(str(source_name))
                    if canonical_name is None:
                        reasons.append(f"unmapped_category:{source_name}")
                        continue
                    annotation_id = str(annotation["id"])
                    if (
                        authority is not None
                        and annotation_id in authority.annotation_classes
                        and authority.annotation_classes[annotation_id]
                        != canonical_name
                    ):
                        reasons.append("manifest_class_mapping_disagrees")
                    box, clamped = _clamp_bbox(
                        list(annotation.get("bbox", [])), width, height
                    )
                    if box is None:
                        reasons.append("invalid_bbox")
                        continue
                    counters["clamped_boxes"] += int(clamped)
                    converted.append(
                        {
                            "annotation_id": annotation_id,
                            "class_name": canonical_name,
                            "class_id": SNI21_IDS[canonical_name],
                            **box,
                            "has_polygon": bool(annotation.get("segmentation")),
                        }
                    )

                if reasons:
                    quarantine.append(
                        {
                            "dataset": dataset,
                            "archive_split": archive_split,
                            "image_id": image_id,
                            "image": str(image_path),
                            "reasons": sorted(set(reasons)),
                            "annotations": len(source_annotations),
                        }
                    )
                    counters["quarantined_images"] += 1
                    counters["quarantined_annotations"] += len(source_annotations)
                    continue

                requested_split = None
                authority_group_id = None
                if authority is not None:
                    requested_split = authority.split
                    authority_group_id = authority.group_id
                elif parent_authority is not None:
                    requested_split, authority_group_id = parent_authority
                records.append(
                    SourceRecord(
                        dataset=dataset,
                        archive_split=archive_split,
                        image_id=image_id,
                        image_path=image_path,
                        width=width,
                        height=height,
                        parent_id=parent_id,
                        sha256=image_sha256(image_path),
                        annotations=converted,
                        requested_split=requested_split,
                        authority_group_id=authority_group_id,
                        orientation_action=orientation_action,
                    )
                )
                counters["accepted_images"] += 1
                counters["accepted_annotations"] += len(converted)
                if image_index % 1000 == 0:
                    print(
                        f"  index {dataset}/{archive_split}: "
                        f"{image_index}/{len(payload.get('images', []))}",
                        flush=True,
                    )
    return records, quarantine, dict(counters)


def _resolve_exact_duplicates(
    records: list[SourceRecord],
) -> tuple[list[SourceRecord], list[dict], list[dict]]:
    by_hash: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    accepted = []
    dropped = []
    quarantine = []
    for digest, items in sorted(by_hash.items()):
        signatures = {item.annotation_signature for item in items}
        requested_splits = {
            item.requested_split for item in items if item.requested_split is not None
        }
        if len(signatures) > 1 or len(requested_splits) > 1:
            reason = (
                "exact_image_annotation_conflict"
                if len(signatures) > 1
                else "exact_image_split_authority_conflict"
            )
            for item in items:
                quarantine.append(
                    {
                        "dataset": item.dataset,
                        "archive_split": item.archive_split,
                        "image_id": item.image_id,
                        "image": str(item.image_path),
                        "reasons": [reason],
                        "annotations": len(item.annotations),
                        "sha256": digest,
                    }
                )
            continue
        items.sort(
            key=lambda item: (
                item.requested_split is None,
                item.dataset,
                item.archive_split,
                item.image_id,
                str(item.image_path),
            )
        )
        accepted.append(items[0])
        for item in items[1:]:
            dropped.append(
                {
                    "source_image": str(item.image_path),
                    "kept_image": str(items[0].image_path),
                    "sha256": digest,
                    "reason": "same_image_same_annotation",
                }
            )
    return accepted, dropped, quarantine


def _assign_groups(
    records: list[SourceRecord], seed: int
) -> tuple[dict[int, str], dict[int, str], dict]:
    union_find = UnionFind(len(records))
    indices_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        indices_by_key[("parent", f"{item.dataset}:{item.parent_id}")].append(index)
        indices_by_key[("sha256", item.sha256)].append(index)
        if item.authority_group_id:
            indices_by_key[("authority", item.authority_group_id)].append(index)
    for indices in indices_by_key.values():
        for index in indices[1:]:
            union_find.union(indices[0], index)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[union_find.find(index)].append(index)

    split_by_index = {}
    group_by_index = {}
    split_counts = Counter()
    group_rows = []
    for group_number, indices in enumerate(
        sorted(components.values(), key=lambda values: min(values))
    ):
        requested = {
            records[index].requested_split
            for index in indices
            if records[index].requested_split is not None
        }
        if len(requested) > 1:
            examples = [str(records[index].image_path) for index in indices[:10]]
            raise RuntimeError(
                "Satu identity group memiliki authority split berbeda: "
                + ", ".join(sorted(requested))
                + f"; contoh={examples}"
            )
        stable_key = "|".join(
            sorted(
                {
                    records[index].authority_group_id
                    or f"{records[index].dataset}:{records[index].parent_id}"
                    for index in indices
                }
            )
        )
        split = next(iter(requested)) if requested else _stable_split(stable_key, seed)
        group_id = f"sni21-group-{group_number:06d}"
        for index in indices:
            split_by_index[index] = split
            group_by_index[index] = group_id
        split_counts[split] += len(indices)
        group_rows.append(
            {
                "group_id": group_id,
                "split": split,
                "images": len(indices),
                "authority_locked": bool(requested),
                "datasets": sorted({records[index].dataset for index in indices}),
                "parents": sorted(
                    {
                        f"{records[index].dataset}:{records[index].parent_id}"
                        for index in indices
                    }
                ),
            }
        )
    return split_by_index, group_by_index, {
        "groups": len(components),
        "images_by_split_before_write": dict(split_counts),
        "group_manifest": group_rows,
    }


def _write_image(
    record: SourceRecord, target: Path, link_mode: str
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if record.orientation_action == "rotate_clockwise":
        with Image.open(record.image_path) as image:
            rotated = image.convert("RGB").transpose(Image.Transpose.ROTATE_270)
            rotated.save(target, quality=95, subsampling=0)
        return
    if link_mode == "hardlink":
        try:
            os.link(record.image_path, target)
            return
        except OSError:
            pass
    shutil.copy2(record.image_path, target)


def _write_yolo_label(record: SourceRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in sorted(
        record.annotations,
        key=lambda annotation: (
            annotation["class_id"],
            annotation["annotation_id"],
        ),
    ):
        x_center = (item["x"] + item["width"] / 2.0) / record.width
        y_center = (item["y"] + item["height"] / 2.0) / record.height
        width = item["width"] / record.width
        height = item["height"] / record.height
        lines.append(
            f"{item['class_id']} {x_center:.8f} {y_center:.8f} "
            f"{width:.8f} {height:.8f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prepare_sni_fullscene(
    adrian_root: str | Path,
    faruq_root: str | Path,
    crop_manifest: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    link_mode: str = "copy",
) -> dict:
    if link_mode not in {"copy", "hardlink"}:
        raise ValueError("link_mode harus 'copy' atau 'hardlink'")
    source_roots = {
        "adrian_detection": Path(adrian_root).expanduser().resolve(),
        "faruq_segmentation": Path(faruq_root).expanduser().resolve(),
    }
    for dataset, root in source_roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"Root {dataset} tidak ditemukan: {root}")
    crop_manifest = Path(crop_manifest).expanduser().resolve()
    if not crop_manifest.is_file():
        raise FileNotFoundError(f"Manifest crop tidak ditemukan: {crop_manifest}")
    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output tidak kosong: {output_root}")

    print("[1/5] Membaca authority split dari crop manifest...", flush=True)
    authorities, parent_authorities, authority_audit = _load_authorities(
        crop_manifest
    )
    print("[2/5] Membaca COCO, mapping SNI-21, dan audit geometri...", flush=True)
    records, quarantine, source_audit = _read_sources(
        source_roots, authorities, parent_authorities
    )
    print("[3/5] Menyelesaikan exact duplicate dan konflik anotasi...", flush=True)
    records, dropped_duplicates, exact_quarantine = _resolve_exact_duplicates(
        records
    )
    quarantine.extend(exact_quarantine)
    print("[4/5] Menetapkan grouped split yang konsisten dengan crop manifest...", flush=True)
    split_by_index, group_by_index, grouping_audit = _assign_groups(records, seed)

    output_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    split_images = Counter()
    split_annotations = Counter()
    split_domains: dict[str, Counter] = defaultdict(Counter)
    split_classes: dict[str, Counter] = defaultdict(Counter)
    used_targets = set()
    print("[5/5] Materialisasi dataset YOLO SNI-21...", flush=True)
    for index, record in enumerate(records, 1):
        split = split_by_index[index - 1]
        output_name = f"{record.dataset}__{record.image_path.name}"
        target_key = (split, output_name.lower())
        if target_key in used_targets:
            raise RuntimeError(
                f"Nama output bertabrakan pada split {split}: {output_name}"
            )
        used_targets.add(target_key)
        image_target = output_root / split / "images" / output_name
        label_target = (
            output_root / split / "labels" / Path(output_name).with_suffix(".txt")
        )
        _write_image(record, image_target, link_mode)
        _write_yolo_label(record, label_target)
        split_images[split] += 1
        split_annotations[split] += len(record.annotations)
        split_domains[split][record.dataset] += 1
        split_classes[split].update(
            item["class_name"] for item in record.annotations
        )
        manifests.append(
            {
                "dataset": record.dataset,
                "archive_split": record.archive_split,
                "source_image_id": record.image_id,
                "source_image": str(record.image_path),
                "source_parent_id": record.parent_id,
                "source_sha256": record.sha256,
                "output_split": split,
                "output_image": str(image_target),
                "output_label": str(label_target),
                "group_id": group_by_index[index - 1],
                "authority_group_id": record.authority_group_id,
                "orientation_action": record.orientation_action,
                "annotations": len(record.annotations),
            }
        )
        if index % 500 == 0 or index == len(records):
            print(f"  materialize {index}/{len(records)} gambar", flush=True)

    names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    data_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": names,
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    missing_classes = {
        split: sorted(set(SNI21_CLASSES) - set(split_classes[split]))
        for split in ("train", "val", "test")
    }
    quarantine_reasons = Counter(
        reason for item in quarantine for reason in item["reasons"]
    )
    group_split_lookup = {}
    cross_split_groups = []
    for item in manifests:
        previous = group_split_lookup.setdefault(
            item["group_id"], item["output_split"]
        )
        if previous != item["output_split"]:
            cross_split_groups.append(item["group_id"])
    result = {
        "format": "coffee_detector.sni21_fullscene.v1",
        "training_executed": False,
        "seed": seed,
        "source_roots": {
            dataset: str(root) for dataset, root in source_roots.items()
        },
        "crop_manifest": str(crop_manifest),
        "output_root": str(output_root),
        "classes": list(SNI21_CLASSES),
        "images_by_split": dict(split_images),
        "annotations_by_split": dict(split_annotations),
        "images_by_split_and_dataset": {
            split: dict(split_domains[split])
            for split in ("train", "val", "test")
        },
        "annotations_by_split_and_class": {
            split: dict(sorted(split_classes[split].items()))
            for split in ("train", "val", "test")
        },
        "missing_classes_by_split": missing_classes,
        "authority_audit": authority_audit,
        "source_audit": source_audit,
        "grouping": {
            "groups": grouping_audit["groups"],
            "cross_split_groups": len(set(cross_split_groups)),
            "authority": (
                "generated_split dari coffee-sni-instance-crop-v1; "
                "parent dan exact hash tidak boleh menyeberang split"
            ),
        },
        "dropped_exact_duplicate_images": len(dropped_duplicates),
        "quarantined_images": len(quarantine),
        "quarantined_annotations": sum(
            int(item.get("annotations", 0)) for item in quarantine
        ),
        "quarantine_reasons": dict(quarantine_reasons),
        "rotated_clockwise_images": source_audit.get("rotated_clockwise", 0),
        "clamped_boxes": source_audit.get("clamped_boxes", 0),
        "test_locked": True,
        "training_ready": (
            not any(missing_classes.values())
            and not cross_split_groups
            and all(split_images[split] > 0 for split in ("train", "val", "test"))
        ),
        "claim_note": (
            "Dataset real sparse/multiobject terkontrol; bukan real dense 300 g. "
            "VA-DCP hanya boleh menambah train."
        ),
    }
    write_json(result, output_root / "audit.json")
    write_json(manifests, output_root / "split_manifest.json")
    write_json(quarantine, output_root / "quarantine.json")
    write_json(dropped_duplicates, output_root / "dropped_duplicates.json")
    write_json(
        grouping_audit["group_manifest"], output_root / "group_manifest.json"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialisasi dua ekspor COCO SNI menjadi grouped YOLO SNI-21 "
            "tanpa menjalankan training."
        )
    )
    parser.add_argument("--adrian-root", required=True)
    parser.add_argument("--faruq-root", required=True)
    parser.add_argument("--crop-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Hardlink menghemat ruang bila source dan output berada pada volume sama.",
    )
    args = parser.parse_args()
    result = prepare_sni_fullscene(
        args.adrian_root,
        args.faruq_root,
        args.crop_manifest,
        args.output_root,
        seed=args.seed,
        link_mode=args.link_mode,
    )
    print("=== MATERIALISASI SNI-21 SELESAI ===")
    print(f"Output         : {result['output_root']}")
    print(f"Images         : {result['images_by_split']}")
    print(f"Annotations    : {result['annotations_by_split']}")
    print(f"Quarantine     : {result['quarantined_images']} gambar")
    print(f"Rotate CW      : {result['rotated_clockwise_images']} gambar")
    print(f"Training ready : {result['training_ready']}")
    print("TRAINING BELUM DIJALANKAN.")


if __name__ == "__main__":
    main()
