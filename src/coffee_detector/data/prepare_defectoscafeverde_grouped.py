"""Download and rebuild DefectosCafeVerde with physical-bean grouped splits.

The public Roboflow versions split individual source images.  The acquisition
described by the accompanying paper records both sides of a physical bean and
the source filenames form consecutive even/odd pairs (for example A0/A1).
This module keeps such pairs together before assigning train/validation/test.

No remote project state is changed.  The API is used read-only to retrieve the
original images and their source annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image


WORKSPACE = "redtraining"
PROJECT = "defectoscafeverde"
API_BASE = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}"
SOURCE_URL = "https://universe.roboflow.com/redtraining/defectoscafeverde"
LICENSE = "CC BY 4.0"
PAPER_URL = "https://www.mdpi.com/2077-0472/16/16/1796"
NAMES = [
    "agrio",
    "broca",
    "caracolillo",
    "concha",
    "elefante",
    "helado",
    "negro",
    "normal",
    "oreja",
    "partido",
    "seca",
    "triangulo",
]
NAME_TO_ID = {name: index for index, name in enumerate(NAMES)}
SOURCE_NAME = re.compile(r"^(?P<prefix>[A-Za-z]+)(?P<number>\d+)\.[^.]+$")
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


@dataclass(frozen=True)
class SourceRecord:
    image_id: str
    source_name: str
    source_split: str
    group_id: str
    image_path: Path
    label_path: Path
    class_counts: Counter[int]
    sha256: str


def infer_physical_group(source_name: str) -> str:
    """Infer the two-sided physical-bean group from the source filename."""
    match = SOURCE_NAME.fullmatch(Path(source_name).name)
    if match is None:
        raise ValueError(f"Nama sumber tidak mengikuti pola prefix+nomor: {source_name}")
    prefix = match.group("prefix").casefold()
    number = int(match.group("number"))
    return f"{prefix}-{number // 2:05d}"


def _request_json(url: str, *, body: dict | None = None, retries: int = 5) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt + 1 == retries:
                raise
            time.sleep(0.5 * 2**attempt)
    raise AssertionError("unreachable")


def _request_bytes(url: str, *, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == retries:
                raise
            time.sleep(0.5 * 2**attempt)
    raise AssertionError("unreachable")


def _search_sources(api_key: str) -> list[dict]:
    records: list[dict] = []
    offset = 0
    limit = 250  # Roboflow currently caps the public search response at 250.
    while True:
        response = _request_json(
            f"{API_BASE}/search?api_key={api_key}",
            body={
                "offset": offset,
                "limit": limit,
                "in_dataset": True,
                "fields": ["id", "name", "split", "owner"],
            },
        )
        page = response.get("results", [])
        if not isinstance(page, list):
            raise RuntimeError("Respons pencarian Roboflow tidak valid")
        records.extend(page)
        print(f"SOURCE INDEX: {len(records)}", flush=True)
        if len(page) < limit:
            break
        offset += len(page)
    ids = [str(row.get("id", "")) for row in records]
    names = [str(row.get("name", "")) for row in records]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise RuntimeError("ID atau nama citra sumber tidak unik")
    return sorted(records, key=lambda row: str(row["name"]).casefold())


def _box_rows(annotation: dict) -> tuple[list[str], Counter[int]]:
    width = float(annotation.get("width", 0))
    height = float(annotation.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("Dimensi anotasi tidak valid")
    rows: list[str] = []
    counts: Counter[int] = Counter()
    for box in annotation.get("boxes", []):
        label = str(box.get("label", "")).casefold()
        if label not in NAME_TO_ID:
            raise ValueError(f"Kelas tidak dikenal: {label}")
        values = [
            float(box["x"]) / width,
            float(box["y"]) / height,
            float(box["width"]) / width,
            float(box["height"]) / height,
        ]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Bounding box di luar rentang YOLO: {values}")
        class_id = NAME_TO_ID[label]
        rows.append(f"{class_id} " + " ".join(f"{value:.10f}" for value in values))
        counts[class_id] += 1
    if not rows:
        raise ValueError("Citra sumber tidak memiliki bounding box")
    return rows, counts


def _safe_detail(detail: dict, sha256: str) -> dict:
    return {
        "id": detail["id"],
        "name": detail["name"],
        "split": detail.get("split"),
        "annotation": detail["annotation"],
        "original_image_sha256": sha256,
    }


def _download_one(
    row: dict,
    api_key: str,
    output_root: Path,
    *,
    fetch_json: Callable[..., dict] = _request_json,
    fetch_bytes: Callable[..., bytes] = _request_bytes,
) -> dict:
    image_id = str(row["id"])
    source_name = str(row["name"])
    stem = f"{Path(source_name).stem}__{image_id}"
    image_path = output_root / "images" / f"{stem}.jpg"
    label_path = output_root / "labels" / f"{stem}.txt"
    detail_path = output_root / "metadata" / f"{image_id}.json"
    if image_path.is_file() and label_path.is_file() and detail_path.is_file():
        return json.loads(detail_path.read_text(encoding="utf-8"))

    detail_response = fetch_json(f"{API_BASE}/images/{image_id}?api_key={api_key}")
    detail = detail_response.get("image")
    if not isinstance(detail, dict) or str(detail.get("id")) != image_id:
        raise RuntimeError(f"Detail citra tidak valid: {image_id}")
    if str(detail.get("name")) != source_name:
        raise RuntimeError(f"Nama detail berubah untuk {image_id}")
    image_url = str(detail.get("urls", {}).get("original", ""))
    if not image_url.startswith("https://source.roboflow.com/"):
        raise RuntimeError(f"URL sumber tidak valid untuk {image_id}")
    image_bytes = fetch_bytes(image_url)
    with Image.open(BytesIO(image_bytes)) as image:
        image.verify()
    with Image.open(BytesIO(image_bytes)) as image:
        image_width, image_height = image.size
    annotation = detail.get("annotation", {})
    if (image_width, image_height) != (
        int(annotation.get("width", -1)),
        int(annotation.get("height", -1)),
    ):
        raise RuntimeError(f"Dimensi citra/anotasi berbeda untuk {source_name}")
    rows, _ = _box_rows(annotation)
    sha256 = hashlib.sha256(image_bytes).hexdigest()
    safe = _safe_detail(detail, sha256)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    detail_path.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")
    return safe


def download_original_sources(
    output_root: str | Path,
    *,
    api_key: str,
    workers: int = 12,
) -> dict:
    """Download the 4,038 original sources and source annotations read-only."""
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY wajib tersedia")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index = _search_sources(api_key)
    completed: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        pending = {
            executor.submit(_download_one, row, api_key, output_root): row for row in index
        }
        for number, future in enumerate(as_completed(pending), start=1):
            completed.append(future.result())
            if number % 100 == 0 or number == len(index):
                print(f"DOWNLOAD ORIGINAL: {number}/{len(index)}", flush=True)

    completed.sort(key=lambda row: str(row["name"]).casefold())
    if len(completed) != 4038:
        raise RuntimeError(f"Kontrak sumber gagal: diharapkan 4038, ditemukan {len(completed)}")
    payload = {
        "format": "coffee_detector.defectoscafeverde.original_sources.v1",
        "workspace": WORKSPACE,
        "project": PROJECT,
        "source_url": SOURCE_URL,
        "paper_url": PAPER_URL,
        "license": LICENSE,
        "images": len(completed),
        "classes": NAMES,
        "records": completed,
        "credentials_persisted": False,
    }
    manifest = output_root / "original_source_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _load_source_records(source_root: Path) -> list[SourceRecord]:
    manifest_path = source_root / "original_source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "coffee_detector.defectoscafeverde.original_sources.v1":
        raise RuntimeError("Format manifest sumber tidak dikenal")
    records = []
    for row in payload["records"]:
        image_id = str(row["id"])
        source_name = str(row["name"])
        stem = f"{Path(source_name).stem}__{image_id}"
        image_path = source_root / "images" / f"{stem}.jpg"
        label_path = source_root / "labels" / f"{stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Artefak sumber tidak lengkap: {stem}")
        rows, counts = _box_rows(row["annotation"])
        if label_path.read_text(encoding="utf-8").splitlines() != rows:
            raise RuntimeError(f"Label YOLO tidak cocok dengan manifest: {source_name}")
        sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if sha256 != row["original_image_sha256"]:
            raise RuntimeError(f"SHA citra sumber berubah: {source_name}")
        records.append(
            SourceRecord(
                image_id=image_id,
                source_name=source_name,
                source_split=str(row.get("split", "")),
                group_id=infer_physical_group(source_name),
                image_path=image_path,
                label_path=label_path,
                class_counts=counts,
                sha256=sha256,
            )
        )
    return records


def assign_grouped_splits(
    records: list[SourceRecord], *, seed: int = 42
) -> dict[str, list[SourceRecord]]:
    """Greedily stratify complete physical groups into 70/20/10 splits."""
    groups: dict[str, list[SourceRecord]] = {}
    for record in records:
        groups.setdefault(record.group_id, []).append(record)
    invalid = {group_id: items for group_id, items in groups.items() if len(items) > 2}
    if invalid:
        raise RuntimeError(f"Physical group berisi lebih dari dua sisi: {sorted(invalid)[:5]}")

    total_images = len(records)
    total_classes: Counter[int] = Counter()
    group_counts: dict[str, Counter[int]] = {}
    for group_id, items in groups.items():
        counts: Counter[int] = Counter()
        for item in items:
            counts.update(item.class_counts)
        group_counts[group_id] = counts
        total_classes.update(counts)

    rng = random.Random(seed)
    assignments: dict[str, list[SourceRecord]] = {split: [] for split in SPLIT_RATIOS}
    assigned_groups: dict[str, list[str]] = {split: [] for split in SPLIT_RATIOS}
    image_counts: Counter[str] = Counter()
    class_counts = {split: Counter() for split in SPLIT_RATIOS}

    def assign(split: str, group_id: str) -> None:
        assigned_groups[split].append(group_id)
        assignments[split].extend(groups[group_id])
        image_counts[split] += len(groups[group_id])
        class_counts[split].update(group_counts[group_id])

    # Partition each group by its dominant annotation class.  This prevents a
    # late, common class from merely filling the remaining global image slots,
    # which can satisfy the total ratio while badly skewing that class.
    primary_groups: dict[int, list[str]] = {class_id: [] for class_id in range(len(NAMES))}
    for group_id, counts in group_counts.items():
        primary = min(
            counts,
            key=lambda class_id: (-counts[class_id], class_id),
        )
        primary_groups[primary].append(group_id)

    for class_id in sorted(primary_groups, key=lambda value: len(primary_groups[value])):
        class_group_ids = primary_groups[class_id]
        if len(class_group_ids) < len(SPLIT_RATIOS):
            raise RuntimeError(
                f"Kelas dominan {NAMES[class_id]} memiliki terlalu sedikit physical group"
            )
        rng.shuffle(class_group_ids)
        class_group_ids.sort(
            key=lambda group_id: (
                group_counts[group_id][class_id],
                len(groups[group_id]),
            ),
            reverse=True,
        )
        total_primary_images = sum(len(groups[group_id]) for group_id in class_group_ids)
        total_primary_instances = sum(
            group_counts[group_id][class_id] for group_id in class_group_ids
        )
        local_images: Counter[str] = Counter()
        local_instances: Counter[str] = Counter()

        def primary_score(candidate_split: str, group_id: str) -> float:
            objective = 0.0
            overflow = 0.0
            for split, ratio in SPLIT_RATIOS.items():
                add_group = split == candidate_split
                proposed_images = local_images[split] + (
                    len(groups[group_id]) if add_group else 0
                )
                proposed_instances = local_instances[split] + (
                    group_counts[group_id][class_id] if add_group else 0
                )
                target_images = total_primary_images * ratio
                target_instances = total_primary_instances * ratio
                objective += ((proposed_images - target_images) / max(target_images, 1)) ** 2
                objective += (
                    (proposed_instances - target_instances) / max(target_instances, 1)
                ) ** 2
                overflow += max(0.0, proposed_images - target_images) / max(target_images, 1)
                overflow += max(0.0, proposed_instances - target_instances) / max(
                    target_instances, 1
                )
            return objective + 2.0 * overflow

        for group_id in class_group_ids:
            split = min(
                SPLIT_RATIOS,
                key=lambda candidate: (
                    primary_score(candidate, group_id),
                    local_images[candidate],
                ),
            )
            assign(split, group_id)
            local_images[split] += len(groups[group_id])
            local_instances[split] += group_counts[group_id][class_id]

    # Small deterministic repair to meet the global image ratio after the
    # class-stratified assignment.  Select each move by its complete
    # class-distribution cost, so ratio repair does not recreate class skew.
    target_images = {
        split: total_images * ratio for split, ratio in SPLIT_RATIOS.items()
    }

    def move_cost(group_id: str, source: str, target: str) -> float:
        objective = 0.0
        for split, ratio in SPLIT_RATIOS.items():
            proposed_images = image_counts[split]
            if split == source:
                proposed_images -= len(groups[group_id])
            elif split == target:
                proposed_images += len(groups[group_id])
            objective += (
                (proposed_images - target_images[split]) / max(target_images[split], 1)
            ) ** 2
            for class_id in range(len(NAMES)):
                proposed = class_counts[split][class_id]
                if split == source:
                    proposed -= group_counts[group_id][class_id]
                elif split == target:
                    proposed += group_counts[group_id][class_id]
                expected = total_classes[class_id] * ratio
                objective += ((proposed - expected) / max(expected, 1)) ** 2
        return objective

    for _ in range(len(groups)):
        deviations = {
            split: image_counts[split] - target_images[split] for split in SPLIT_RATIOS
        }
        source = max(deviations, key=deviations.get)
        target = min(deviations, key=deviations.get)
        if deviations[source] <= 1.0 or deviations[target] >= -1.0:
            break
        capacity = max(1, int(round(-deviations[target])))
        candidates = [
            group_id
            for group_id in assigned_groups[source]
            if len(groups[group_id]) <= capacity
            and all(
                class_counts[source][class_id] - group_counts[group_id][class_id] > 0
                for class_id in group_counts[group_id]
            )
        ]
        if not candidates:
            raise RuntimeError("Tidak dapat memperbaiki rasio tanpa menghilangkan kelas")
        group_id = min(candidates, key=lambda value: (move_cost(value, source, target), value))
        assigned_groups[source].remove(group_id)
        assigned_groups[target].append(group_id)
        for record in groups[group_id]:
            assignments[source].remove(record)
            assignments[target].append(record)
        image_counts[source] -= len(groups[group_id])
        image_counts[target] += len(groups[group_id])
        class_counts[source].subtract(group_counts[group_id])
        class_counts[target].update(group_counts[group_id])
    else:
        raise RuntimeError("Perbaikan rasio tidak konvergen")

    for split in SPLIT_RATIOS:
        missing = [class_id for class_id in range(len(NAMES)) if class_counts[split][class_id] == 0]
        if missing:
            raise RuntimeError(f"Split {split} kehilangan kelas: {missing}")
    seen: dict[str, str] = {}
    for split, group_ids in assigned_groups.items():
        for group_id in group_ids:
            if group_id in seen:
                raise RuntimeError(f"Physical group bocor: {group_id} ({seen[group_id]}, {split})")
            seen[group_id] = split
    return assignments


def build_grouped_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    copy_mode: str = "copy",
) -> dict:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if copy_mode not in {"copy", "hardlink"}:
        raise ValueError("copy_mode harus copy atau hardlink")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output tidak kosong: {output_root}")
    records = _load_source_records(source_root)
    if len(records) != 4038:
        raise RuntimeError(f"Kontrak sumber gagal: {len(records)} != 4038")
    assignments = assign_grouped_splits(records, seed=seed)
    manifest_rows = []
    class_counts = {}
    group_sets = {}
    for split, split_records in assignments.items():
        counts: Counter[int] = Counter()
        group_sets[split] = {record.group_id for record in split_records}
        for record in sorted(split_records, key=lambda item: item.source_name.casefold()):
            image_target = output_root / split / "images" / record.image_path.name
            label_target = output_root / split / "labels" / record.label_path.name
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            if copy_mode == "hardlink":
                os.link(record.image_path, image_target)
                os.link(record.label_path, label_target)
            else:
                shutil.copy2(record.image_path, image_target)
                shutil.copy2(record.label_path, label_target)
            counts.update(record.class_counts)
            manifest_rows.append(
                {
                    "image_id": record.image_id,
                    "source_name": record.source_name,
                    "source_split": record.source_split,
                    "physical_group_id": record.group_id,
                    "output_split": split,
                    "image": str(image_target.relative_to(output_root)).replace("\\", "/"),
                    "label": str(label_target.relative_to(output_root)).replace("\\", "/"),
                    "sha256": record.sha256,
                }
            )
        class_counts[split] = {NAMES[index]: counts[index] for index in range(len(NAMES))}

    cross_split = (
        (group_sets["train"] & group_sets["val"])
        | (group_sets["train"] & group_sets["test"])
        | (group_sets["val"] & group_sets["test"])
    )
    if cross_split:
        raise RuntimeError(f"Physical-group leakage: {sorted(cross_split)[:5]}")
    data_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(NAMES),
        "names": {index: name for index, name in enumerate(NAMES)},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (output_root / "grouped_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "format": "coffee_detector.defectoscafeverde.grouped_summary.v1",
        "status": "AUDIT_REQUIRED_BEFORE_TRAINING",
        "source_url": SOURCE_URL,
        "paper_url": PAPER_URL,
        "license": LICENSE,
        "grouping_rule": "lowercase alphabetic prefix + floor(sequence_number / 2)",
        "grouping_status": "inferred_from_paper_acquisition_and_filename_sequence",
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "source_images": len(records),
        "physical_groups": len(set(record.group_id for record in records)),
        "images_by_split": {split: len(items) for split, items in assignments.items()},
        "groups_by_split": {split: len(groups) for split, groups in group_sets.items()},
        "class_instances_by_split": class_counts,
        "cross_split_physical_groups": len(cross_split),
        "all_classes_present_each_split": all(
            all(value > 0 for value in class_counts[split].values()) for split in SPLIT_RATIOS
        ),
        "augmentation_applied": False,
        "training_authorized": False,
        "test_accessed": False,
    }
    sizes = Counter(record.group_id for record in records)
    summary["two_view_groups"] = sum(size == 2 for size in sizes.values())
    summary["single_view_groups"] = sum(size == 1 for size in sizes.values())
    (output_root / "grouped_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw DefectosCafeVerde and build a physical-bean grouped split."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--output-root", required=True)
    download.add_argument("--workers", type=int, default=12)
    split = subparsers.add_parser("split")
    split.add_argument("--source-root", required=True)
    split.add_argument("--output-root", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()
    if args.command == "download":
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        result = download_original_sources(
            args.output_root, api_key=api_key, workers=args.workers
        )
        print(json.dumps({key: result[key] for key in ("format", "images", "classes")}, indent=2))
    else:
        print(
            json.dumps(
                build_grouped_dataset(
                    args.source_root,
                    args.output_root,
                    seed=args.seed,
                    copy_mode=args.copy_mode,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
