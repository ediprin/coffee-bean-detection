from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("valid", "val", "validation"),
    "test": ("test",),
}


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    yaml_path: Path
    names: dict[int, str]
    splits: dict[str, tuple[Path, Path]]


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ImageRecord:
    split: str
    image_path: Path
    label_path: Path
    boxes: tuple[Box, ...]
    sha256: str
    dhash: int
    mean_rgb: tuple[float, float, float]
    parent_id: str

    @property
    def class_counts(self) -> Counter[int]:
        return Counter(box.class_id for box in self.boxes)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _load_names(yaml_path: Path) -> dict[int, str]:
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError(f"'names' tidak ditemukan atau tidak valid: {yaml_path}")


def discover_layout(root: str | Path) -> DatasetLayout:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root}")

    yaml_candidates = [root / "data.yaml", root / "dataset.yaml"]
    yaml_candidates.extend(sorted(root.glob("*.yaml")))
    yaml_path = next((path for path in yaml_candidates if path.is_file()), None)
    if yaml_path is None:
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {root}")

    splits: dict[str, tuple[Path, Path]] = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            images = root / alias / "images"
            labels = root / alias / "labels"
            if images.is_dir() and labels.is_dir():
                splits[canonical] = (images, labels)
                break
    if "train" not in splits:
        raise FileNotFoundError(
            f"Split train YOLO tidak ditemukan di {root}; diharapkan train/images dan train/labels"
        )
    return DatasetLayout(root, yaml_path, _load_names(yaml_path), splits)


def _polygon_to_box(class_id: int, values: list[float]) -> Box:
    if len(values) < 6 or len(values) % 2:
        raise ValueError("Anotasi polygon harus berisi pasangan koordinat x,y")
    xs = values[0::2]
    ys = values[1::2]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return Box(
        class_id,
        (left + right) / 2,
        (top + bottom) / 2,
        right - left,
        bottom - top,
    )


def parse_label(path: Path, valid_class_ids: set[int]) -> tuple[Box, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Label tidak ditemukan: {path}")
    boxes: list[Box] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"Label tidak numerik {path}:{line_number}") from error
        if class_id not in valid_class_ids:
            raise ValueError(f"Class id {class_id} di luar data.yaml: {path}:{line_number}")
        if len(values) == 4:
            box = Box(class_id, *values)
        else:
            box = _polygon_to_box(class_id, values)
        coordinates = (box.x_center, box.y_center, box.width, box.height)
        if not all(np.isfinite(value) for value in coordinates):
            raise ValueError(f"Koordinat non-finite: {path}:{line_number}")
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"Koordinat di luar rentang [0,1]: {path}:{line_number}")
        if box.width <= 0 or box.height <= 0:
            raise ValueError(f"Box kosong: {path}:{line_number}")
        boxes.append(box)
    return tuple(boxes)


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    differences = pixels[:, 1:] > pixels[:, :-1]
    result = 0
    for bit in differences.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def image_mean_rgb(path: Path) -> tuple[float, float, float]:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR))
    means = pixels.reshape(-1, 3).mean(axis=0)
    return tuple(float(value) for value in means)


def roboflow_parent_id(path: Path) -> str:
    stem = path.stem
    if ".rf." in stem:
        return stem.split(".rf.", 1)[0]
    return stem


def collect_records(layout: DatasetLayout) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    errors: list[str] = []
    valid_ids = set(layout.names)
    for split, (image_root, label_root) in layout.splits.items():
        image_paths = sorted(
            path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in image_paths:
            relative = image_path.relative_to(image_root)
            label_path = (label_root / relative).with_suffix(".txt")
            try:
                boxes = parse_label(label_path, valid_ids)
                records.append(
                    ImageRecord(
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        boxes=boxes,
                        sha256=image_sha256(image_path),
                        dhash=image_dhash(image_path),
                        mean_rgb=image_mean_rgb(image_path),
                        parent_id=roboflow_parent_id(image_path),
                    )
                )
            except (OSError, ValueError) as error:
                errors.append(str(error))
    return records, errors


def duplicate_components(
    records: list[ImageRecord], near_threshold: int = 4
) -> tuple[list[list[int]], dict[str, int]]:
    union_find = UnionFind(len(records))
    counters = {"exact_pairs": 0, "parent_pairs": 0, "near_pairs": 0}

    for attribute, counter_name in (("sha256", "exact_pairs"), ("parent_id", "parent_pairs")):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[str(getattr(record, attribute))].append(index)
        for indices in groups.values():
            for index in indices[1:]:
                union_find.union(indices[0], index)
                counters[counter_name] += 1

    # Eight 8-bit bands make candidate discovery cheap. With Hamming distance <= 4,
    # at least one band must be identical.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        candidates: set[int] = set()
        for band in range(8):
            value = (record.dhash >> (band * 8)) & 0xFF
            candidates.update(buckets[(band, value)])
        for candidate in candidates:
            color_distance = max(
                abs(left - right)
                for left, right in zip(record.mean_rgb, records[candidate].mean_rgb)
            )
            if (
                (record.dhash ^ records[candidate].dhash).bit_count() <= near_threshold
                and color_distance <= 12.0
            ):
                union_find.union(index, candidate)
                counters["near_pairs"] += 1
        for band in range(8):
            value = (record.dhash >> (band * 8)) & 0xFF
            buckets[(band, value)].append(index)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[union_find.find(index)].append(index)
    return list(components.values()), counters


def build_audit(layout: DatasetLayout, near_threshold: int = 4) -> dict:
    records, errors = collect_records(layout)
    components, pair_counts = duplicate_components(records, near_threshold)
    split_images = Counter(record.split for record in records)
    split_boxes = Counter()
    class_boxes = Counter()
    class_images = Counter()
    empty_images = Counter()
    for record in records:
        split_boxes[record.split] += len(record.boxes)
        if not record.boxes:
            empty_images[record.split] += 1
        class_boxes.update(record.class_counts)
        class_images.update(set(record.class_counts))

    cross_split_groups = []
    for component in components:
        splits = sorted({records[index].split for index in component})
        if len(splits) > 1:
            cross_split_groups.append(
                {
                    "splits": splits,
                    "files": [str(records[index].image_path) for index in component],
                }
            )

    exact_conflicts = []
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)
    for same_images in by_hash.values():
        signatures = {
            tuple(sorted((box.class_id, box.x_center, box.y_center, box.width, box.height) for box in item.boxes))
            for item in same_images
        }
        if len(signatures) > 1:
            exact_conflicts.append([str(item.label_path) for item in same_images])

    return {
        "dataset_root": str(layout.root),
        "yaml": str(layout.yaml_path),
        "classes": layout.names,
        "images_by_split": dict(split_images),
        "boxes_by_split": dict(split_boxes),
        "empty_images_by_split": dict(empty_images),
        "boxes_by_class": {layout.names[key]: class_boxes[key] for key in sorted(layout.names)},
        "images_by_class": {layout.names[key]: class_images[key] for key in sorted(layout.names)},
        "duplicate_pair_candidates": pair_counts,
        "duplicate_components": sum(len(component) > 1 for component in components),
        "cross_split_duplicate_components": len(cross_split_groups),
        "cross_split_examples": cross_split_groups[:50],
        "exact_image_annotation_conflicts": exact_conflicts[:50],
        "errors": errors,
        "safe_for_training": not errors and not cross_split_groups and not exact_conflicts,
    }


def write_json(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def iter_component_records(
    records: list[ImageRecord], components: Iterable[list[int]]
) -> Iterable[list[ImageRecord]]:
    for component in components:
        yield [records[index] for index in component]
