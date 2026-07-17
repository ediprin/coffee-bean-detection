from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from .dataset import (
    ImageRecord,
    collect_records,
    discover_layout,
    duplicate_components,
    iter_component_records,
    write_json,
)


@dataclass
class Group:
    group_id: str
    records: list[ImageRecord]
    class_counts: Counter[int]

    @property
    def size(self) -> int:
        return len(self.records)


def _make_groups(records: list[ImageRecord], near_threshold: int) -> list[Group]:
    components, _ = duplicate_components(records, near_threshold)
    groups = []
    for index, items in enumerate(iter_component_records(records, components)):
        counts: Counter[int] = Counter()
        for item in items:
            counts.update(item.class_counts)
        groups.append(Group(f"group-{index:06d}", items, counts))
    return groups


def _assign_groups(
    groups: list[Group], class_ids: list[int], seed: int, ratios: dict[str, float]
) -> dict[str, list[Group]]:
    rng = random.Random(seed)
    total_images = sum(group.size for group in groups)
    total_classes: Counter[int] = Counter()
    for group in groups:
        total_classes.update(group.class_counts)

    shuffled = groups[:]
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda group: (
            max((count / max(total_classes[class_id], 1) for class_id, count in group.class_counts.items()), default=0),
            group.size,
        ),
        reverse=True,
    )
    assigned = {split: [] for split in ratios}
    split_images = Counter()
    split_classes = {split: Counter() for split in ratios}

    def score(split: str, group: Group) -> float:
        target_images = total_images * ratios[split]
        image_error = ((split_images[split] + group.size - target_images) / max(target_images, 1)) ** 2
        class_error = 0.0
        for class_id in class_ids:
            target = total_classes[class_id] * ratios[split]
            proposed = split_classes[split][class_id] + group.class_counts[class_id]
            class_error += ((proposed - target) / max(target, 1)) ** 2
        overflow = max(0.0, split_images[split] + group.size - target_images) / max(target_images, 1)
        return image_error + class_error / max(len(class_ids), 1) + 2.0 * overflow

    split_order = list(ratios)
    for group in shuffled:
        best = min(split_order, key=lambda split: (score(split, group), split_images[split]))
        assigned[best].append(group)
        split_images[best] += group.size
        split_classes[best].update(group.class_counts)
    return assigned


def _copy_record(record: ImageRecord, output_root: Path, split: str, suffix: str = "") -> dict:
    image_dir = output_root / split / "images"
    label_dir = output_root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{record.image_path.stem}{suffix}"
    image_target = image_dir / f"{output_stem}{record.image_path.suffix.lower()}"
    label_target = label_dir / f"{output_stem}.txt"
    counter = 1
    while image_target.exists() or label_target.exists():
        output_stem = f"{record.image_path.stem}{suffix}-{counter}"
        image_target = image_dir / f"{output_stem}{record.image_path.suffix.lower()}"
        label_target = label_dir / f"{output_stem}.txt"
        counter += 1
    shutil.copy2(record.image_path, image_target)
    shutil.copy2(record.label_path, label_target)
    return {
        "source_image": str(record.image_path),
        "source_split": record.split,
        "output_image": str(image_target),
        "sha256": record.sha256,
        "parent_id": record.parent_id,
    }


def prepare_dataset(
    data_root: str | Path,
    output_root: str | Path,
    seed: int = 42,
    near_threshold: int = 4,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    drop_exact_duplicates: bool = True,
) -> dict:
    if abs(sum(ratios) - 1.0) > 1e-9 or any(value <= 0 for value in ratios):
        raise ValueError("Rasio train/val/test harus positif dan berjumlah 1")
    layout = discover_layout(data_root)
    records, errors = collect_records(layout)
    if errors:
        raise RuntimeError("Dataset mengandung label/gambar tidak valid:\n- " + "\n- ".join(errors[:20]))
    exact_signatures: dict[str, set[tuple]] = {}
    for record in records:
        signature = tuple(
            sorted(
                (box.class_id, box.x_center, box.y_center, box.width, box.height)
                for box in record.boxes
            )
        )
        exact_signatures.setdefault(record.sha256, set()).add(signature)
    conflicts = [sha for sha, signatures in exact_signatures.items() if len(signatures) > 1]
    if conflicts:
        raise RuntimeError(
            f"Ada {len(conflicts)} exact duplicate dengan anotasi berbeda; selesaikan konflik sebelum split."
        )
    groups = _make_groups(records, near_threshold)
    assignments = _assign_groups(
        groups,
        sorted(layout.names),
        seed,
        {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
    )
    empty_assignments = [split for split, split_groups in assignments.items() if not split_groups]
    if empty_assignments:
        raise RuntimeError(
            "Grouped split menghasilkan split kosong "
            f"({', '.join(empty_assignments)}); dataset memiliki terlalu sedikit grup independen."
        )

    output_root = Path(output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output tidak kosong: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    seen_hashes: set[str] = set()
    dropped = []
    for split, split_groups in assignments.items():
        for group in split_groups:
            for record in group.records:
                if drop_exact_duplicates and record.sha256 in seen_hashes:
                    dropped.append(str(record.image_path))
                    continue
                seen_hashes.add(record.sha256)
                row = _copy_record(record, output_root, split)
                row.update({"output_split": split, "group_id": group.group_id})
                manifest.append(row)

    data_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": layout.names,
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    result = {
        "source": str(layout.root),
        "output": str(output_root),
        "seed": seed,
        "near_threshold": near_threshold,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "images": dict(Counter(row["output_split"] for row in manifest)),
        "groups": {split: len(items) for split, items in assignments.items()},
        "dropped_exact_duplicates": len(dropped),
        "dropped_files": dropped[:100],
        "manifest": manifest,
    }
    write_json(result, output_root / "split_manifest.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Buat grouped split YOLO tanpa mengubah dataset sumber.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--keep-exact-duplicates", action="store_true")
    args = parser.parse_args()
    result = prepare_dataset(
        args.data_root,
        args.output_root,
        seed=args.seed,
        near_threshold=args.near_threshold,
        drop_exact_duplicates=not args.keep_exact_duplicates,
    )
    print("=== GROUPED SPLIT SELESAI ===")
    print(f"Source : {result['source']}")
    print(f"Output : {result['output']}")
    print(f"Images : {result['images']}")
    print(f"Groups : {result['groups']}")
    print(f"Exact duplicate dibuang: {result['dropped_exact_duplicates']}")


if __name__ == "__main__":
    main()
