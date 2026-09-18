"""Build a leakage-controlled Jundullah/Coffee Standard primary candidate.

The public Roboflow v8 export contains generated siblings and an unsafe official
split.  This module deliberately keeps one deterministic representative per
identity component and preserves the original 25-label SNI+ICO ontology.  It
does not authorize model training; visual annotation and provenance review are
separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from coffee_detector.dataset import (
    IMAGE_SUFFIXES,
    UnionFind,
    collect_records,
    discover_layout,
    write_json,
)


J25_CLASSES = (
    "Biji Berjamur -Moldy Bean-",
    "Biji Berkulit Ari",
    "Biji Berkulit Tanduk",
    "Biji Berlubang Lebih dari satu",
    "Biji Berlubang Satu",
    "Biji Bertutul",
    "Biji Cokelat",
    "Biji Hitam Pecah",
    "Biji Hitam Penuh",
    "Biji Hitam Sebagian",
    "Biji Muda",
    "Biji Normal",
    "Biji Pecah",
    "Brown Yellow -Sour Bean-",
    "Kerikil",
    "Kopi Gelondong",
    "Kulit Kopi Ukuran Besar",
    "Kulit Kopi Ukuran Kecil",
    "Kulit Kopi Ukuran Sedang",
    "Kulit Tanduk Ukuran Besar",
    "Kulit Tanduk Ukuran Kecil",
    "Kulit Tanduk Ukuran Sedang",
    "Ranting Ukuran Besar",
    "Ranting Ukuran Kecil",
    "Ranting Ukuran Sedang",
)
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
RETRACTION = (
    "RETRACTED J25 rebuild: stripped Roboflow filenames are not authoritative source IDs. "
    "Use the author-provided official split and the thesis-provenance audit instead."
)


def _resolve_source_root(root: Path) -> Path:
    candidates = [root, *sorted({path.parent for path in root.rglob("data.yaml")})]
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


def _stable_number(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _component_objective(
    assignments: np.ndarray,
    sizes: np.ndarray,
    counts: np.ndarray,
    fractions: np.ndarray,
) -> float:
    total_size = float(sizes.sum())
    total_counts = counts.sum(axis=0).astype(np.float64)
    score = 0.0
    for split_index, fraction in enumerate(fractions):
        selected = assignments == split_index
        split_size = float(sizes[selected].sum())
        split_counts = counts[selected].sum(axis=0).astype(np.float64)
        target_size = max(1.0, total_size * fraction)
        target_counts = np.maximum(1.0, total_counts * fraction)
        score += ((split_size - target_size) / target_size) ** 2
        active = total_counts > 0
        score += float(np.mean(((split_counts[active] - target_counts[active]) / target_counts[active]) ** 2))
        missing = active & (split_counts == 0)
        score += float(missing.sum()) * 100.0
    return score


def grouped_three_way_assignment(
    components: list[dict], *, seed: int, restarts: int = 64
) -> tuple[dict[str, str], dict]:
    """Assign identity components to train/val/test with deterministic search."""
    if not components:
        raise ValueError("Tidak ada identity component untuk dibagi")
    sizes = np.asarray([row["images"] for row in components], dtype=np.int64)
    counts = np.asarray([row["class_counts"] for row in components], dtype=np.int64)
    fractions = np.asarray(list(SPLIT_FRACTIONS.values()), dtype=np.float64)
    best: tuple[float, np.ndarray] | None = None
    for restart in range(restarts):
        order = sorted(
            range(len(components)),
            key=lambda index: _stable_number(
                seed + restart * 1009, str(components[index]["component_id"])
            ),
        )
        assignments = np.asarray(
            [index % len(SPLIT_FRACTIONS) for index in range(len(components))],
            dtype=np.int64,
        )
        assignments[:] = 0
        # Seed validation/test with class-rich components before local search.
        rich = sorted(order, key=lambda index: (-np.count_nonzero(counts[index]), order.index(index)))
        if len(rich) >= 2:
            assignments[rich[0]] = 1
            assignments[rich[1]] = 2
        score = _component_objective(assignments, sizes, counts, fractions)
        for _ in range(8):
            improved = False
            for index in order:
                current = int(assignments[index])
                candidate_best = score
                destination_best = current
                for destination in range(3):
                    if destination == current:
                        continue
                    assignments[index] = destination
                    candidate = _component_objective(assignments, sizes, counts, fractions)
                    if candidate + 1e-12 < candidate_best:
                        candidate_best = candidate
                        destination_best = destination
                assignments[index] = destination_best
                if destination_best != current:
                    score = candidate_best
                    improved = True
            if not improved:
                break
        candidate = (score, assignments.copy())
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    score, assignments = best
    split_names = tuple(SPLIT_FRACTIONS)
    mapping = {
        str(component["component_id"]): split_names[int(assignments[index])]
        for index, component in enumerate(components)
    }
    return mapping, {
        "objective": float(score),
        "seed": seed,
        "restarts": restarts,
        "target_fractions": SPLIT_FRACTIONS,
    }


def _link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
        return
    try:
        os.link(source, target)
    except OSError:
        if mode == "hardlink":
            raise
        shutil.copy2(source, target)


def _choose_representative(records: list, indices: list[int], seed: int):
    # Prefer the sibling retaining the most annotations; ties are stable and do
    # not use validation/test model behavior.
    return min(
        (records[index] for index in indices),
        key=lambda row: (-len(row.boxes), _stable_number(seed, row.image_path.name)),
    )


def prepare_coffee_standard_primary(
    source_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    link_mode: str = "auto",
) -> dict:
    raise RuntimeError(RETRACTION)
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if link_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError("link_mode harus auto, hardlink, atau copy")
    if output_root.exists() and any(output_root.iterdir()):
        summary_path = output_root / "coffee_standard_j25_summary.json"
        if summary_path.is_file():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached.get("status") == "complete":
                return cached
        raise FileExistsError(f"Output tidak kosong: {output_root}")

    layout = discover_layout(_resolve_source_root(source_root))
    ordered_names = tuple(layout.names[index] for index in sorted(layout.names))
    if ordered_names != J25_CLASSES:
        raise ValueError(
            "Ontologi export tidak sama dengan J25 yang dibekukan; "
            f"diterima={ordered_names}"
        )
    records, errors = collect_records(layout, compute_visual_features=False, progress=True)
    if errors:
        raise RuntimeError(f"Dataset memiliki image/label invalid: {errors[:10]}")

    union = UnionFind(len(records))
    by_parent: dict[str, list[int]] = defaultdict(list)
    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_parent[record.parent_id].append(index)
        by_hash[record.sha256].append(index)
    for groups in (by_parent, by_hash):
        for indices in groups.values():
            for index in indices[1:]:
                union.union(indices[0], index)
    raw_components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        raw_components[union.find(index)].append(index)

    components, representatives = [], {}
    for ordinal, indices in enumerate(sorted(raw_components.values(), key=min)):
        component_id = f"j25-parent-{ordinal:05d}"
        representative = _choose_representative(records, indices, seed)
        counts = np.zeros(len(J25_CLASSES), dtype=np.int64)
        for box in representative.boxes:
            counts[box.class_id] += 1
        components.append(
            {
                "component_id": component_id,
                "images": 1,
                "source_siblings": len(indices),
                "class_counts": counts.tolist(),
                "parent_ids": sorted({records[index].parent_id for index in indices}),
                "source_hashes": sorted({records[index].sha256 for index in indices}),
                "representative": str(representative.image_path),
            }
        )
        representatives[component_id] = representative

    assignments, optimization = grouped_three_way_assignment(components, seed=seed)
    output_root.mkdir(parents=True, exist_ok=True)
    class_counts: dict[str, Counter] = defaultdict(Counter)
    image_counts = Counter()
    box_counts = Counter()
    manifest = []
    for component in components:
        component_id = component["component_id"]
        split = assignments[component_id]
        record = representatives[component_id]
        suffix = record.image_path.suffix.lower()
        image_name = f"{component_id}{suffix}"
        label_name = f"{component_id}.txt"
        image_target = output_root / split / "images" / image_name
        label_target = output_root / split / "labels" / label_name
        _link_or_copy(record.image_path, image_target, link_mode)
        _link_or_copy(record.label_path, label_target, link_mode)
        image_counts[split] += 1
        box_counts[split] += len(record.boxes)
        for box in record.boxes:
            class_counts[split][box.class_id] += 1
        manifest.append(
            {
                "component_id": component_id,
                "output_split": split,
                "source_split": record.split,
                "source_parent_id": record.parent_id,
                "source_sha256": record.sha256,
                "source_image": str(record.image_path),
                "source_label": str(record.label_path),
                "output_image": str(image_target),
                "output_label": str(label_target),
                "source_siblings": component["source_siblings"],
                "class_counts": component["class_counts"],
            }
        )

    (output_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output_root),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": {index: name for index, name in enumerate(J25_CLASSES)},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    write_json(manifest, output_root / "coffee_standard_j25_manifest.json")
    write_json(components, output_root / "coffee_standard_j25_components.json")

    missing = {
        split: [
            J25_CLASSES[index]
            for index in range(len(J25_CLASSES))
            if class_counts[split][index] == 0
        ]
        for split in SPLIT_FRACTIONS
    }
    achieved = {
        split: image_counts[split] / max(1, len(components)) for split in SPLIT_FRACTIONS
    }
    parent_sets = {
        split: {
            parent
            for component in components
            if assignments[component["component_id"]] == split
            for parent in component["parent_ids"]
        }
        for split in SPLIT_FRACTIONS
    }
    hash_sets = {
        split: {
            digest
            for component in components
            if assignments[component["component_id"]] == split
            for digest in component["source_hashes"]
        }
        for split in SPLIT_FRACTIONS
    }
    overlap_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    parent_overlap = sum(len(parent_sets[left] & parent_sets[right]) for left, right in overlap_pairs)
    hash_overlap = sum(len(hash_sets[left] & hash_sets[right]) for left, right in overlap_pairs)
    technical_gates = {
        "original_25_label_ontology_preserved": True,
        "one_representative_per_identity_component": len(manifest) == len(components),
        "zero_parent_overlap": parent_overlap == 0,
        "zero_exact_hash_overlap": hash_overlap == 0,
        "all_25_classes_in_train": not missing["train"],
        "all_25_classes_in_validation": not missing["val"],
        "all_25_classes_in_test": not missing["test"],
        "split_fraction_within_3_points": all(
            abs(achieved[split] - SPLIT_FRACTIONS[split]) <= 0.03
            for split in SPLIT_FRACTIONS
        ),
    }
    summary = {
        "format": "coffee_detector.coffee_standard_j25_primary_candidate.v1",
        "status": "complete",
        "role": "provisional_primary_candidate_pending_visual_and_provenance_review",
        "source_root": str(layout.root),
        "output_root": str(output_root),
        "seed": seed,
        "source_derivative_images": len(records),
        "identity_components": len(components),
        "selected_representatives": len(manifest),
        "discarded_generated_siblings": len(records) - len(manifest),
        "class_count": len(J25_CLASSES),
        "ontology_scope": "original SNI+ICO labels; not asserted equivalent to canonical SNI21",
        "images_by_split": dict(image_counts),
        "boxes_by_split": dict(box_counts),
        "achieved_split_fractions": achieved,
        "class_distribution": {
            split: {
                J25_CLASSES[index]: class_counts[split][index]
                for index in range(len(J25_CLASSES))
            }
            for split in SPLIT_FRACTIONS
        },
        "missing_classes_by_split": missing,
        "cross_split_parent_ids": parent_overlap,
        "cross_split_exact_hashes": hash_overlap,
        "optimization": optimization,
        "technical_gates": technical_gates,
        "technical_split_ready": all(technical_gates.values()),
        "unresolved_provenance_gates": [
            "paper reports 2,000 augmented images and 20 classes, while thesis/public v8 reports 451 images and 25 classes",
            "public v8 export yields a different recoverable identity-component count than the thesis image count",
            "selected representatives require visual annotation review because original raw captures are not explicitly marked",
        ],
        "training_authorized": False,
        "test_access_authorized": False,
        "training_executed": False,
        "test_images_accessed": False,
        "next_action": "visual_annotation_review_and_author_or_metadata_provenance_resolution",
    }
    write_json(summary, output_root / "coffee_standard_j25_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the leakage-controlled Coffee Standard J25 primary candidate."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy"), default="auto")
    args = parser.parse_args()
    result = prepare_coffee_standard_primary(
        args.source_root, args.output_root, seed=args.seed, link_mode=args.link_mode
    )
    print("STATUS:", result["status"])
    print("ROLE:", result["role"])
    print("SOURCE/REPRESENTATIVES:", result["source_derivative_images"], result["selected_representatives"])
    print("SPLITS:", result["images_by_split"])
    print("TECHNICAL READY:", result["technical_split_ready"])
    print("TRAINING AUTHORIZED:", result["training_authorized"])
    print("SUMMARY:", Path(args.output_root).resolve() / "coffee_standard_j25_summary.json")


if __name__ == "__main__":
    main()
