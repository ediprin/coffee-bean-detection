"""Build the conservative Coffee Standard J25 v2 primary candidate.

Version 1 preferred the generated sibling with the largest number of boxes.
The visual audit showed that this can select a crop/copy-paste outlier.  Version
2 therefore quarantines every identity component whose siblings disagree on
the per-class annotation counts, then selects the visual medoid of each
remaining component.  No model result is used and training remains locked.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from coffee_detector.data.prepare_coffee_standard_primary import (
    J25_CLASSES,
    SPLIT_FRACTIONS,
    _link_or_copy,
    _resolve_source_root,
    _stable_number,
    grouped_three_way_assignment,
)
from coffee_detector.dataset import UnionFind, collect_records, discover_layout, write_json


FORMAT = "coffee_detector.coffee_standard_j25_primary_candidate.v2"
SELECTION_POLICY = "annotation-consistent identity quarantine plus visual medoid"


def _count_signature(record) -> tuple[int, ...]:
    counts = Counter(box.class_id for box in record.boxes)
    return tuple(counts[index] for index in range(len(J25_CLASSES)))


def _visual_distance(left, right) -> float:
    hash_distance = (left.dhash ^ right.dhash).bit_count() / 64.0
    colour_distance = max(
        abs(left_value - right_value)
        for left_value, right_value in zip(left.mean_rgb, right.mean_rgb)
    ) / 255.0
    return float(hash_distance + 0.25 * colour_distance)


def _visual_medoid(records: list, indices: list[int], seed: int):
    """Choose the observed sibling closest to all siblings in image space."""
    rows = [records[index] for index in indices]
    scored = []
    for row in rows:
        score = sum(_visual_distance(row, other) for other in rows)
        scored.append((score, _stable_number(seed, row.image_path.name), row))
    score, _, selected = min(scored, key=lambda item: (item[0], item[1]))
    return selected, float(score)


def _build_identity_components(records: list, seed: int) -> tuple[list[dict], dict, list[dict]]:
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

    components: list[dict] = []
    representatives = {}
    quarantined: list[dict] = []
    for ordinal, indices in enumerate(sorted(raw_components.values(), key=min)):
        component_id = f"j25-parent-{ordinal:05d}"
        signatures = [_count_signature(records[index]) for index in indices]
        unique_signatures = sorted(set(signatures))
        common = {
            "component_id": component_id,
            "source_siblings": len(indices),
            "parent_ids": sorted({records[index].parent_id for index in indices}),
            "source_hashes": sorted({records[index].sha256 for index in indices}),
        }
        if len(unique_signatures) != 1:
            quarantined.append(
                {
                    **common,
                    "reason": "sibling_class_count_disagreement",
                    "class_count_signatures": [list(signature) for signature in unique_signatures],
                    "source_records": [
                        {
                            "split": records[index].split,
                            "image": str(records[index].image_path),
                            "label": str(records[index].label_path),
                            "class_counts": list(signatures[position]),
                        }
                        for position, index in enumerate(indices)
                    ],
                }
            )
            continue

        representative, medoid_score = _visual_medoid(records, indices, seed)
        counts = np.asarray(unique_signatures[0], dtype=np.int64)
        components.append(
            {
                **common,
                "images": 1,
                "class_counts": counts.tolist(),
                "representative": str(representative.image_path),
                "selection_policy": "visual_medoid_within_annotation_consistent_identity",
                "visual_medoid_distance_sum": medoid_score,
            }
        )
        representatives[component_id] = representative
    return components, representatives, quarantined


def prepare_coffee_standard_primary_v2(
    source_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    link_mode: str = "auto",
) -> dict:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if link_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError("link_mode harus auto, hardlink, atau copy")
    if output_root.exists() and any(output_root.iterdir()):
        summary_path = output_root / "coffee_standard_j25_v2_summary.json"
        if summary_path.is_file():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached.get("status") == "complete" and cached.get("format") == FORMAT:
                return cached
        raise FileExistsError(f"Output tidak kosong: {output_root}")

    layout = discover_layout(_resolve_source_root(source_root))
    ordered_names = tuple(layout.names[index] for index in sorted(layout.names))
    if ordered_names != J25_CLASSES:
        raise ValueError(f"Ontologi export tidak sama dengan J25 yang dibekukan; diterima={ordered_names}")
    records, errors = collect_records(layout, compute_visual_features=True, progress=True)
    if errors:
        raise RuntimeError(f"Dataset memiliki image/label invalid: {errors[:10]}")

    components, representatives, quarantined = _build_identity_components(records, seed)
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
        image_target = output_root / split / "images" / f"{component_id}{record.image_path.suffix.lower()}"
        label_target = output_root / split / "labels" / f"{component_id}.txt"
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
                "selection_policy": component["selection_policy"],
                "visual_medoid_distance_sum": component["visual_medoid_distance_sum"],
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
    write_json(manifest, output_root / "coffee_standard_j25_v2_manifest.json")
    write_json(components, output_root / "coffee_standard_j25_v2_components.json")
    write_json(quarantined, output_root / "coffee_standard_j25_v2_quarantine.json")

    missing = {
        split: [
            J25_CLASSES[index]
            for index in range(len(J25_CLASSES))
            if class_counts[split][index] == 0
        ]
        for split in SPLIT_FRACTIONS
    }
    achieved = {split: image_counts[split] / max(1, len(components)) for split in SPLIT_FRACTIONS}
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
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    parent_overlap = sum(len(parent_sets[left] & parent_sets[right]) for left, right in pairs)
    hash_overlap = sum(len(hash_sets[left] & hash_sets[right]) for left, right in pairs)
    quarantined_images = sum(row["source_siblings"] for row in quarantined)
    technical_gates = {
        "original_25_label_ontology_preserved": True,
        "all_sibling_count_disagreements_quarantined": all(
            row["reason"] == "sibling_class_count_disagreement" for row in quarantined
        ),
        "one_visual_medoid_per_eligible_identity": len(manifest) == len(components),
        "zero_parent_overlap": parent_overlap == 0,
        "zero_exact_hash_overlap": hash_overlap == 0,
        "all_25_classes_in_train": not missing["train"],
        "all_25_classes_in_validation": not missing["val"],
        "all_25_classes_in_test": not missing["test"],
        "split_fraction_within_3_points": all(
            abs(achieved[split] - SPLIT_FRACTIONS[split]) <= 0.03 for split in SPLIT_FRACTIONS
        ),
    }
    summary = {
        "format": FORMAT,
        "status": "complete",
        "role": "provisional_primary_candidate_pending_v2_visual_and_provenance_review",
        "source_root": str(layout.root),
        "output_root": str(output_root),
        "seed": seed,
        "selection_policy": SELECTION_POLICY,
        "source_derivative_images": len(records),
        "source_identity_components": len(components) + len(quarantined),
        "eligible_identity_components": len(components),
        "quarantined_identity_components": len(quarantined),
        "quarantined_source_images": quarantined_images,
        "selected_representatives": len(manifest),
        "discarded_consistent_siblings": len(records) - quarantined_images - len(manifest),
        "class_count": len(J25_CLASSES),
        "ontology_scope": "original SNI+ICO labels; not asserted equivalent to canonical SNI21",
        "images_by_split": dict(image_counts),
        "boxes_by_split": dict(box_counts),
        "achieved_split_fractions": achieved,
        "class_distribution": {
            split: {J25_CLASSES[index]: class_counts[split][index] for index in range(len(J25_CLASSES))}
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
            "original raw captures are not explicitly marked in the public export",
        ],
        "training_authorized": False,
        "test_access_authorized": False,
        "training_executed": False,
        "test_images_accessed": False,
        "next_action": "run_v2_visual_review_then_resolve_author_or_metadata_provenance",
    }
    write_json(summary, output_root / "coffee_standard_j25_v2_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare conservative Coffee Standard J25 v2.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "copy"), default="auto")
    args = parser.parse_args()
    result = prepare_coffee_standard_primary_v2(
        args.source_root, args.output_root, seed=args.seed, link_mode=args.link_mode
    )
    print("STATUS:", result["status"])
    print("SOURCE IDENTITIES:", result["source_identity_components"])
    print("QUARANTINED:", result["quarantined_identity_components"])
    print("SELECTED:", result["selected_representatives"])
    print("SPLITS:", result["images_by_split"])
    print("MISSING:", result["missing_classes_by_split"])
    print("TECHNICAL GATES:", result["technical_gates"])
    print("TECHNICAL READY:", result["technical_split_ready"])
    print("TRAINING AUTHORIZED:", result["training_authorized"])
    print("SUMMARY:", Path(args.output_root).resolve() / "coffee_standard_j25_v2_summary.json")


if __name__ == "__main__":
    main()
