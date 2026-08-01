from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .audit_dataset import audit_dataset
from .dataset import UnionFind, parse_label, write_json
from .prepare_sni_fullscene import SNI21_CLASSES


def _class_counts(label_path: Path) -> np.ndarray:
    counts = np.zeros(len(SNI21_CLASSES), dtype=np.int64)
    for box in parse_label(label_path, set(range(len(SNI21_CLASSES)))):
        counts[box.class_id] += 1
    return counts


def _stable_number(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _objective(
    val_images: int,
    val_classes: np.ndarray,
    total_images: int,
    total_classes: np.ndarray,
    val_fraction: float,
) -> float:
    target_images = max(1.0, total_images * val_fraction)
    target_classes = np.maximum(1.0, total_classes * val_fraction)
    image_error = ((val_images - target_images) / target_images) ** 2
    class_error = float(np.mean(((val_classes - target_classes) / target_classes) ** 2))
    active = total_classes > 0
    minimum_val = np.minimum(10, np.maximum(1, np.floor(total_classes * 0.08))).astype(int)
    minimum_train = np.minimum(10, np.maximum(1, np.floor(total_classes * 0.08))).astype(int)
    train_classes = total_classes - val_classes
    shortage = np.maximum(0, minimum_val - val_classes) + np.maximum(0, minimum_train - train_classes)
    support_penalty = float(shortage[active].sum()) * 25.0
    return support_penalty + image_error + class_error


def grouped_stratified_assignment(
    groups: list[dict], *, seed: int, val_fraction: float, restarts: int = 48
) -> tuple[set[str], dict]:
    if not 0.05 <= val_fraction <= 0.40:
        raise ValueError("val_fraction harus berada pada 0.05..0.40")
    total_images = sum(int(group["images"]) for group in groups)
    total_classes = sum(
        (np.asarray(group["class_counts"], dtype=np.int64) for group in groups),
        start=np.zeros(len(SNI21_CLASSES), dtype=np.int64),
    )
    if not groups or total_images == 0:
        raise ValueError("Tidak ada group untuk dibagi")

    best = None
    for restart in range(restarts):
        order = sorted(
            range(len(groups)),
            key=lambda index: _stable_number(
                seed + restart * 1009, str(groups[index]["group_id"])
            ),
        )
        selected = np.zeros(len(groups), dtype=bool)
        val_images = 0
        val_classes = np.zeros(len(SNI21_CLASSES), dtype=np.int64)
        score = _objective(
            val_images, val_classes, total_images, total_classes, val_fraction
        )

        for index in order:
            group = groups[index]
            candidate_images = val_images + int(group["images"])
            candidate_classes = val_classes + np.asarray(group["class_counts"])
            candidate_score = _objective(
                candidate_images,
                candidate_classes,
                total_images,
                total_classes,
                val_fraction,
            )
            if candidate_score < score:
                selected[index] = True
                val_images = candidate_images
                val_classes = candidate_classes
                score = candidate_score

        for _ in range(4):
            improved = False
            for index in order:
                group = groups[index]
                direction = -1 if selected[index] else 1
                candidate_images = val_images + direction * int(group["images"])
                candidate_classes = val_classes + direction * np.asarray(
                    group["class_counts"]
                )
                candidate_score = _objective(
                    candidate_images,
                    candidate_classes,
                    total_images,
                    total_classes,
                    val_fraction,
                )
                if candidate_score + 1e-12 < score:
                    selected[index] = not selected[index]
                    val_images = candidate_images
                    val_classes = candidate_classes
                    score = candidate_score
                    improved = True
            if not improved:
                break

        candidate = (score, selected.copy(), val_images, val_classes.copy())
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    score, selected, val_images, val_classes = best
    val_groups = {
        str(groups[index]["group_id"])
        for index in range(len(groups))
        if selected[index]
    }
    return val_groups, {
        "objective": float(score),
        "target_val_fraction": val_fraction,
        "achieved_val_fraction": val_images / total_images,
        "total_images": total_images,
        "val_images": int(val_images),
        "train_images": int(total_images - val_images),
        "total_class_counts": total_classes.tolist(),
        "val_class_counts": val_classes.tolist(),
        "train_class_counts": (total_classes - val_classes).tolist(),
        "restarts": restarts,
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


def group_faruq_development(
    repaired_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    val_fraction: float = 0.15,
    link_mode: str = "auto",
) -> dict:
    repaired_root = Path(repaired_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if link_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError("link_mode harus auto, hardlink, atau copy")
    manifest_path = repaired_root / "faruq_geometry_repair_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest Faruq-v2 tidak ditemukan: {manifest_path}")
    if output_root.exists() and any(output_root.iterdir()):
        summary_path = output_root / "faruq_grouped_summary.json"
        if summary_path.is_file():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached.get("status") == "complete":
                print(f"REUSE FARUQ GROUPED: {output_root}", flush=True)
                return cached
        raise FileExistsError(f"Output grouped tidak kosong/complete: {output_root}")

    source_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for row in source_rows:
        original_split = str(row["split"])
        image_name = Path(row["output_image"]).name
        label_name = Path(row["output_label"]).name
        image_path = repaired_root / original_split / "images" / image_name
        label_path = repaired_root / original_split / "labels" / label_name
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Pasangan image/label hilang: {image_path}")
        records.append(
            {
                **row,
                "image_path": image_path,
                "label_path": label_path,
                "class_counts": _class_counts(label_path),
            }
        )

    union = UnionFind(len(records))
    indices_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        indices_by_key[("parent", str(row["source_parent_id"]))].append(index)
        indices_by_key[("sha256", str(row["source_sha256"]))].append(index)
    for indices in indices_by_key.values():
        for index in indices[1:]:
            union.union(indices[0], index)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[union.find(index)].append(index)

    groups = []
    group_for_index = {}
    for ordinal, indices in enumerate(
        sorted(components.values(), key=lambda values: min(values))
    ):
        parents = sorted({str(records[index]["source_parent_id"]) for index in indices})
        hashes = sorted({str(records[index]["source_sha256"]) for index in indices})
        group_id = f"faruq-group-{ordinal:05d}"
        counts = sum(
            (records[index]["class_counts"] for index in indices),
            start=np.zeros(len(SNI21_CLASSES), dtype=np.int64),
        )
        groups.append(
            {
                "group_id": group_id,
                "indices": indices,
                "images": len(indices),
                "class_counts": counts.tolist(),
                "parents": parents,
                "hashes": hashes,
            }
        )
        for index in indices:
            group_for_index[index] = group_id

    val_groups, optimization = grouped_stratified_assignment(
        groups, seed=seed, val_fraction=val_fraction
    )
    output_root.mkdir(parents=True, exist_ok=True)
    split_images = Counter()
    split_boxes = Counter()
    split_classes: dict[str, Counter] = defaultdict(Counter)
    parents_by_split: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    output_manifest = []
    used_names: dict[tuple[str, str], Path] = {}

    for index, row in enumerate(records):
        group_id = group_for_index[index]
        split = "val" if group_id in val_groups else "train"
        image_name = row["image_path"].name
        key = (split, image_name.lower())
        if key in used_names:
            raise RuntimeError(
                f"Nama gambar bertabrakan: {row['image_path']} dan {used_names[key]}"
            )
        used_names[key] = row["image_path"]
        image_target = output_root / split / "images" / image_name
        label_target = output_root / split / "labels" / row["label_path"].name
        _link_or_copy(row["image_path"], image_target, link_mode)
        _link_or_copy(row["label_path"], label_target, link_mode)
        class_counts = np.asarray(row["class_counts"], dtype=np.int64)
        split_images[split] += 1
        split_boxes[split] += int(class_counts.sum())
        for class_id, count in enumerate(class_counts):
            if count:
                split_classes[split][SNI21_CLASSES[class_id]] += int(count)
        parents_by_split[split].add(str(row["source_parent_id"]))
        hashes_by_split[split].add(str(row["source_sha256"]))
        output_manifest.append(
            {
                "group_id": group_id,
                "source_parent_id": row["source_parent_id"],
                "source_sha256": row["source_sha256"],
                "original_split": row["split"],
                "output_split": split,
                "input_image": str(row["image_path"]),
                "output_image": str(image_target),
                "output_label": str(label_target),
                "class_counts": class_counts.tolist(),
            }
        )

    names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output_root),
                "train": "train/images",
                "val": "val/images",
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    write_json(output_manifest, output_root / "faruq_grouped_manifest.json")
    write_json(groups, output_root / "faruq_group_manifest.json")
    dataset_audit = audit_dataset(
        output_root, output_root / "dataset_audit.json", near_threshold=-1
    )
    parent_overlap = parents_by_split["train"] & parents_by_split["val"]
    hash_overlap = hashes_by_split["train"] & hashes_by_split["val"]
    missing_classes = {
        split: sorted(set(SNI21_CLASSES) - set(split_classes[split]))
        for split in ("train", "val")
    }
    min_val_support = min(split_classes["val"].values(), default=0)
    achieved = split_images["val"] / max(1, sum(split_images.values()))
    gates = {
        "zero_parent_overlap": not parent_overlap,
        "zero_exact_hash_overlap": not hash_overlap,
        "all_classes_present": not any(missing_classes.values()),
        "minimum_val_class_support_at_least_10": min_val_support >= 10,
        "val_fraction_within_2_points": abs(achieved - val_fraction) <= 0.02,
        "dataset_audit_safe": bool(dataset_audit["safe_for_training"]),
    }
    summary = {
        "format": "coffee_detector.faruq_grouped_development.v1",
        "status": "complete",
        "repaired_root": str(repaired_root),
        "output_root": str(output_root),
        "seed": seed,
        "target_val_fraction": val_fraction,
        "achieved_val_fraction": achieved,
        "groups": len(groups),
        "images_by_split": dict(split_images),
        "annotations_by_split": dict(split_boxes),
        "annotations_by_split_and_class": {
            split: dict(sorted(split_classes[split].items()))
            for split in ("train", "val")
        },
        "missing_classes_by_split": missing_classes,
        "minimum_val_class_support": min_val_support,
        "cross_split_parent_identities": len(parent_overlap),
        "cross_split_exact_hashes": len(hash_overlap),
        "optimization": optimization,
        "gates": gates,
        "training_ready": all(gates.values()),
        "training_executed": False,
        "inference_executed": False,
        "test_images_accessed": False,
        "test_locked": True,
        "dataset_audit": str(output_root / "dataset_audit.json"),
    }
    write_json(summary, output_root / "faruq_grouped_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-free grouped Faruq train/validation split."
    )
    parser.add_argument("--repaired-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--link-mode", choices=("auto", "hardlink", "copy"), default="auto"
    )
    args = parser.parse_args()
    result = group_faruq_development(
        args.repaired_root,
        args.output_root,
        seed=args.seed,
        val_fraction=args.val_fraction,
        link_mode=args.link_mode,
    )
    print("\n=== FARUQ GROUPED DEVELOPMENT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
