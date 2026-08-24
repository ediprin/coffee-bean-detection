"""Build a leakage-free, test-only Faruq evaluation package.

The Roboflow archive split is not an evaluation authority: parent identities
cross its train/valid/test folders.  This module opens the original test split
only after model selection is frozen, removes every parent/hash seen by the
grouped development manifest, keeps one deterministic representative per
remaining parent, and applies the already-frozen polygon/EXIF geometry rule.
It never trains or runs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .audit_faruq_mask_geometry import (
    _alignment_score,
    _candidate_images,
    _current_transform,
    _mask_from_polygons,
    _polygons,
)
from .dataset import write_json
from .prepare_sni_fullscene import (
    FARUQ_CLASS_MAP,
    SNI21_CLASSES,
    SNI21_IDS,
    canonical_source_identity,
)
from .repair_faruq_mask_geometry import _polygon_bbox, _transform_image, _write_yolo


FROZEN_SCORE_LONG_SIDE = 192
FROZEN_MIN_IMPROVEMENT = 0.02
FROZEN_MIN_IMAGES = 50
FROZEN_MIN_INSTANCES_PER_CLASS = 10
FROZEN_MIN_PARENTS_PER_CLASS = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _test_annotation_file(raw_root: Path) -> Path:
    test_root = raw_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"Split test Faruq tidak ditemukan: {test_root}")
    candidates = sorted(
        {
            path.resolve()
            for pattern in ("*.json", "*.coco.json")
            for path in test_root.glob(pattern)
            if path.is_file()
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Diharapkan satu COCO JSON test pada {test_root}, ditemukan {len(candidates)}"
        )
    return candidates[0]


def _development_identities(manifest_path: Path) -> tuple[set[str], set[str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest Faruq-v3 tidak ditemukan: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Manifest Faruq-v3 kosong atau bukan list")
    parents, hashes = set(), set()
    for row in payload:
        if str(row.get("output_split")) not in {"train", "val"}:
            raise RuntimeError("Manifest development memuat split selain train/val")
        parents.add(str(row["source_parent_id"]))
        hashes.add(str(row["source_sha256"]))
    return parents, hashes


def _geometry_selection(
    source: Path,
    expected_size: tuple[int, int],
    polygons: list[list[float]],
    *,
    score_long_side: int,
    min_improvement: float,
) -> dict:
    scale = score_long_side / max(expected_size)
    score_size = (
        max(16, int(round(expected_size[0] * scale))),
        max(16, int(round(expected_size[1] * scale))),
    )
    mask = _mask_from_polygons(polygons, expected_size, score_size)
    with Image.open(source) as opened:
        raw_size = opened.size
        candidates = _candidate_images(opened, expected_size)
    scores = {name: _alignment_score(image, mask) for name, image in candidates.items()}
    for image in candidates.values():
        image.close()
    best = max(scores, key=scores.get)
    current = _current_transform(raw_size, expected_size)
    current_score = scores.get(current, float("-inf"))
    improvement = float(scores[best] - current_score)
    flagged = (
        best != current
        and np.isfinite(improvement)
        and improvement >= min_improvement
    )
    selected = best if flagged else current
    if selected not in scores:
        raise ValueError(
            f"Tidak ada transformasi geometri valid untuk {source}: current={current}"
        )
    return {
        "raw_size": list(raw_size),
        "current_transform": current,
        "best_transform": best,
        "selected_transform": selected,
        "best_improvement": improvement,
        "flagged_orientation": flagged,
    }


def prepare_faruq_locked_test(
    raw_root: str | Path,
    development_manifest: str | Path,
    output_root: str | Path,
    *,
    score_long_side: int = FROZEN_SCORE_LONG_SIDE,
    min_improvement: float = FROZEN_MIN_IMPROVEMENT,
    minimum_images: int = FROZEN_MIN_IMAGES,
    minimum_instances_per_class: int = FROZEN_MIN_INSTANCES_PER_CLASS,
    minimum_parents_per_class: int = FROZEN_MIN_PARENTS_PER_CLASS,
) -> dict:
    """Open and decontaminate the frozen raw test split without inference."""

    raw_root = Path(raw_root).expanduser().resolve()
    development_manifest = Path(development_manifest).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    summary_path = output_root / "faruq_locked_test_eligibility.json"
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if cached.get("status") == "complete":
            print(f"REUSE FARUQ LOCKED TEST AUDIT: {output_root}", flush=True)
            return cached
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output locked test parsial/tidak valid: {output_root}")

    annotation_path = _test_annotation_file(raw_root)
    development_parents, development_hashes = _development_identities(
        development_manifest
    )
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        int(item["id"]): str(item["name"])
        for item in payload.get("categories", [])
    }
    by_image: dict[int | str, list[dict]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        by_image[annotation["image_id"]].append(annotation)

    candidates = []
    excluded = []
    counters = Counter()
    for image_record in payload.get("images", []):
        source = annotation_path.parent / str(image_record["file_name"])
        if not source.is_file():
            excluded.append({"file_name": str(image_record["file_name"]), "reason": "missing_image"})
            counters["excluded_missing_image"] += 1
            continue
        parent_id = canonical_source_identity(str(image_record["file_name"]))
        digest = _sha256(source)
        overlap_reasons = []
        if parent_id in development_parents:
            overlap_reasons.append("development_parent_overlap")
        if digest in development_hashes:
            overlap_reasons.append("development_hash_overlap")
        if overlap_reasons:
            excluded.append(
                {
                    "file_name": str(image_record["file_name"]),
                    "parent_id": parent_id,
                    "sha256": digest,
                    "reason": "+".join(overlap_reasons),
                }
            )
            counters["excluded_development_overlap"] += 1
            continue
        candidates.append(
            {
                "record": image_record,
                "source": source,
                "parent_id": parent_id,
                "sha256": digest,
            }
        )

    # Roboflow test contains augmented siblings.  Keep exactly one stable
    # representative per independent parent so AP is not pseudo-replicated.
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        by_parent[item["parent_id"]].append(item)
    selected = []
    for parent_id, siblings in sorted(by_parent.items()):
        ordered = sorted(siblings, key=lambda item: str(item["record"]["file_name"]).lower())
        selected.append(ordered[0])
        for item in ordered[1:]:
            excluded.append(
                {
                    "file_name": str(item["record"]["file_name"]),
                    "parent_id": parent_id,
                    "sha256": item["sha256"],
                    "reason": "test_parent_pseudoreplicate",
                }
            )
            counters["excluded_test_parent_pseudoreplicate"] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    test_images = []
    test_annotations = []
    manifest = []
    quarantine = []
    class_instances = Counter()
    class_parents: dict[str, set[str]] = defaultdict(set)
    seen_hashes = set()
    next_annotation_id = 1
    for ordinal, item in enumerate(selected, 1):
        image_record = item["record"]
        source = item["source"]
        parent_id = item["parent_id"]
        if item["sha256"] in seen_hashes:
            quarantine.append(
                {"file_name": str(image_record["file_name"]), "reason": "duplicate_test_hash"}
            )
            counters["quarantined_duplicate_test_hash"] += 1
            continue
        seen_hashes.add(item["sha256"])
        width, height = int(image_record["width"]), int(image_record["height"])
        converted = []
        reasons = []
        raw_annotations = by_image.get(image_record["id"], [])
        polygons = _polygons(raw_annotations)
        for annotation in raw_annotations:
            source_name = categories.get(int(annotation["category_id"]))
            canonical_name = FARUQ_CLASS_MAP.get(str(source_name))
            if canonical_name is None:
                reasons.append(f"unmapped_category:{source_name}")
                continue
            geometry = _polygon_bbox(_polygons([annotation]), width, height)
            if geometry is None:
                reasons.append(f"invalid_polygon:{annotation.get('id')}")
                continue
            cleaned_polygons, bbox = geometry
            class_id = SNI21_IDS[canonical_name]
            converted.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_record["id"],
                    "category_id": class_id,
                    "segmentation": cleaned_polygons,
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                }
            )
            next_annotation_id += 1
        if not polygons or not converted:
            reasons.append("no_valid_annotations")
        try:
            geometry = _geometry_selection(
                source,
                (width, height),
                polygons,
                score_long_side=score_long_side,
                min_improvement=min_improvement,
            )
        except (OSError, ValueError) as error:
            reasons.append(f"geometry:{type(error).__name__}")
            geometry = None
        if reasons or geometry is None:
            quarantine.append(
                {
                    "file_name": str(image_record["file_name"]),
                    "parent_id": parent_id,
                    "reasons": sorted(set(reasons)),
                }
            )
            counters["quarantined_invalid"] += 1
            continue

        relative_name = Path(str(image_record["file_name"])).name
        image_target = output_root / "test" / "images" / relative_name
        label_target = output_root / "test" / "labels" / Path(relative_name).with_suffix(".txt")
        _transform_image(
            source,
            image_target,
            str(geometry["selected_transform"]),
            (width, height),
        )
        _write_yolo(label_target, converted, width, height)
        for annotation in converted:
            class_name = SNI21_CLASSES[int(annotation["category_id"])]
            class_instances[class_name] += 1
            class_parents[class_name].add(parent_id)
        test_images.append(
            {
                "id": image_record["id"],
                "file_name": f"images/{relative_name}",
                "width": width,
                "height": height,
            }
        )
        test_annotations.extend(converted)
        manifest.append(
            {
                "source_image": str(source),
                "output_image": str(image_target),
                "output_label": str(label_target),
                "source_parent_id": parent_id,
                "source_sha256": item["sha256"],
                "annotations": len(converted),
                **geometry,
            }
        )
        counters["images_written"] += 1
        counters["annotations_written"] += len(converted)
        counters["geometry_override_used"] += int(geometry["flagged_orientation"])
        if ordinal % 25 == 0 or ordinal == len(selected):
            print(
                f"LOCKED TEST {ordinal}/{len(selected)} | written={counters['images_written']}",
                flush=True,
            )

    names = {index: name for index, name in enumerate(SNI21_CLASSES)}
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output_root),
                "val": "test/images",
                "test": "test/images",
                "names": names,
                "locked_test_only": True,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    coco = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": [{"id": index, "name": name} for index, name in enumerate(SNI21_CLASSES)],
    }
    (output_root / "test" / "_annotations.coco.json").write_text(
        json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_json(manifest, output_root / "faruq_locked_test_manifest.json")
    write_json(excluded, output_root / "faruq_locked_test_excluded.json")
    write_json(quarantine, output_root / "faruq_locked_test_quarantine.json")

    instance_counts = {name: int(class_instances[name]) for name in SNI21_CLASSES}
    parent_counts = {name: len(class_parents[name]) for name in SNI21_CLASSES}
    missing_classes = [name for name in SNI21_CLASSES if instance_counts[name] == 0]
    manifest_parents = {row["source_parent_id"] for row in manifest}
    manifest_hashes = {row["source_sha256"] for row in manifest}
    gates = {
        "zero_development_parent_overlap": not (manifest_parents & development_parents),
        "zero_development_hash_overlap": not (manifest_hashes & development_hashes),
        "one_image_per_test_parent": len(manifest_parents) == len(manifest),
        "minimum_independent_images": len(manifest) >= minimum_images,
        "all_21_classes_present": not missing_classes,
        "minimum_instances_per_class": min(instance_counts.values(), default=0)
        >= minimum_instances_per_class,
        "minimum_parents_per_class": min(parent_counts.values(), default=0)
        >= minimum_parents_per_class,
        "zero_quarantined_selected_images": not quarantine,
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    summary = {
        "format": "coffee_detector.faruq_locked_test_eligibility.v1",
        "status": "complete",
        "decision": decision,
        "next_action": (
            "AUTHORIZE_FROZEN_ACMC_TEST_INFERENCE"
            if decision == "PASS"
            else "STOP_TEST_INFERENCE_USE_GROUPED_CV_OR_EXTERNAL_TEST"
        ),
        "raw_test_annotation": str(annotation_path),
        "development_manifest": str(development_manifest),
        "output_root": str(output_root),
        "test_images_accessed": True,
        "test_annotations_accessed": True,
        "training_executed": False,
        "inference_executed": False,
        "raw_test_images": len(payload.get("images", [])),
        "eligible_parent_candidates": len(by_parent),
        "materialized_images": len(manifest),
        "materialized_annotations": len(test_annotations),
        "excluded_images": len(excluded),
        "quarantined_images": len(quarantine),
        "instances_by_class": instance_counts,
        "parents_by_class": parent_counts,
        "missing_classes": missing_classes,
        "thresholds": {
            "score_long_side": score_long_side,
            "min_geometry_improvement": min_improvement,
            "minimum_images": minimum_images,
            "minimum_instances_per_class": minimum_instances_per_class,
            "minimum_parents_per_class": minimum_parents_per_class,
        },
        "gates": gates,
        "manifest": str(output_root / "faruq_locked_test_manifest.json"),
    }
    write_json(summary, summary_path)
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a parent/hash-disjoint Faruq locked test package"
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = prepare_faruq_locked_test(
        args.raw_root, args.development_manifest, args.output_root
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
