from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from .audit_faruq_mask_geometry import TRANSFORMS, _annotation_files, _polygons
from .dataset import write_json
from .prepare_sni_fullscene import (
    FARUQ_CLASS_MAP,
    SNI21_CLASSES,
    SNI21_IDS,
    canonical_source_identity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_index(records_path: Path) -> dict[tuple[str, str], dict]:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    index = {}
    for record in records:
        key = (str(record["split"]), str(record["image_id"]))
        if key in index:
            raise RuntimeError(f"Record audit ganda: {key}")
        index[key] = record
    return index


def _selected_transform(record: dict) -> str:
    """Use the audited best transform only when it cleared the frozen margin."""

    selected = (
        record["best_transform"]
        if record["flagged_orientation"]
        else record["current_transform"]
    )
    if selected not in TRANSFORMS:
        raise ValueError(f"Transformasi audit tidak dikenal: {selected}")
    return selected


def _transform_image(
    source: Path, target: Path, transform_name: str, expected_size: tuple[int, int]
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        rgb = opened.convert("RGB")
        transform = TRANSFORMS[transform_name]
        output = rgb if transform is None else rgb.transpose(transform)
        if output.size != expected_size:
            raise ValueError(
                f"Transformasi {transform_name} tidak menghasilkan ukuran COCO: "
                f"source={source}, output={output.size}, expected={expected_size}"
            )
        save_kwargs = {"quality": 95, "subsampling": 0} if target.suffix.lower() in {".jpg", ".jpeg"} else {}
        output.save(target, **save_kwargs)
        if output is not rgb:
            output.close()
        rgb.close()


def _polygon_bbox(
    polygons: list[list[float]], width: int, height: int
) -> tuple[list[list[float]], list[float]] | None:
    cleaned = []
    xs = []
    ys = []
    for polygon in polygons:
        points = []
        for index in range(0, len(polygon), 2):
            x = min(float(width), max(0.0, float(polygon[index])))
            y = min(float(height), max(0.0, float(polygon[index + 1])))
            points.extend((x, y))
            xs.append(x)
            ys.append(y)
        if len(points) >= 6:
            cleaned.append(points)
    if not cleaned:
        return None
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return cleaned, [x1, y1, x2 - x1, y2 - y1]


def _write_yolo(path: Path, annotations: list[dict], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for annotation in sorted(annotations, key=lambda item: (item["category_id"], item["id"])):
        x, y, box_width, box_height = annotation["bbox"]
        lines.append(
            f"{annotation['category_id']} "
            f"{(x + box_width / 2) / width:.8f} "
            f"{(y + box_height / 2) / height:.8f} "
            f"{box_width / width:.8f} {box_height / height:.8f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def repair_faruq_mask_geometry(
    raw_root: str | Path,
    geometry_records: str | Path,
    output_root: str | Path,
) -> dict:
    """Materialize corrected Faruq train/validation without reading test.

    The COCO polygons already live in the declared COCO coordinate frame. Only
    the image pixels are transformed; boxes are recomputed from polygons rather
    than copied from the earlier, contaminated bounding-box conversion.
    """

    raw_root = Path(raw_root).expanduser().resolve()
    geometry_records = Path(geometry_records).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not geometry_records.is_file():
        raise FileNotFoundError(f"Record audit tidak ditemukan: {geometry_records}")
    if output_root.exists() and any(output_root.iterdir()):
        summary_path = output_root / "faruq_geometry_repair_summary.json"
        if summary_path.is_file():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if cached.get("status") == "complete":
                print(f"REUSE FARUQ REPAIR: {output_root}", flush=True)
                return cached
        raise FileExistsError(f"Output repair tidak kosong/complete: {output_root}")

    audit_records = _audit_index(geometry_records)
    annotation_files = _annotation_files(raw_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    quarantine = []
    counters = Counter()
    split_class_counts: dict[str, Counter] = defaultdict(Counter)
    identities_by_split: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)

    for split, annotation_path in annotation_files:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {
            int(item["id"]): str(item["name"])
            for item in payload.get("categories", [])
        }
        by_image: dict[int | str, list[dict]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            by_image[annotation["image_id"]].append(annotation)
        output_images = []
        output_annotations = []

        for ordinal, image_record in enumerate(payload.get("images", []), 1):
            image_id = str(image_record["id"])
            source = annotation_path.parent / str(image_record["file_name"])
            key = (split, image_id)
            audit = audit_records.get(key)
            reasons = []
            if not source.is_file():
                reasons.append("missing_image")
            if audit is None:
                reasons.append("missing_geometry_audit")
            width = int(image_record["width"])
            height = int(image_record["height"])
            converted = []
            for annotation in by_image.get(image_record["id"], []):
                source_name = categories.get(int(annotation["category_id"]))
                canonical_name = FARUQ_CLASS_MAP.get(str(source_name))
                if canonical_name is None:
                    reasons.append(f"unmapped_category:{source_name}")
                    continue
                geometry = _polygon_bbox(_polygons([annotation]), width, height)
                if geometry is None:
                    reasons.append(f"invalid_polygon:{annotation.get('id')}")
                    continue
                polygons, bbox = geometry
                class_id = SNI21_IDS[canonical_name]
                converted.append(
                    {
                        "id": int(annotation["id"]),
                        "image_id": image_record["id"],
                        "category_id": class_id,
                        "segmentation": polygons,
                        "bbox": bbox,
                        "area": float(bbox[2] * bbox[3]),
                        "iscrowd": 0,
                    }
                )
            if not converted:
                reasons.append("no_valid_annotations")
            if reasons:
                quarantine.append(
                    {
                        "split": split,
                        "image_id": image_id,
                        "image": str(source),
                        "reasons": sorted(set(reasons)),
                    }
                )
                counters["quarantined_images"] += 1
                continue

            transform_name = _selected_transform(audit)
            relative_name = Path(str(image_record["file_name"])).name
            image_target = output_root / split / "images" / relative_name
            label_target = image_target.with_suffix(".txt").parent.parent / "labels" / image_target.with_suffix(".txt").name
            _transform_image(source, image_target, transform_name, (width, height))
            _write_yolo(label_target, converted, width, height)
            split_class_counts[split].update(
                SNI21_CLASSES[annotation["category_id"]]
                for annotation in converted
            )
            parent_id = canonical_source_identity(relative_name)
            digest = _sha256(source)
            identities_by_split[split].add(parent_id)
            hashes_by_split[split].add(digest)
            output_images.append(
                {
                    "id": image_record["id"],
                    "file_name": f"images/{relative_name}",
                    "width": width,
                    "height": height,
                }
            )
            output_annotations.extend(converted)
            counters["images_written"] += 1
            counters["annotations_written"] += len(converted)
            counters[f"transform:{transform_name}"] += 1
            counters["audit_override_used"] += int(bool(audit["flagged_orientation"]))
            manifest.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "source_image": str(source),
                    "output_image": str(image_target),
                    "output_label": str(label_target),
                    "source_parent_id": parent_id,
                    "source_sha256": digest,
                    "current_transform": audit["current_transform"],
                    "best_transform": audit["best_transform"],
                    "flagged_orientation": bool(audit["flagged_orientation"]),
                    "selected_transform": transform_name,
                    "annotations": len(converted),
                }
            )
            if ordinal % 250 == 0 or ordinal == len(payload.get("images", [])):
                print(
                    f"REPAIR {split}: {ordinal}/{len(payload.get('images', []))} | "
                    f"written={counters['images_written']}",
                    flush=True,
                )

        coco_payload = {
            "images": output_images,
            "annotations": output_annotations,
            "categories": [
                {"id": index, "name": name}
                for index, name in enumerate(SNI21_CLASSES)
            ],
        }
        (output_root / split / "_annotations.coco.json").write_text(
            json.dumps(coco_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    parent_overlap = sorted(identities_by_split["train"] & identities_by_split["val"])
    hash_overlap = sorted(hashes_by_split["train"] & hashes_by_split["val"])
    data_yaml = {
        "path": str(output_root),
        "train": "train/images",
        "val": "val/images",
        "names": {index: name for index, name in enumerate(SNI21_CLASSES)},
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(manifest, output_root / "faruq_geometry_repair_manifest.json")
    write_json(quarantine, output_root / "faruq_geometry_repair_quarantine.json")
    summary = {
        "format": "coffee_detector.faruq_mask_geometry_repair.v1",
        "status": "complete",
        "raw_root": str(raw_root),
        "geometry_records": str(geometry_records),
        "output_root": str(output_root),
        "splits_accessed": sorted({split for split, _ in annotation_files}),
        "test_images_accessed": False,
        "training_executed": False,
        "inference_executed": False,
        "counters": dict(counters),
        "annotations_by_split_and_class": {
            split: dict(sorted(split_class_counts[split].items()))
            for split in ("train", "val")
        },
        "cross_split_parent_identities": len(parent_overlap),
        "cross_split_exact_hashes": len(hash_overlap),
        "parent_overlap_examples": parent_overlap[:20],
        "quarantined_images": len(quarantine),
        "geometry_materialized": True,
        "training_ready": False,
        "test_locked": True,
        "next_action": (
            "Run the mask-geometry audit on this repaired output and inspect a "
            "fresh contact sheet. Split leakage must also be resolved before training."
        ),
    }
    summary_path = output_root / "faruq_geometry_repair_summary.json"
    write_json(summary, summary_path)
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair Faruq train/validation image orientation from frozen mask audit."
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--geometry-records", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = repair_faruq_mask_geometry(
        args.raw_root, args.geometry_records, args.output_root
    )
    print("\n=== FARUQ MASK-GEOMETRY REPAIR ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
