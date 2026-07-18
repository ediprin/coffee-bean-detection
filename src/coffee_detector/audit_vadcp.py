from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .audit_dataset import audit_dataset
from .vadcp.masks import mask_bbox


def decode_uncompressed_rle(payload: dict) -> np.ndarray:
    height, width = (int(value) for value in payload["size"])
    values = np.empty(height * width, dtype=np.uint8)
    offset = 0
    current = 0
    for raw_count in payload["counts"]:
        count = int(raw_count)
        if count < 0 or offset + count > values.size:
            raise ValueError("RLE count tidak valid")
        values[offset : offset + count] = current
        offset += count
        current = 1 - current
    if offset != values.size:
        raise ValueError(f"RLE size tidak cocok: {offset} != {values.size}")
    return values.reshape((height, width), order="F").astype(bool)


def _same_bbox(left: list | None, right: tuple[int, int, int, int] | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return tuple(int(value) for value in left) == tuple(int(value) for value in right)


def audit_vadcp_dataset(
    data_root: str | Path,
    output: str | Path | None = None,
    *,
    tolerance: float = 1e-8,
) -> dict:
    data_root = Path(data_root).expanduser().resolve()
    metadata_path = data_root / "metadata" / "instances_synthetic_train.json"
    manifest_path = data_root / "metadata" / "generation_manifest.json"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Metadata VA-DCP belum lengkap di {data_root / 'metadata'}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in metadata["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    errors: list[str] = []
    warnings: list[str] = []
    visibility = Counter()
    classes = Counter()
    densities = Counter()
    seen_annotation_ids = set()

    for row in metadata["annotations"]:
        annotation_id = int(row["id"])
        if annotation_id in seen_annotation_ids:
            errors.append(f"annotation id duplikat: {annotation_id}")
        seen_annotation_ids.add(annotation_id)
        image_id = int(row["image_id"])
        if image_id not in images:
            errors.append(f"annotation {annotation_id}: image_id tidak ditemukan")
            continue
        by_image[image_id].append(row)
        try:
            visible = decode_uncompressed_rle(row["segmentation"])
            full = decode_uncompressed_rle(row["full_segmentation"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"annotation {annotation_id}: {error}")
            continue
        if visible.shape != full.shape:
            errors.append(f"annotation {annotation_id}: ukuran visible/full berbeda")
            continue
        if np.any(visible & ~full):
            errors.append(f"annotation {annotation_id}: visible_mask bukan subset full_mask")
        recomputed = float(visible.sum()) / max(int(full.sum()), 1)
        if abs(recomputed - float(row["visibility_ratio"])) > tolerance:
            errors.append(
                f"annotation {annotation_id}: visibility ratio salah "
                f"{row['visibility_ratio']} != {recomputed}"
            )
        if not _same_bbox(row.get("bbox"), mask_bbox(visible)):
            errors.append(f"annotation {annotation_id}: visible bbox tidak konsisten")
        if not _same_bbox(row.get("full_bbox"), mask_bbox(full)):
            errors.append(f"annotation {annotation_id}: full bbox tidak konsisten")
        source_split = str(row.get("source_split", ""))
        if source_split not in {"train", "unspecified"}:
            errors.append(
                f"annotation {annotation_id}: aset {source_split} masuk synthetic train"
            )
        if not int(row.get("ignore", 0)):
            visibility[str(row["visibility_bin"])] += 1
            classes[int(row["category_id"])] += 1

    for image_id, image_row in images.items():
        image_path = data_root / image_row["file_name"]
        if not image_path.is_file():
            errors.append(f"image sintetis tidak ditemukan: {image_path}")
        rows = sorted(by_image.get(image_id, []), key=lambda item: int(item["z_order"]))
        densities[len(rows)] += 1
        z_orders = [int(row["z_order"]) for row in rows]
        if z_orders != list(range(len(rows))):
            errors.append(f"image {image_id}: z_order tidak kontigu {z_orders}")
            continue
        occlusion = None
        for row in reversed(rows):
            full = decode_uncompressed_rle(row["full_segmentation"])
            visible = decode_uncompressed_rle(row["segmentation"])
            if occlusion is None:
                occlusion = np.zeros_like(full, dtype=bool)
            expected = full & ~occlusion
            if not np.array_equal(expected, visible):
                errors.append(
                    f"image {image_id} annotation {row['id']}: visible mask tidak cocok z-order"
                )
            occlusion |= full
        label_path = (
            data_root
            / "train"
            / "labels"
            / Path(image_row["file_name"]).with_suffix(".txt").name
        )
        if not label_path.is_file():
            errors.append(f"label sintetis tidak ditemukan: {label_path}")
        else:
            actual_lines = sum(bool(line.strip()) for line in label_path.read_text(encoding="utf-8").splitlines())
            expected_lines = sum(not int(row.get("ignore", 0)) for row in rows)
            if actual_lines != expected_lines:
                errors.append(
                    f"image {image_id}: label lines {actual_lines} != {expected_lines}"
                )

    target_rates = manifest.get("focus_target_hit_rate", {})
    for name, rate in target_rates.items():
        if rate is not None and float(rate) < 0.90:
            warnings.append(
                f"Target visibility {name} hanya tercapai {float(rate):.1%}; "
                "periksa skala objek atau jumlah placement attempts."
            )
    general_path = data_root / "metadata" / "dataset_audit.json"
    general = audit_dataset(data_root, general_path, near_threshold=-1)
    if not general["safe_for_training"]:
        errors.append("Audit dataset YOLO umum menyatakan dataset belum aman")
    report = {
        "format": "coffee_detector.vadcp_audit.v1",
        "dataset_root": str(data_root),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "synthetic_images": len(images),
        "synthetic_annotations": len(metadata["annotations"]),
        "labeled_instances_by_visibility": dict(sorted(visibility.items())),
        "labeled_instances_by_class_id": {
            str(key): classes[key] for key in sorted(classes)
        },
        "scene_density": {str(key): densities[key] for key in sorted(densities)},
        "focus_target_hit_rate": {
            str(key): value for key, value in sorted(target_rates.items())
        },
        "errors": errors[:200],
        "error_count": len(errors),
        "warnings": warnings,
        "safe_for_training": not errors,
        "general_dataset_audit": str(general_path),
    }
    if output is None:
        output = data_root / "metadata" / "vadcp_audit.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_audit_summary(report: dict, *, label: str | None = None) -> None:
    """Print a compact audit summary for CLIs and notebooks."""
    title = "=== AUDIT VA-DCP"
    if label:
        title += f" — {label}"
    print(title + " ===")
    print(f"Images          : {report['synthetic_images']}")
    print(f"Annotations     : {report['synthetic_annotations']}")
    print(f"Visibility      : {report['labeled_instances_by_visibility']}")
    print(f"Target hit rate : {report.get('focus_target_hit_rate', {})}")
    print(f"Scene density   : {report['scene_density']}")
    print(f"Warnings        : {len(report['warnings'])}")
    print(f"Errors          : {report['error_count']}")
    print(f"AMAN TRAINING   : {'YA' if report['safe_for_training'] else 'BELUM'}")
    if report["warnings"]:
        print("WARNING EXAMPLES")
        for item in report["warnings"][:10]:
            print("-", item)
    if report["errors"]:
        print("ERROR EXAMPLES")
        for item in report["errors"][:20]:
            print("-", item)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit mask, z-order, visibility, leakage, dan label dataset VA-DCP."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_vadcp_dataset(args.data_root, args.output)
    print_audit_summary(report)


if __name__ == "__main__":
    main()
