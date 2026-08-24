from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout, parse_label


SIZE_FAMILIES = {
    "kulit_kopi": {
        "small": "kulit_kopi_ukuran_kecil",
        "medium": "kulit_kopi_ukuran_sedang",
        "large": "kulit_kopi_ukuran_besar",
    },
    "kulit_tanduk": {
        "small": "kulit_tanduk_ukuran_kecil",
        "medium": "kulit_tanduk_ukuran_sedang",
        "large": "kulit_tanduk_ukuran_besar",
    },
    "tanah_batu_ranting": {
        "small": "tanah_batu_ranting_kecil",
        "medium": "tanah_batu_ranting_sedang",
        "large": "tanah_batu_ranting_besar",
    },
}

LOCAL_DEFECT_CLASSES = {
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
}

ORDER = ("small", "medium", "large")
FEATURES = ("normalized_area", "normalized_long_side", "relative_area")


def _auc_for_order(lower: list[float], upper: list[float]) -> float | None:
    """Probability that a sample from the upper size exceeds the lower size."""

    if not lower or not upper:
        return None
    left = np.asarray(lower, dtype=np.float64)[:, None]
    right = np.asarray(upper, dtype=np.float64)[None, :]
    comparisons = left.size * right.size
    return float(
        ((right > left).sum() + 0.5 * (right == left).sum()) / comparisons
    )


def _family_for_class(class_name: str) -> str | None:
    for family, levels in SIZE_FAMILIES.items():
        if class_name in levels.values():
            return family
    return None


def _confusion_kind(expected: str, predicted: str) -> str:
    expected_family = _family_for_class(expected)
    predicted_family = _family_for_class(predicted)
    if expected_family is not None and expected_family == predicted_family:
        return "within_family_size"
    if expected in LOCAL_DEFECT_CLASSES and predicted in LOCAL_DEFECT_CLASSES:
        return "local_defect_similarity"
    return "cross_family_or_material"


def _collect_split(layout, split: str) -> list[dict]:
    image_root, label_root = layout.splits[split]
    valid_ids = set(layout.names)
    rows: list[dict] = []
    image_paths = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    for index, image_path in enumerate(image_paths, 1):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        boxes = parse_label(label_path, valid_ids)
        if not boxes:
            continue
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        areas = np.asarray([box.width * box.height for box in boxes], dtype=np.float64)
        median_area = float(np.median(areas))
        for box, area in zip(boxes, areas):
            rows.append(
                {
                    "class_name": layout.names[box.class_id],
                    "normalized_area": float(area),
                    "normalized_long_side": float(max(box.width, box.height)),
                    "relative_area": float(area / median_area),
                    "pixel_area": float(
                        box.width * image_width * box.height * image_height
                    ),
                    "scene_objects": len(boxes),
                }
            )
        if index % 500 == 0 or index == len(image_paths):
            print(f"IDENTIFIABILITY {split}: {index}/{len(image_paths)}", flush=True)
    return rows


def _size_report(rows: list[dict]) -> dict:
    output = {}
    for family, level_names in SIZE_FAMILIES.items():
        by_level = {
            level: [row for row in rows if row["class_name"] == class_name]
            for level, class_name in level_names.items()
        }
        feature_reports = {}
        for feature in FEATURES:
            values = {
                level: [float(row[feature]) for row in by_level[level]]
                for level in ORDER
            }
            medians = {
                level: (float(np.median(values[level])) if values[level] else None)
                for level in ORDER
            }
            pairs = {}
            for lower, upper in (("small", "medium"), ("medium", "large"), ("small", "large")):
                pairs[f"{lower}_lt_{upper}"] = _auc_for_order(
                    values[lower], values[upper]
                )
            available = [value for value in pairs.values() if value is not None]
            macro_auc = float(np.mean(available)) if available else None
            ordered = (
                all(medians[level] is not None for level in ORDER)
                and medians["small"] < medians["medium"] < medians["large"]
            )
            feature_reports[feature] = {
                "medians": medians,
                "pairwise_order_auc": pairs,
                "macro_order_auc": macro_auc,
                "strict_median_order": bool(ordered),
            }
        eligible = [
            (name, report)
            for name, report in feature_reports.items()
            if report["macro_order_auc"] is not None
        ]
        best_name, best_report = max(
            eligible,
            key=lambda item: item[1]["macro_order_auc"],
            default=(None, {"macro_order_auc": None, "strict_median_order": False}),
        )
        best_auc = best_report["macro_order_auc"]
        if best_auc is None:
            signal = "insufficient_samples"
        elif best_auc >= 0.80 and best_report["strict_median_order"]:
            signal = "strong"
        elif best_auc >= 0.65:
            signal = "partial"
        else:
            signal = "weak"
        output[family] = {
            "classes": level_names,
            "samples": {level: len(by_level[level]) for level in ORDER},
            "features": feature_reports,
            "best_feature": best_name,
            "best_macro_order_auc": best_auc,
            "geometry_signal": signal,
        }
    return output


def _confusion_report(path: Path | None) -> dict:
    if path is None:
        return {"source": None, "rows": [], "counts_by_kind": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    counts = Counter()
    for raw in payload.get("top_directional_confusions", []):
        row = dict(raw)
        row["kind"] = _confusion_kind(row["expected"], row["predicted"])
        counts[row["kind"]] += int(row["count"])
        rows.append(row)
    return {
        "source": str(path),
        "rows": rows,
        "counts_by_kind": dict(counts),
    }


def audit_faruq_v3_label_identifiability(
    data_root: str | Path,
    output: str | Path,
    *,
    diagnostic: str | Path | None = None,
) -> dict:
    """Audit whether SNI size labels are observable before changing the model."""

    layout = discover_layout(data_root)
    if "val" not in layout.splits:
        raise FileNotFoundError("Validation split tidak ditemukan")
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit development tidak boleh menyediakan test")
    diagnostic_path = (
        Path(diagnostic).expanduser().resolve() if diagnostic is not None else None
    )
    if diagnostic_path is not None and not diagnostic_path.is_file():
        raise FileNotFoundError(f"Diagnostic D0 tidak ditemukan: {diagnostic_path}")

    split_rows = {
        split: _collect_split(layout, split)
        for split in ("train", "val")
        if split in layout.splits
    }
    split_reports = {}
    for split, rows in split_rows.items():
        scenes = Counter(int(row["scene_objects"]) for row in rows)
        split_reports[split] = {
            "instances": len(rows),
            "single_object_instance_fraction": float(
                sum(count for objects, count in scenes.items() if objects == 1) / len(rows)
            )
            if rows
            else 0.0,
            "size_families": _size_report(rows),
        }

    val_families = split_reports["val"]["size_families"]
    signals = [family["geometry_signal"] for family in val_families.values()]
    if signals and all(signal == "strong" for signal in signals):
        decision = "GEOMETRY_HEAD_JUSTIFIED"
        next_action = "freeze_geometry_conditioned_classification_protocol"
    elif any(signal in {"weak", "insufficient_samples"} for signal in signals):
        decision = "DATA_OR_SCALE_LIMITED"
        next_action = "repair_or_calibrate_size_labels_before_model_change"
    else:
        decision = "PARTIAL_SIGNAL"
        next_action = "inspect_size_contact_sheets_and_source_scale_before_model_change"

    payload = {
        "protocol": "faruq-v3-label-identifiability-v1",
        "training_executed": False,
        "evaluation_splits": ["train", "val"],
        "test_images_accessed": False,
        "dataset_root": str(layout.root),
        "split_reports": split_reports,
        "confusion_taxonomy": _confusion_report(diagnostic_path),
        "decision": decision,
        "next_action": next_action,
        "thresholds": {
            "strong_macro_order_auc": 0.80,
            "partial_macro_order_auc": 0.65,
            "requires_strict_median_order_for_strong": True,
        },
        "limitations": [
            "Bounding-box geometry is an image-space proxy, not a physical millimetre measurement.",
            "Development train/validation evidence cannot replace an independent test.",
            "A weak signal can reflect inconsistent camera scale, ambiguous labels, or both.",
        ],
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faruq-v3 validation-safe SNI label identifiability audit."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostic")
    args = parser.parse_args()
    result = audit_faruq_v3_label_identifiability(
        args.data_root,
        args.output,
        diagnostic=args.diagnostic,
    )
    print(json.dumps({
        "decision": result["decision"],
        "next_action": result["next_action"],
        "val_size_families": result["split_reports"]["val"]["size_families"],
        "confusion_counts": result["confusion_taxonomy"]["counts_by_kind"],
        "training_executed": result["training_executed"],
        "test_images_accessed": result["test_images_accessed"],
    }, indent=2, ensure_ascii=False))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
