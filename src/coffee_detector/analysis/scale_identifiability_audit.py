from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from coffee_detector.dataset import discover_layout, parse_label

EVENT_PROTOCOL = "faruq-v3-validation-object-events-v1"
CONSENSUS_PROTOCOL = "faruq-v3-cross-model-hard-confusion-consensus-v1"
PROTOCOL = "faruq-v3-scale-identifiability-audit-v1"
SIZE_ORDER = {"kecil": 0, "sedang": 1, "besar": 2}
PRIMARY_FEATURES = ("long_side_norm", "area_norm")
ALL_FEATURES = ("long_side_norm", "area_norm", "long_side_px", "area_px")


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _load_event(path: str | Path, expected_model: str) -> dict:
    payload = _load_json(path)
    if payload.get("protocol") != EVENT_PROTOCOL:
        raise RuntimeError(f"Event protocol tidak kompatibel: {path}")
    if payload.get("model") != expected_model:
        raise RuntimeError(f"Model event salah: expected={expected_model}, got={payload.get('model')}")
    if int(payload.get("seed", -1)) != 42 or payload.get("evaluation_split") != "val":
        raise RuntimeError(f"Event bukan seed42 val: {path}")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError(f"Event menunjukkan akses test: {path}")
    if not isinstance(payload.get("events"), dict) or not payload["events"]:
        raise RuntimeError(f"Event kosong: {path}")
    return payload


def _parse_size_class(name: str) -> tuple[str, str, int] | None:
    value = str(name)
    match = re.match(r"^(.*)_ukuran_(kecil|sedang|besar)$", value)
    if match is None:
        match = re.match(r"^(.*)_(kecil|sedang|besar)$", value)
    if match is None:
        return None
    base, level = match.group(1), match.group(2)
    return base, level, SIZE_ORDER[level]


def _load_size_pairs(path: str | Path) -> list[dict]:
    payload = _load_json(path)
    if payload.get("protocol") != CONSENSUS_PROTOCOL:
        raise RuntimeError("Consensus JSON tidak kompatibel")
    if int(payload.get("seed", -1)) != 42 or payload.get("evaluation_split") != "val":
        raise RuntimeError("Consensus bukan seed42 val")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError("Consensus menunjukkan akses test")

    result = []
    for row in payload.get("undirected_families", []):
        if not row.get("frozen_consensus_hard_family"):
            continue
        family = str(row["family"])
        left, right = family.split(" <-> ", 1)
        a, b = _parse_size_class(left), _parse_size_class(right)
        if a is None or b is None or a[0] != b[0] or a[2] == b[2]:
            continue
        if a[2] < b[2]:
            low_name, low_info, high_name, high_info = left, a, right, b
        else:
            low_name, low_info, high_name, high_info = right, b, left, a
        result.append({
            "family": family,
            "base": low_info[0],
            "low_class": low_name,
            "low_level": low_info[1],
            "high_class": high_name,
            "high_level": high_info[1],
            "consensus_total_support_seed42_six_variants": int(row.get("total_support", 0)),
        })
    if not result:
        raise RuntimeError("Tidak ada frozen hard pair berbasis strata ukuran")
    result.sort(key=lambda r: r["consensus_total_support_seed42_six_variants"], reverse=True)
    return result


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def _expected_auc(low: np.ndarray, high: np.ndarray) -> float:
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if low.size == 0 or high.size == 0:
        return float("nan")
    diff = high[:, None] - low[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def _best_balanced_accuracy(low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if low.size == 0 or high.size == 0:
        return float("nan"), float("nan")
    values = np.unique(np.concatenate([low, high]))
    if values.size == 1:
        return 0.5, float(values[0])
    thresholds = np.concatenate([
        [values[0] - 1e-12],
        (values[:-1] + values[1:]) * 0.5,
        [values[-1] + 1e-12],
    ])
    best_score, best_threshold = -1.0, float(thresholds[0])
    for threshold in thresholds:
        tnr = float((low <= threshold).mean())
        tpr = float((high > threshold).mean())
        score = 0.5 * (tnr + tpr)
        if score > best_score:
            best_score, best_threshold = score, float(threshold)
    return float(best_score), best_threshold


def _iqr_overlap(low_stats: dict, high_stats: dict) -> dict:
    lo = max(float(low_stats["q25"]), float(high_stats["q25"]))
    hi = min(float(low_stats["q75"]), float(high_stats["q75"]))
    union_lo = min(float(low_stats["q25"]), float(high_stats["q25"]))
    union_hi = max(float(low_stats["q75"]), float(high_stats["q75"]))
    width = max(0.0, hi - lo)
    union = max(0.0, union_hi - union_lo)
    return {
        "exists": bool(hi >= lo),
        "low": float(lo),
        "high": float(hi),
        "width": float(width),
        "fraction_of_iqr_union": float(width / union) if union > 0 else 1.0,
    }


def _pair_error(event: dict, family: str) -> bool:
    if not event.get("matched") or event.get("correct"):
        return False
    gt = event.get("gt_class_name")
    pred = event.get("pred_class_name")
    if gt is None or pred is None:
        return False
    return " <-> ".join(sorted((str(gt), str(pred)))) == family


def _event_feature_rows(data_root: Path, cpe: dict, cir: dict) -> dict[str, dict]:
    layout = discover_layout(data_root)
    image_root, label_root = layout.splits["val"]
    image_cache: dict[str, tuple[int, int, list]] = {}
    rows = {}
    for key, cpe_row in cpe["events"].items():
        cir_row = cir["events"][key]
        image_key = str(cpe_row["image"])
        if image_key not in image_cache:
            image_path = image_root / image_key
            image = cv2.imread(str(image_path))
            if image is None:
                raise OSError(f"Gagal baca gambar: {image_path}")
            h, w = image.shape[:2]
            label_path = (label_root / Path(image_key)).with_suffix(".txt")
            annotations = parse_label(label_path, set(layout.names))
            image_cache[image_key] = (h, w, annotations)
        h, w, annotations = image_cache[image_key]
        target_index = int(cpe_row["target_index"])
        ann = annotations[target_index]
        bw_norm = float(ann.width)
        bh_norm = float(ann.height)
        bw_px = bw_norm * w
        bh_px = bh_norm * h
        rows[key] = {
            "target_key": key,
            "gt_class_name": str(cpe_row["gt_class_name"]),
            "long_side_norm": max(bw_norm, bh_norm),
            "area_norm": bw_norm * bh_norm,
            "long_side_px": max(bw_px, bh_px),
            "area_px": bw_px * bh_px,
            "cpe0": cpe_row,
            "cir0": cir_row,
        }
    return rows


def _overlap_error_summary(rows: list[dict], feature: str, band: dict, family: str) -> dict:
    if not band.get("exists"):
        return {
            "all_pair_gt_in_overlap": 0,
            "all_pair_gt_total": len(rows),
            "cpe0_pair_errors_in_overlap": 0,
            "cpe0_pair_errors_total": sum(_pair_error(r["cpe0"], family) for r in rows),
            "cir0_pair_errors_in_overlap": 0,
            "cir0_pair_errors_total": sum(_pair_error(r["cir0"], family) for r in rows),
        }
    lo, hi = float(band["low"]), float(band["high"])
    inside = [r for r in rows if lo <= float(r[feature]) <= hi]
    cpe_errors = [r for r in rows if _pair_error(r["cpe0"], family)]
    cir_errors = [r for r in rows if _pair_error(r["cir0"], family)]
    return {
        "all_pair_gt_in_overlap": len(inside),
        "all_pair_gt_total": len(rows),
        "all_pair_gt_overlap_rate": len(inside) / len(rows) if rows else 0.0,
        "cpe0_pair_errors_in_overlap": sum(lo <= float(r[feature]) <= hi for r in cpe_errors),
        "cpe0_pair_errors_total": len(cpe_errors),
        "cpe0_pair_error_overlap_rate": (
            sum(lo <= float(r[feature]) <= hi for r in cpe_errors) / len(cpe_errors)
            if cpe_errors else 0.0
        ),
        "cir0_pair_errors_in_overlap": sum(lo <= float(r[feature]) <= hi for r in cir_errors),
        "cir0_pair_errors_total": len(cir_errors),
        "cir0_pair_error_overlap_rate": (
            sum(lo <= float(r[feature]) <= hi for r in cir_errors) / len(cir_errors)
            if cir_errors else 0.0
        ),
    }


def _plot_pair(output_root: Path, family: str, low_name: str, high_name: str, rows: list[dict], feature: str) -> str:
    low = np.asarray([r[feature] for r in rows if r["gt_class_name"] == low_name], dtype=np.float64)
    high = np.asarray([r[feature] for r in rows if r["gt_class_name"] == high_name], dtype=np.float64)
    fig = plt.figure(figsize=(8, 5))
    plt.hist(low, bins=12, alpha=0.55, density=True, label=low_name)
    plt.hist(high, bins=12, alpha=0.55, density=True, label=high_name)
    plt.axvline(float(np.median(low)), linestyle="--")
    plt.axvline(float(np.median(high)), linestyle="--")
    plt.xlabel(feature)
    plt.ylabel("density")
    plt.title(f"{family}\nApparent-scale distribution on validation GT")
    plt.legend(fontsize=8)
    plt.tight_layout()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", family).strip("_")
    path = output_root / "plots" / f"{safe}__{feature}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def run(cpe0_event, cir0_event, consensus_json, data_root, output_root) -> dict:
    cpe = _load_event(cpe0_event, "CPE0")
    cir = _load_event(cir0_event, "CIR0")
    if set(cpe["events"]) != set(cir["events"]):
        raise RuntimeError("Target universe CPE0/CIR0 berbeda")
    data_root = Path(data_root).expanduser().resolve()
    layout = discover_layout(data_root)
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit menolak data root yang mengekspos test")
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    size_pairs = _load_size_pairs(consensus_json)
    features = _event_feature_rows(data_root, cpe, cir)
    pair_results = []
    for pair in size_pairs:
        family = pair["family"]
        low_name, high_name = pair["low_class"], pair["high_class"]
        rows = [r for r in features.values() if r["gt_class_name"] in (low_name, high_name)]
        if not rows:
            continue
        feature_results = {}
        plot_paths = {}
        for feature in ALL_FEATURES:
            low = np.asarray([r[feature] for r in rows if r["gt_class_name"] == low_name], dtype=np.float64)
            high = np.asarray([r[feature] for r in rows if r["gt_class_name"] == high_name], dtype=np.float64)
            low_stats, high_stats = _summary(low), _summary(high)
            auc = _expected_auc(low, high)
            best_ba, threshold = _best_balanced_accuracy(low, high)
            overlap = _iqr_overlap(low_stats, high_stats)
            feature_results[feature] = {
                "low_class": low_stats,
                "high_class": high_stats,
                "expected_order_auc": auc,
                "best_single_threshold_balanced_accuracy_expected_direction": best_ba,
                "best_threshold": threshold,
                "median_expected_order_consistent": bool(high_stats["median"] > low_stats["median"]),
                "iqr_overlap": overlap,
                "error_overlap": _overlap_error_summary(rows, feature, overlap, family),
            }
            if feature in PRIMARY_FEATURES:
                plot_paths[feature] = _plot_pair(output_root, family, low_name, high_name, rows, feature)

        pair_results.append({
            **pair,
            "gt_instances_in_pair": len(rows),
            "gt_count_by_class": {
                low_name: sum(r["gt_class_name"] == low_name for r in rows),
                high_name: sum(r["gt_class_name"] == high_name for r in rows),
            },
            "cpe0_pair_errors": sum(_pair_error(r["cpe0"], family) for r in rows),
            "cir0_pair_errors": sum(_pair_error(r["cir0"], family) for r in rows),
            "features": feature_results,
            "plots": plot_paths,
        })

    result = {
        "protocol": PROTOCOL,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "n_frozen_size_hard_pairs": len(pair_results),
        "feature_definitions": {
            "long_side_norm": "max(normalized_bbox_width, normalized_bbox_height)",
            "area_norm": "normalized_bbox_width * normalized_bbox_height",
            "long_side_px": "max(bbox_width_pixels, bbox_height_pixels)",
            "area_px": "bbox_width_pixels * bbox_height_pixels",
        },
        "interpretation_guardrail": (
            "This audit measures apparent object scale in the image, not physical millimetre size. "
            "Overlap therefore supports limited image-scale identifiability under the acquisition protocol, "
            "but cannot by itself prove a wrong label or wrong grading rule. Pixel features are additionally confounded by source resolution."
        ),
        "pairs": pair_results,
        "screening_decision_remains": "STOP_CIRCLE_CPE",
    }
    destination = output_root / "scale_identifiability_audit.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", destination)
    print("FROZEN SIZE HARD PAIRS:", len(pair_results))
    print("SCREENING DECISION REMAINS:", result["screening_decision_remains"])
    for row in pair_results:
        long_result = row["features"]["long_side_norm"]
        area_result = row["features"]["area_norm"]
        print(
            row["family"],
            "n=", row["gt_instances_in_pair"],
            "long_auc=", round(long_result["expected_order_auc"], 4),
            "long_BA=", round(long_result["best_single_threshold_balanced_accuracy_expected_direction"], 4),
            "area_auc=", round(area_result["expected_order_auc"], 4),
            "area_BA=", round(area_result["best_single_threshold_balanced_accuracy_expected_direction"], 4),
            "CPE0err=", row["cpe0_pair_errors"],
            "CIR0err=", row["cir0_pair_errors"],
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpe0-event", required=True)
    parser.add_argument("--cir0-event", required=True)
    parser.add_argument("--consensus-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run(args.cpe0_event, args.cir0_event, args.consensus_json, args.data_root, args.output_root)


if __name__ == "__main__":
    main()
