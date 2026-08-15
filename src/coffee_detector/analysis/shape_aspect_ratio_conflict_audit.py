from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from coffee_detector.dataset import discover_layout, parse_label

EVENT_PROTOCOL = "faruq-v3-validation-object-events-v1"
SCALE_PROTOCOL = "faruq-v3-scale-identifiability-audit-v1"
PROTOCOL = "faruq-v3-shape-aspect-ratio-conflict-audit-v1"


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


def _load_scale(path: str | Path) -> dict:
    payload = _load_json(path)
    if payload.get("protocol") != SCALE_PROTOCOL:
        raise RuntimeError("Scale audit protocol tidak kompatibel")
    if int(payload.get("seed", -1)) != 42 or payload.get("evaluation_split") != "val":
        raise RuntimeError("Scale audit bukan seed42 val")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError("Scale audit menunjukkan akses test")
    if not payload.get("pairs"):
        raise RuntimeError("Scale audit tidak memiliki pair")
    return payload


def _summary(values: list[float] | np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(x.min()),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
    }


def _auc_greater(group_a: list[float], group_b: list[float]) -> float:
    """P(random A > random B) + 0.5 ties; 0.5 means no rank shift."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def _shape_extremeness(aspect_ratio: float, median: float, iqr: float) -> float:
    denom = max(float(iqr), 1e-12)
    return abs(float(aspect_ratio) - float(median)) / denom


def _predict_size(value: float, threshold: float, low_class: str, high_class: str) -> str:
    return high_class if float(value) > float(threshold) else low_class


def _geometry_state(row: dict, pair: dict) -> str:
    low_name = str(pair["low_class"])
    high_name = str(pair["high_class"])
    gt = str(row["gt_class_name"])
    long_t = float(pair["features"]["long_side_norm"]["best_threshold"])
    area_t = float(pair["features"]["area_norm"]["best_threshold"])
    long_pred = _predict_size(row["long_side_norm"], long_t, low_name, high_name)
    area_pred = _predict_size(row["area_norm"], area_t, low_name, high_name)
    if long_pred == gt and area_pred == gt:
        return "both_support_gt"
    if long_pred != gt and area_pred != gt:
        return "both_support_other"
    return "mixed_geometry"


def _pair_error(event: dict, family: str) -> bool:
    if not event.get("matched") or event.get("correct"):
        return False
    gt = event.get("gt_class_name")
    pred = event.get("pred_class_name")
    if gt is None or pred is None:
        return False
    return " <-> ".join(sorted((str(gt), str(pred)))) == family


def _event_rows(data_root: Path, cpe: dict, cir: dict) -> dict[str, dict]:
    layout = discover_layout(data_root)
    image_root, label_root = layout.splits["val"]
    cache: dict[str, tuple[int, int, list]] = {}
    rows: dict[str, dict] = {}
    for key, cpe_row in cpe["events"].items():
        cir_row = cir["events"][key]
        image_key = str(cpe_row["image"])
        if image_key not in cache:
            image_path = image_root / image_key
            image = cv2.imread(str(image_path))
            if image is None:
                raise OSError(f"Gagal baca gambar: {image_path}")
            h, w = image.shape[:2]
            label_path = (label_root / Path(image_key)).with_suffix(".txt")
            annotations = parse_label(label_path, set(layout.names))
            cache[image_key] = (h, w, annotations)
        _, _, annotations = cache[image_key]
        idx = int(cpe_row["target_index"])
        ann = annotations[idx]
        bw = max(float(ann.width), 1e-12)
        bh = max(float(ann.height), 1e-12)
        long_side = max(bw, bh)
        short_side = min(bw, bh)
        rows[key] = {
            "target_key": key,
            "gt_class_name": str(cpe_row["gt_class_name"]),
            "width_norm": bw,
            "height_norm": bh,
            "long_side_norm": long_side,
            "short_side_norm": short_side,
            "area_norm": bw * bh,
            "aspect_ratio": long_side / short_side,
            "cpe0": cpe_row,
            "cir0": cir_row,
        }
    return rows


def _state_stats(rows: list[dict]) -> dict:
    return {
        "objects": len(rows),
        "aspect_ratio": _summary([r["aspect_ratio"] for r in rows]),
        "short_side_norm": _summary([r["short_side_norm"] for r in rows]),
        "extreme_shape_objects": sum(bool(r["extreme_shape_gt_1_iqr"]) for r in rows),
        "extreme_shape_rate": (
            sum(bool(r["extreme_shape_gt_1_iqr"]) for r in rows) / len(rows) if rows else 0.0
        ),
    }


def _model_stats(rows: list[dict], model_key: str, family: str) -> dict:
    errors = [r for r in rows if _pair_error(r[model_key], family)]
    state_counts = Counter(r["geometry_state"] for r in errors)
    extreme_errors = [r for r in errors if r["extreme_shape_gt_1_iqr"]]
    extreme_objects = [r for r in rows if r["extreme_shape_gt_1_iqr"]]
    non_extreme = [r for r in rows if not r["extreme_shape_gt_1_iqr"]]
    non_extreme_errors = [r for r in errors if not r["extreme_shape_gt_1_iqr"]]
    return {
        "pair_errors_total": len(errors),
        "pair_errors_by_geometry_state": dict(state_counts),
        "extreme_shape": {
            "objects": len(extreme_objects),
            "errors": len(extreme_errors),
            "error_rate": len(extreme_errors) / len(extreme_objects) if extreme_objects else 0.0,
            "share_of_pair_errors": len(extreme_errors) / len(errors) if errors else 0.0,
        },
        "non_extreme_shape": {
            "objects": len(non_extreme),
            "errors": len(non_extreme_errors),
            "error_rate": len(non_extreme_errors) / len(non_extreme) if non_extreme else 0.0,
            "share_of_pair_errors": len(non_extreme_errors) / len(errors) if errors else 0.0,
        },
    }


def run(cpe0_event, cir0_event, scale_json, data_root, output) -> dict:
    cpe = _load_event(cpe0_event, "CPE0")
    cir = _load_event(cir0_event, "CIR0")
    if set(cpe["events"]) != set(cir["events"]):
        raise RuntimeError("Target universe CPE0/CIR0 berbeda")
    scale = _load_scale(scale_json)
    data_root = Path(data_root).expanduser().resolve()
    layout = discover_layout(data_root)
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit menolak data root yang mengekspos test")

    all_rows = _event_rows(data_root, cpe, cir)
    pair_results = []
    global_states = Counter()
    global_extreme_by_state: dict[str, Counter] = {}
    global_model_counts = {
        "CPE0": Counter(objects=0, errors=0, extreme_objects=0, extreme_errors=0),
        "CIR0": Counter(objects=0, errors=0, extreme_objects=0, extreme_errors=0),
    }

    for pair in scale["pairs"]:
        family = str(pair["family"])
        low_name = str(pair["low_class"])
        high_name = str(pair["high_class"])
        rows = [r.copy() for r in all_rows.values() if r["gt_class_name"] in (low_name, high_name)]
        if not rows:
            continue

        ars = np.asarray([r["aspect_ratio"] for r in rows], dtype=np.float64)
        ar_q25 = float(np.quantile(ars, 0.25))
        ar_med = float(np.median(ars))
        ar_q75 = float(np.quantile(ars, 0.75))
        ar_iqr = max(ar_q75 - ar_q25, 1e-12)

        for r in rows:
            r["geometry_state"] = _geometry_state(r, pair)
            r["shape_extremeness_iqr"] = _shape_extremeness(r["aspect_ratio"], ar_med, ar_iqr)
            r["extreme_shape_gt_1_iqr"] = bool(r["shape_extremeness_iqr"] > 1.0)
            global_states[r["geometry_state"]] += 1
            global_extreme_by_state.setdefault(r["geometry_state"], Counter())
            global_extreme_by_state[r["geometry_state"]]["objects"] += 1
            if r["extreme_shape_gt_1_iqr"]:
                global_extreme_by_state[r["geometry_state"]]["extreme"] += 1

        by_state = {
            state: _state_stats([r for r in rows if r["geometry_state"] == state])
            for state in ("both_support_gt", "mixed_geometry", "both_support_other")
        }
        mixed = [r["aspect_ratio"] for r in rows if r["geometry_state"] == "mixed_geometry"]
        clear = [r["aspect_ratio"] for r in rows if r["geometry_state"] == "both_support_gt"]

        models = {}
        for model_name, model_key in (("CPE0", "cpe0"), ("CIR0", "cir0")):
            models[model_name] = _model_stats(rows, model_key, family)
            m = global_model_counts[model_name]
            m["objects"] += len(rows)
            err = [r for r in rows if _pair_error(r[model_key], family)]
            m["errors"] += len(err)
            ex = [r for r in rows if r["extreme_shape_gt_1_iqr"]]
            m["extreme_objects"] += len(ex)
            m["extreme_errors"] += sum(_pair_error(r[model_key], family) for r in ex)

        pair_results.append({
            "family": family,
            "low_class": low_name,
            "high_class": high_name,
            "gt_instances_in_pair": len(rows),
            "aspect_ratio_reference": {
                "q25": ar_q25,
                "median": ar_med,
                "q75": ar_q75,
                "iqr": ar_iqr,
                "extreme_shape_definition": "abs(aspect_ratio - pair_median) / pair_IQR > 1",
            },
            "state_shape_stats": by_state,
            "mixed_vs_both_support_gt": {
                "aspect_ratio_auc_mixed_greater": _auc_greater(mixed, clear),
                "mixed_extreme_shape_rate": by_state["mixed_geometry"]["extreme_shape_rate"],
                "both_support_gt_extreme_shape_rate": by_state["both_support_gt"]["extreme_shape_rate"],
                "extreme_rate_difference_mixed_minus_clear": (
                    by_state["mixed_geometry"]["extreme_shape_rate"]
                    - by_state["both_support_gt"]["extreme_shape_rate"]
                ),
            },
            "models": models,
        })

    global_state_shape = {}
    for state, counts in global_extreme_by_state.items():
        objects = int(counts.get("objects", 0))
        extreme = int(counts.get("extreme", 0))
        global_state_shape[state] = {
            "objects": objects,
            "extreme_shape_objects": extreme,
            "extreme_shape_rate": extreme / objects if objects else 0.0,
        }

    global_models = {}
    for model_name, counts in global_model_counts.items():
        objects = int(counts["objects"])
        errors = int(counts["errors"])
        extreme_objects = int(counts["extreme_objects"])
        extreme_errors = int(counts["extreme_errors"])
        non_objects = objects - extreme_objects
        non_errors = errors - extreme_errors
        global_models[model_name] = {
            "pair_memberships": objects,
            "pair_errors": errors,
            "extreme_shape": {
                "objects": extreme_objects,
                "errors": extreme_errors,
                "error_rate": extreme_errors / extreme_objects if extreme_objects else 0.0,
                "share_of_pair_errors": extreme_errors / errors if errors else 0.0,
            },
            "non_extreme_shape": {
                "objects": non_objects,
                "errors": non_errors,
                "error_rate": non_errors / non_objects if non_objects else 0.0,
                "share_of_pair_errors": non_errors / errors if errors else 0.0,
            },
        }

    result = {
        "protocol": PROTOCOL,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "source_scale_audit": str(Path(scale_json).expanduser().resolve()),
        "method_note": (
            "Descriptive same-split geometry audit. Shape extremeness is frozen as absolute aspect-ratio distance "
            "from the pair median greater than one pair IQR. Association does not establish causality."
        ),
        "global_geometry_state_counts_across_pair_memberships": dict(global_states),
        "global_shape_by_geometry_state": global_state_shape,
        "global_model_shape_error_summary": global_models,
        "pairs": pair_results,
        "geometry_track_decision_rule": (
            "If mixed_geometry is not meaningfully more shape-extreme than both_support_gt and model pair-errors "
            "are not elevated on extreme shapes, close the geometry hypothesis track rather than building a geometry-conditioned model."
        ),
        "screening_decision_remains": "STOP_CIRCLE_CPE",
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", destination)
    print("GLOBAL STATES:", result["global_geometry_state_counts_across_pair_memberships"])
    print("GLOBAL SHAPE BY STATE:", json.dumps(global_state_shape, indent=2, ensure_ascii=False))
    print("GLOBAL MODEL SHAPE-ERROR SUMMARY:", json.dumps(global_models, indent=2, ensure_ascii=False))
    print("SCREENING DECISION REMAINS:", result["screening_decision_remains"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpe0-event", required=True)
    parser.add_argument("--cir0-event", required=True)
    parser.add_argument("--scale-json", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.cpe0_event, args.cir0_event, args.scale_json, args.data_root, args.output)


if __name__ == "__main__":
    main()
