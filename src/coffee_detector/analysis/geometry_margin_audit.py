from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from coffee_detector.analysis.scale_identifiability_audit import (
    PROTOCOL as SCALE_PROTOCOL,
    _event_feature_rows,
    _load_event,
    _pair_error,
)
from coffee_detector.dataset import discover_layout

PROTOCOL = "faruq-v3-geometry-margin-audit-v1"
BANDS = (
    ("boundary_le_0p25_iqr", 0.25),
    ("near_0p25_to_0p5_iqr", 0.50),
    ("mid_0p5_to_1_iqr", 1.00),
    ("far_gt_1_iqr", float("inf")),
)


def _load_scale(path: str | Path) -> dict:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("protocol") != SCALE_PROTOCOL:
        raise RuntimeError("Scale audit protocol tidak kompatibel")
    if int(payload.get("seed", -1)) != 42 or payload.get("evaluation_split") != "val":
        raise RuntimeError("Scale audit bukan seed42 val")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError("Scale audit menunjukkan akses test")
    return payload


def _robust_scale(low_stats: dict, high_stats: dict) -> float:
    q25 = min(float(low_stats["q25"]), float(high_stats["q25"]))
    q75 = max(float(low_stats["q75"]), float(high_stats["q75"]))
    pooled_iqr = q75 - q25
    if pooled_iqr > 0:
        return pooled_iqr
    stds = [float(low_stats.get("std", 0.0)), float(high_stats.get("std", 0.0))]
    fallback = max(stds)
    return fallback if fallback > 0 else 1.0


def _margin(value: float, threshold: float, scale: float) -> float:
    return abs(float(value) - float(threshold)) / float(scale)


def _band(value: float) -> str:
    for name, upper in BANDS:
        if value <= upper:
            return name
    raise AssertionError(value)


def _cue_supports_gt(value: float, threshold: float, gt: str, low_class: str, high_class: str) -> bool:
    predicted = low_class if float(value) <= float(threshold) else high_class
    return predicted == gt


def _rate(errors: int, total: int) -> float:
    return errors / total if total else 0.0


def _model_summary(rows: list[dict], model: str, family: str) -> dict:
    total_errors = sum(_pair_error(r[model], family) for r in rows)
    by_band_total = Counter(r["joint_margin_band"] for r in rows)
    by_band_errors = Counter(r["joint_margin_band"] for r in rows if _pair_error(r[model], family))
    clear_gt_rows = [r for r in rows if r["both_support_gt"]]
    clear_gt_errors = [r for r in clear_gt_rows if _pair_error(r[model], family)]
    far_clear_rows = [r for r in clear_gt_rows if r["joint_margin_norm"] > 1.0]
    far_clear_errors = [r for r in far_clear_rows if _pair_error(r[model], family)]
    boundary_rows = [r for r in rows if r["joint_margin_norm"] <= 0.25]
    boundary_errors = [r for r in boundary_rows if _pair_error(r[model], family)]
    return {
        "pair_errors_total": int(total_errors),
        "by_margin_band": {
            name: {
                "objects": int(by_band_total.get(name, 0)),
                "errors": int(by_band_errors.get(name, 0)),
                "error_rate": _rate(int(by_band_errors.get(name, 0)), int(by_band_total.get(name, 0))),
            }
            for name, _ in BANDS
        },
        "clear_gt_geometry": {
            "objects": len(clear_gt_rows),
            "errors": len(clear_gt_errors),
            "error_rate": _rate(len(clear_gt_errors), len(clear_gt_rows)),
        },
        "far_clear_gt_geometry_gt_1_iqr": {
            "objects": len(far_clear_rows),
            "errors": len(far_clear_errors),
            "error_rate": _rate(len(far_clear_errors), len(far_clear_rows)),
            "share_of_pair_errors": _rate(len(far_clear_errors), total_errors),
        },
        "boundary_le_0p25_iqr": {
            "objects": len(boundary_rows),
            "errors": len(boundary_errors),
            "error_rate": _rate(len(boundary_errors), len(boundary_rows)),
            "share_of_pair_errors": _rate(len(boundary_errors), total_errors),
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

    features = _event_feature_rows(data_root, cpe, cir)
    pair_results = []
    global_rows: list[dict] = []

    for pair in scale.get("pairs", []):
        family = str(pair["family"])
        low_class = str(pair["low_class"])
        high_class = str(pair["high_class"])
        pair_rows = [r for r in features.values() if r["gt_class_name"] in (low_class, high_class)]
        if not pair_rows:
            continue

        L = pair["features"]["long_side_norm"]
        A = pair["features"]["area_norm"]
        thresholds = {
            "long_side_norm": float(L["best_threshold"]),
            "area_norm": float(A["best_threshold"]),
        }
        scales = {
            "long_side_norm": _robust_scale(L["low_class"], L["high_class"]),
            "area_norm": _robust_scale(A["low_class"], A["high_class"]),
        }

        enriched = []
        for row in pair_rows:
            gt = str(row["gt_class_name"])
            lm = _margin(row["long_side_norm"], thresholds["long_side_norm"], scales["long_side_norm"])
            am = _margin(row["area_norm"], thresholds["area_norm"], scales["area_norm"])
            ls = _cue_supports_gt(row["long_side_norm"], thresholds["long_side_norm"], gt, low_class, high_class)
            ars = _cue_supports_gt(row["area_norm"], thresholds["area_norm"], gt, low_class, high_class)
            item = {
                **row,
                "family": family,
                "long_margin_norm": lm,
                "area_margin_norm": am,
                "joint_margin_norm": min(lm, am),
                "joint_margin_band": _band(min(lm, am)),
                "long_supports_gt": bool(ls),
                "area_supports_gt": bool(ars),
                "both_support_gt": bool(ls and ars),
            }
            enriched.append(item)
            global_rows.append(item)

        pair_results.append({
            "family": family,
            "low_class": low_class,
            "high_class": high_class,
            "gt_instances_in_pair": len(enriched),
            "thresholds_reused": thresholds,
            "robust_margin_scales": scales,
            "models": {
                "CPE0": _model_summary(enriched, "cpe0", family),
                "CIR0": _model_summary(enriched, "cir0", family),
            },
        })

    global_models = {}
    for display, key in (("CPE0", "cpe0"), ("CIR0", "cir0")):
        total_errors = sum(_pair_error(r[key], r["family"]) for r in global_rows)
        by_band_total = Counter(r["joint_margin_band"] for r in global_rows)
        by_band_errors = Counter(r["joint_margin_band"] for r in global_rows if _pair_error(r[key], r["family"]))
        far_clear = [r for r in global_rows if r["both_support_gt"] and r["joint_margin_norm"] > 1.0]
        far_clear_err = [r for r in far_clear if _pair_error(r[key], r["family"])]
        global_models[display] = {
            "pair_errors_total_across_pair_memberships": int(total_errors),
            "by_margin_band": {
                name: {
                    "objects": int(by_band_total.get(name, 0)),
                    "errors": int(by_band_errors.get(name, 0)),
                    "error_rate": _rate(int(by_band_errors.get(name, 0)), int(by_band_total.get(name, 0))),
                }
                for name, _ in BANDS
            },
            "far_clear_gt_geometry_gt_1_iqr": {
                "objects": len(far_clear),
                "errors": len(far_clear_err),
                "error_rate": _rate(len(far_clear_err), len(far_clear)),
                "share_of_pair_errors": _rate(len(far_clear_err), total_errors),
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
            "Descriptive same-split audit. Thresholds are reused from the prior scale audit. "
            "For each cue, absolute distance to threshold is normalized by pooled pair IQR; "
            "joint proximity is the minimum normalized margin across long-side and area."
        ),
        "frozen_margin_bands_before_results": {
            "boundary_le_0p25_iqr": "joint margin <= 0.25 pooled-IQR",
            "near_0p25_to_0p5_iqr": "0.25 < joint margin <= 0.5 pooled-IQR",
            "mid_0p5_to_1_iqr": "0.5 < joint margin <= 1 pooled-IQR",
            "far_gt_1_iqr": "joint margin > 1 pooled-IQR",
        },
        "global_model_summary": global_models,
        "pairs": pair_results,
        "interpretation_guardrail": (
            "High error near the boundary supports association with borderline geometry; errors far from the boundary "
            "while both cues support GT are compatible with underused available geometry, but this same-split audit "
            "does not establish causality or justify a new model by itself."
        ),
        "screening_decision_remains": "STOP_CIRCLE_CPE",
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", destination)
    print("GLOBAL MODEL SUMMARY:", json.dumps(global_models, ensure_ascii=False))
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
