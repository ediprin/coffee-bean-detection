"""Validation-only IGEM1 -> AF2 targeted rescue audit from frozen object events.

This post-training diagnostic asks whether AF2 rescues IGEM1 errors in a concentrated,
class/confusion-specific way that could justify a later targeted auxiliary objective.
It does not train, tune, or access test. Confusion families are defined mechanically
from IGEM1 errors, not by semantic hand-grouping:

- directed family: GT class -> IGEM1 wrong predicted class
- undirected family: unordered {GT class, IGEM1 wrong predicted class}

Classification-only results require IGEM1 to have matched the GT at IoU >= 0.50 but
predicted the wrong class. Total-error results are reported separately so misses are
not silently mixed into a classification claim.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

PROTOCOL = "faruq-v3-validation-object-events-v1"
AUDIT_PROTOCOL = "faruq-v3-igem-af2-targeted-rescue-v1"


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / total) + z * z / (4.0 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _load(path: str | Path, expected_model: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Protocol event tidak kompatibel: {source}")
    if payload.get("model") != expected_model:
        raise RuntimeError(f"Expected model {expected_model}, got {payload.get('model')}: {source}")
    if int(payload.get("seed", -1)) != 42:
        raise RuntimeError(f"Audit dikunci seed42: {source}")
    if payload.get("evaluation_split") != "val":
        raise RuntimeError(f"Event bukan validation-only: {source}")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError(f"Event menunjukkan akses test: {source}")
    if not isinstance(payload.get("events"), dict) or not payload["events"]:
        raise RuntimeError(f"Event kosong: {source}")
    return payload


def _event_pairs(primary: dict, candidate: dict) -> list[tuple[dict, dict]]:
    a_events, b_events = primary["events"], candidate["events"]
    if set(a_events) != set(b_events):
        raise RuntimeError("Target universe IGEM1 dan AF2 berbeda")
    result = []
    for key in sorted(a_events):
        a, b = a_events[key], b_events[key]
        if a.get("gt_class_id") != b.get("gt_class_id") or a.get("gt_class_name") != b.get("gt_class_name"):
            raise RuntimeError(f"GT mismatch: {key}")
        result.append((a, b))
    return result


def _concentration(counts: Counter[str]) -> dict:
    values = sorted(counts.values(), reverse=True)
    total = sum(values)
    if not total:
        return {
            "total_rescues": 0,
            "n_nonzero_families": 0,
            "top1_share": 0.0,
            "top3_share": 0.0,
            "top5_share": 0.0,
            "hhi": 0.0,
            "normalized_entropy": 0.0,
        }
    shares = [value / total for value in values]
    entropy = -sum(p * math.log(p) for p in shares if p > 0)
    normalized_entropy = entropy / math.log(len(shares)) if len(shares) > 1 else 0.0
    return {
        "total_rescues": total,
        "n_nonzero_families": len(values),
        "top1_share": _safe_div(sum(values[:1]), total),
        "top3_share": _safe_div(sum(values[:3]), total),
        "top5_share": _safe_div(sum(values[:5]), total),
        "hhi": sum(p * p for p in shares),
        "normalized_entropy": normalized_entropy,
    }


def audit(primary: dict, candidate: dict) -> dict:
    pairs = _event_pairs(primary, candidate)
    classes = sorted({a["gt_class_name"] for a, _ in pairs})

    per_class: dict[str, dict] = {}
    directed_support: Counter[str] = Counter()
    directed_rescue: Counter[str] = Counter()
    undirected_support: Counter[str] = Counter()
    undirected_rescue: Counter[str] = Counter()
    directed_examples: dict[str, list[str]] = defaultdict(list)

    global_total_errors = 0
    global_total_rescues = 0
    global_cls_errors = 0
    global_cls_rescues = 0
    global_harms = 0

    for class_name in classes:
        class_pairs = [(a, b) for a, b in pairs if a["gt_class_name"] == class_name]
        igem_errors = [(a, b) for a, b in class_pairs if not bool(a.get("correct"))]
        igem_cls_errors = [
            (a, b) for a, b in class_pairs
            if bool(a.get("matched")) and not bool(a.get("correct")) and a.get("pred_class_name") is not None
        ]
        total_rescues = [(a, b) for a, b in igem_errors if bool(b.get("correct"))]
        cls_rescues = [(a, b) for a, b in igem_cls_errors if bool(b.get("correct"))]
        harms = [(a, b) for a, b in class_pairs if bool(a.get("correct")) and not bool(b.get("correct"))]

        low_total, high_total = _wilson(len(total_rescues), len(igem_errors))
        low_cls, high_cls = _wilson(len(cls_rescues), len(igem_cls_errors))
        per_class[class_name] = {
            "targets": len(class_pairs),
            "igem_errors_total": len(igem_errors),
            "af2_rescues_total": len(total_rescues),
            "total_error_rescue_rate": _safe_div(len(total_rescues), len(igem_errors)),
            "total_error_rescue_wilson95_low": low_total,
            "total_error_rescue_wilson95_high": high_total,
            "igem_classification_errors_iou50": len(igem_cls_errors),
            "af2_classification_rescues_iou50": len(cls_rescues),
            "classification_rescue_rate": _safe_div(len(cls_rescues), len(igem_cls_errors)),
            "classification_rescue_wilson95_low": low_cls,
            "classification_rescue_wilson95_high": high_cls,
            "af2_harms_when_igem_correct": len(harms),
            "net_rescue_minus_harm": len(total_rescues) - len(harms),
        }

        global_total_errors += len(igem_errors)
        global_total_rescues += len(total_rescues)
        global_cls_errors += len(igem_cls_errors)
        global_cls_rescues += len(cls_rescues)
        global_harms += len(harms)

        for a, b in igem_cls_errors:
            gt = str(a["gt_class_name"])
            pred = str(a["pred_class_name"])
            directed = f"{gt} -> {pred}"
            undirected = " <-> ".join(sorted((gt, pred)))
            directed_support[directed] += 1
            undirected_support[undirected] += 1
            if bool(b.get("correct")):
                directed_rescue[directed] += 1
                undirected_rescue[undirected] += 1
                if len(directed_examples[directed]) < 8:
                    directed_examples[directed].append(str(a.get("target_key") or a.get("image")))

    directed_rows = []
    for family, support in directed_support.items():
        rescues = directed_rescue[family]
        low, high = _wilson(rescues, support)
        directed_rows.append({
            "family": family,
            "support_igem_classification_errors": support,
            "af2_rescues": rescues,
            "rescue_rate": _safe_div(rescues, support),
            "wilson95_low": low,
            "wilson95_high": high,
            "example_targets": directed_examples.get(family, []),
        })
    directed_rows.sort(key=lambda row: (row["af2_rescues"], row["support_igem_classification_errors"], row["rescue_rate"]), reverse=True)

    undirected_rows = []
    for family, support in undirected_support.items():
        rescues = undirected_rescue[family]
        low, high = _wilson(rescues, support)
        undirected_rows.append({
            "family": family,
            "support_igem_classification_errors": support,
            "af2_rescues": rescues,
            "rescue_rate": _safe_div(rescues, support),
            "wilson95_low": low,
            "wilson95_high": high,
        })
    undirected_rows.sort(key=lambda row: (row["af2_rescues"], row["support_igem_classification_errors"], row["rescue_rate"]), reverse=True)

    total_low, total_high = _wilson(global_total_rescues, global_total_errors)
    cls_low, cls_high = _wilson(global_cls_rescues, global_cls_errors)

    class_rescue_counter = Counter({name: row["af2_classification_rescues_iou50"] for name, row in per_class.items() if row["af2_classification_rescues_iou50"] > 0})

    return {
        "protocol": AUDIT_PROTOCOL,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "primary": primary["model"],
        "candidate": candidate["model"],
        "definitions": {
            "total_error_rescue": "IGEM1 wrong for any reason and AF2 correct on the same GT-anchored target",
            "classification_error": "IGEM1 matched the GT at IoU>=0.50 but predicted the wrong class",
            "classification_rescue": "classification_error where AF2 is correct on the same target",
            "directed_confusion_family": "GT class -> IGEM1 wrong predicted class",
            "undirected_confusion_family": "unordered pair {GT class, IGEM1 wrong predicted class}",
        },
        "global": {
            "targets": len(pairs),
            "igem_errors_total": global_total_errors,
            "af2_rescues_total": global_total_rescues,
            "total_error_rescue_rate": _safe_div(global_total_rescues, global_total_errors),
            "total_error_rescue_wilson95_low": total_low,
            "total_error_rescue_wilson95_high": total_high,
            "igem_classification_errors_iou50": global_cls_errors,
            "af2_classification_rescues_iou50": global_cls_rescues,
            "classification_rescue_rate": _safe_div(global_cls_rescues, global_cls_errors),
            "classification_rescue_wilson95_low": cls_low,
            "classification_rescue_wilson95_high": cls_high,
            "af2_harms_when_igem_correct": global_harms,
            "net_total_rescue_minus_harm": global_total_rescues - global_harms,
        },
        "per_class": per_class,
        "directed_confusion_families": directed_rows,
        "undirected_confusion_families": undirected_rows,
        "rescue_concentration": {
            "by_directed_confusion_family": _concentration(directed_rescue),
            "by_undirected_confusion_family": _concentration(undirected_rescue),
            "by_gt_class": _concentration(class_rescue_counter),
        },
        "interpretation_warning": (
            "Seed42 validation diagnostic only. High rescue rate with tiny support is not evidence of a stable target. "
            "Do not define a training objective from a post-hoc family without subsequent frozen replication."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--igem", required=True)
    parser.add_argument("--af2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = audit(_load(args.igem, "IGEM1"), _load(args.af2, "AF2"))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", output)
    print("GLOBAL:", result["global"])
    print("CONCENTRATION:", result["rescue_concentration"])
    print("TOP DIRECTED FAMILIES:")
    for row in result["directed_confusion_families"][:10]:
        print(row)


if __name__ == "__main__":
    main()
