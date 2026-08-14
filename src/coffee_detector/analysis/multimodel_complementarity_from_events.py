"""Aggregate seed-42 validation object events across multiple detector variants."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

EXPECTED_PROTOCOL = "faruq-v3-validation-object-events-v1"


def _safe_div(a, b):
    return float(a / b) if b else 0.0


def _load_event_file(path: str | Path):
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol") != EXPECTED_PROTOCOL:
        raise RuntimeError(f"Protocol event tidak kompatibel: {source}")
    if int(payload.get("seed", -1)) != 42:
        raise RuntimeError(f"Audit multimodel dikunci ke seed42: {source}")
    if payload.get("evaluation_split") != "val":
        raise RuntimeError(f"Event bukan validation-only: {source}")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError(f"Event menunjukkan akses test: {source}")
    return payload


def _pair(a_name, a_rows, b_name, b_rows):
    if set(a_rows) != set(b_rows):
        raise RuntimeError(f"Target universe berbeda: {a_name} vs {b_name}")
    total = len(a_rows)
    both_correct = a_only = b_only = neither = 0
    both_matched = a_to_b_cls = b_to_a_cls = 0
    top_a_to_b, top_b_to_a = {}, {}
    for key in a_rows:
        a, b = a_rows[key], b_rows[key]
        if a["gt_class_id"] != b["gt_class_id"]:
            raise RuntimeError(f"GT mismatch: {key}")
        ac, bc = bool(a["correct"]), bool(b["correct"])
        if ac and bc:
            both_correct += 1
        elif ac:
            a_only += 1
        elif bc:
            b_only += 1
        else:
            neither += 1
        if a["matched"] and b["matched"]:
            both_matched += 1
            if (not ac) and bc:
                a_to_b_cls += 1
            if (not bc) and ac:
                b_to_a_cls += 1
        if (not ac) and bc and a["matched"] and a["pred_class_name"] != a["gt_class_name"]:
            k = (a["gt_class_name"], a["pred_class_name"])
            top_a_to_b[k] = top_a_to_b.get(k, 0) + 1
        if (not bc) and ac and b["matched"] and b["pred_class_name"] != b["gt_class_name"]:
            k = (b["gt_class_name"], b["pred_class_name"])
            top_b_to_a[k] = top_b_to_a.get(k, 0) + 1

    a_correct = both_correct + a_only
    b_correct = both_correct + b_only
    a_errors = total - a_correct
    b_errors = total - b_correct
    error_intersection = neither
    error_union = total - both_correct
    oracle_correct = total - neither
    best_correct = max(a_correct, b_correct)

    def top(counter):
        rows = [
            {"gt": gt, "wrong_pred": pred, "count": count}
            for (gt, pred), count in counter.items()
        ]
        return sorted(rows, key=lambda x: (-x["count"], x["gt"], x["wrong_pred"]))[:15]

    return {
        "model_a": a_name,
        "model_b": b_name,
        "targets": total,
        "a_accuracy_iou50": _safe_div(a_correct, total),
        "b_accuracy_iou50": _safe_div(b_correct, total),
        "b_minus_a_accuracy": _safe_div(b_correct - a_correct, total),
        "a_to_b_rescue_rate": _safe_div(b_only, a_errors),
        "b_to_a_rescue_rate": _safe_div(a_only, b_errors),
        "both_correct": both_correct,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "neither_correct": neither,
        "error_jaccard": _safe_div(error_intersection, error_union),
        "oracle_accuracy_iou50": _safe_div(oracle_correct, total),
        "oracle_gain_over_best": _safe_div(oracle_correct - best_correct, total),
        "jointly_matched_targets": both_matched,
        "a_to_b_classification_only_rescue_rate": _safe_div(a_to_b_cls, both_matched),
        "b_to_a_classification_only_rescue_rate": _safe_div(b_to_a_cls, both_matched),
        "top_a_wrong_b_rescues": top(top_a_to_b),
        "top_b_wrong_a_rescues": top(top_b_to_a),
    }


def _model_summary(name, rows):
    total = len(rows)
    correct = sum(bool(r["correct"]) for r in rows.values())
    matched = sum(bool(r["matched"]) for r in rows.values())
    accessible = sum(bool(r["accessible"]) for r in rows.values())
    return {
        "model": name,
        "targets": total,
        "accuracy_iou50": _safe_div(correct, total),
        "matched_recall_iou50": _safe_div(matched, total),
        "accessibility_iou50": _safe_div(accessible, total),
        "errors": total - correct,
    }


def aggregate(event_files: dict[str, str | Path], output_root: str | Path):
    if len(event_files) < 2:
        raise ValueError("Minimal dua model")
    loaded = {name: _load_event_file(path) for name, path in event_files.items()}
    universes = {name: set(payload["events"]) for name, payload in loaded.items()}
    first_name = next(iter(universes))
    first = universes[first_name]
    for name, universe in universes.items():
        if universe != first:
            raise RuntimeError(f"Target universe {name} tidak sama dengan {first_name}")

    models = {name: _model_summary(name, payload["events"]) for name, payload in loaded.items()}
    pairs = []
    for a, b in combinations(loaded, 2):
        pairs.append(_pair(a, loaded[a]["events"], b, loaded[b]["events"]))
    pairs_by_oracle = sorted(pairs, key=lambda r: (-r["oracle_gain_over_best"], r["error_jaccard"], r["model_a"], r["model_b"]))

    all_keys = sorted(first)
    any_correct = sum(
        any(bool(loaded[name]["events"][key]["correct"]) for name in loaded)
        for key in all_keys
    )
    best_single = max(int(round(models[name]["accuracy_iou50"] * len(all_keys))) for name in models)
    payload = {
        "protocol": "faruq-v3-multimodel-complementarity-seed42-v1",
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "models": models,
        "pairwise": pairs,
        "pairwise_ranked_by_oracle_headroom": pairs_by_oracle,
        "all_model_oracle": {
            "targets": len(all_keys),
            "oracle_accuracy_iou50": _safe_div(any_correct, len(all_keys)),
            "oracle_gain_over_best_single": _safe_div(any_correct - best_single, len(all_keys)),
        },
    }
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = output_root / "multimodel_complementarity_seed42.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_root / "pairwise_complementarity_seed42.csv").open("w", newline="", encoding="utf-8") as f:
        scalar_keys = [
            "model_a", "model_b", "targets", "a_accuracy_iou50", "b_accuracy_iou50",
            "b_minus_a_accuracy", "a_to_b_rescue_rate", "b_to_a_rescue_rate",
            "error_jaccard", "oracle_accuracy_iou50", "oracle_gain_over_best",
            "jointly_matched_targets", "a_to_b_classification_only_rescue_rate",
            "b_to_a_classification_only_rescue_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in pairs_by_oracle:
            writer.writerow({key: row[key] for key in scalar_keys})
    payload["summary"] = str(summary)
    return payload


def _parse_event_arg(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Gunakan MODEL=/path/events.json, dapat: {value}")
        name, path = value.split("=", 1)
        result[name] = path
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", action="append", required=True, help="MODEL=/path/events.json")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = aggregate(_parse_event_arg(args.event), args.output_root)
    print("MODEL ACCURACY:")
    for row in sorted(result["models"].values(), key=lambda x: -x["accuracy_iou50"]):
        print(row["model"], f"acc={row['accuracy_iou50']:.2%}")
    print("TOP PAIRS BY ORACLE HEADROOM:")
    for row in result["pairwise_ranked_by_oracle_headroom"][:10]:
        print(
            f"{row['model_a']} vs {row['model_b']}",
            f"J={row['error_jaccard']:.2%}",
            f"oracle_gain={row['oracle_gain_over_best']:.2%}",
            f"A->B={row['a_to_b_rescue_rate']:.2%}",
            f"B->A={row['b_to_a_rescue_rate']:.2%}",
        )
    print("ALL-MODEL ORACLE:", result["all_model_oracle"])
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
