"""Cross-model hard-confusion consensus audit from frozen validation object events.

Validation-only, post-training analysis. It uses class-agnostic IoU>=0.50 GT-aligned
object events already exported for seed42. No inference, training, or test access.

Goal: identify confusion families that recur across independently modified models.
A recurring family is stronger evidence for a shared decision-boundary problem than
an error seen in only one model.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EVENT_PROTOCOL = "faruq-v3-validation-object-events-v1"
PROTOCOL = "faruq-v3-cross-model-hard-confusion-consensus-v1"

# Frozen before reading this audit's output.
MIN_MODELS_FOR_CONSENSUS = 4
MIN_TOTAL_SUPPORT_FOR_CONSENSUS = 8


def _load_event(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol") != EVENT_PROTOCOL:
        raise RuntimeError(f"Protocol event tidak kompatibel: {source}")
    if payload.get("evaluation_split") != "val":
        raise RuntimeError(f"Bukan validation event: {source}")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError(f"Event menunjukkan akses test: {source}")
    if int(payload.get("seed", -1)) != 42:
        raise RuntimeError(f"Audit dikunci ke seed42: {source}")
    if not isinstance(payload.get("events"), dict) or not payload["events"]:
        raise RuntimeError(f"Event kosong: {source}")
    return payload


def _concentration(counter: Counter[str]) -> dict:
    values = sorted(counter.values(), reverse=True)
    total = sum(values)
    if not total:
        return {
            "total_errors": 0,
            "n_nonzero_families": 0,
            "top1_share": 0.0,
            "top3_share": 0.0,
            "top5_share": 0.0,
            "hhi": 0.0,
            "normalized_entropy": 0.0,
        }
    shares = [v / total for v in values]
    entropy = -sum(p * math.log(p) for p in shares if p > 0)
    norm_entropy = entropy / math.log(len(shares)) if len(shares) > 1 else 0.0
    return {
        "total_errors": total,
        "n_nonzero_families": len(values),
        "top1_share": sum(values[:1]) / total,
        "top3_share": sum(values[:3]) / total,
        "top5_share": sum(values[:5]) / total,
        "hhi": sum(p * p for p in shares),
        "normalized_entropy": norm_entropy,
    }


def _model_confusions(payload: dict) -> tuple[Counter[str], Counter[str], Counter[str]]:
    directed: Counter[str] = Counter()
    undirected: Counter[str] = Counter()
    by_gt: Counter[str] = Counter()
    for row in payload["events"].values():
        # classification-only error: model has already localized/matched this GT
        if not row.get("matched") or row.get("correct"):
            continue
        gt = row.get("gt_class_name")
        pred = row.get("pred_class_name")
        if gt is None or pred is None or gt == pred:
            continue
        directed[f"{gt} -> {pred}"] += 1
        undirected[" <-> ".join(sorted((str(gt), str(pred))))] += 1
        by_gt[str(gt)] += 1
    return directed, undirected, by_gt


def _consensus_rows(per_model: dict[str, Counter[str]]) -> list[dict]:
    all_families = sorted({family for counter in per_model.values() for family in counter})
    rows = []
    for family in all_families:
        supports = {model: int(counter.get(family, 0)) for model, counter in per_model.items()}
        present = [model for model, count in supports.items() if count > 0]
        total = sum(supports.values())
        rows.append({
            "family": family,
            "models_with_error": len(present),
            "model_fraction": len(present) / len(per_model),
            "total_support": total,
            "mean_support_per_model": total / len(per_model),
            "max_single_model_support": max(supports.values()) if supports else 0,
            "models_present": present,
            "support_by_model": supports,
            "frozen_consensus_hard_family": (
                len(present) >= MIN_MODELS_FOR_CONSENSUS
                and total >= MIN_TOTAL_SUPPORT_FOR_CONSENSUS
            ),
        })
    rows.sort(
        key=lambda r: (
            r["frozen_consensus_hard_family"],
            r["models_with_error"],
            r["total_support"],
        ),
        reverse=True,
    )
    return rows


def run(event_paths: list[str | Path], output: str | Path) -> dict:
    payloads = [_load_event(path) for path in event_paths]
    names = [p["model"] for p in payloads]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Nama model duplikat: {names}")
    universes = [set(p["events"]) for p in payloads]
    if any(u != universes[0] for u in universes[1:]):
        raise RuntimeError("Target universe antar-event berbeda")

    directed_by_model: dict[str, Counter[str]] = {}
    undirected_by_model: dict[str, Counter[str]] = {}
    gt_by_model: dict[str, Counter[str]] = {}
    model_summaries = {}

    for payload in payloads:
        model = payload["model"]
        directed, undirected, by_gt = _model_confusions(payload)
        directed_by_model[model] = directed
        undirected_by_model[model] = undirected
        gt_by_model[model] = by_gt
        model_summaries[model] = {
            "classification_errors_iou50": int(sum(directed.values())),
            "directed_concentration": _concentration(directed),
            "undirected_concentration": _concentration(undirected),
            "gt_error_concentration": _concentration(by_gt),
        }

    directed_rows = _consensus_rows(directed_by_model)
    undirected_rows = _consensus_rows(undirected_by_model)
    gt_rows = _consensus_rows(gt_by_model)

    result = {
        "protocol": PROTOCOL,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "models": names,
        "targets": len(universes[0]),
        "frozen_before_results": True,
        "frozen_consensus_gate": {
            "min_models_with_error": MIN_MODELS_FOR_CONSENSUS,
            "min_total_support": MIN_TOTAL_SUPPORT_FOR_CONSENSUS,
            "purpose": "descriptive gate for recurring hard confusion families; not a model-selection claim",
        },
        "model_summaries": model_summaries,
        "directed_families": directed_rows,
        "undirected_families": undirected_rows,
        "gt_classes": gt_rows,
        "consensus_counts": {
            "directed": sum(r["frozen_consensus_hard_family"] for r in directed_rows),
            "undirected": sum(r["frozen_consensus_hard_family"] for r in undirected_rows),
            "gt_class": sum(r["frozen_consensus_hard_family"] for r in gt_rows),
        },
        "interpretation_warning": (
            "Seed42-only cross-model diagnostic. Repeated confusion across models supports shared difficulty, "
            "but does not establish multi-seed robustness or justify test access."
        ),
    }

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", destination)
    print("CONSENSUS COUNTS:", result["consensus_counts"])
    for label, rows in (("DIRECTED", directed_rows), ("UNDIRECTED", undirected_rows)):
        print(f"\nTOP {label} FAMILIES")
        for row in rows[:12]:
            print(
                row["family"],
                "models=", row["models_with_error"],
                "support=", row["total_support"],
                "CONSENSUS" if row["frozen_consensus_hard_family"] else "",
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.event, args.output)


if __name__ == "__main__":
    main()
