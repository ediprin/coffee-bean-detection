"""GT-anchored confidence-selector feasibility audit from frozen validation events.

This is a post-training diagnostic only. It does not train, tune, or access test.
The object-event files already align each model's final detection to the same GT
object class-agnostically at IoU >= 0.50. We use that alignment only to ask a
narrow question: on objects that *both* models localize and classify differently,
does raw final-detection confidence identify which model is correct?

Because the alignment itself uses GT, results from this module are NOT a
deployable routing result. A deployable router would require prediction-to-
prediction matching without GT in a separately frozen protocol.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

PROTOCOL = "faruq-v3-validation-object-events-v1"
PAIR_PROTOCOL = "faruq-v3-gt-anchored-selector-feasibility-v1"
THRESHOLDS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
GAP_BINS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01))


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _load_event(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Protocol event tidak kompatibel: {source}")
    if payload.get("evaluation_split") != "val":
        raise RuntimeError(f"Event bukan validation-only: {source}")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError(f"Event menunjukkan akses test: {source}")
    if int(payload.get("seed", -1)) != 42:
        raise RuntimeError(f"Audit selector saat ini dikunci ke seed42: {source}")
    if not isinstance(payload.get("events"), dict) or not payload["events"]:
        raise RuntimeError(f"Event kosong: {source}")
    return payload


def _rank_auc(scores: list[float], labels: list[bool]) -> float | None:
    """Mann-Whitney AUC with average ranks for ties, no sklearn/scipy dependency."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # one-based ranks i+1 ... j
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    rank_sum_pos = sum(rank for rank, label in zip(ranks, labels) if label)
    u = rank_sum_pos - positives * (positives + 1) / 2.0
    return float(u / (positives * negatives))


def _confidence_self_awareness(events: dict) -> dict:
    rows = [row for row in events.values() if row.get("matched") and row.get("confidence") is not None]
    scores = [float(row["confidence"]) for row in rows]
    labels = [bool(row["correct"]) for row in rows]
    correct_scores = [score for score, label in zip(scores, labels) if label]
    wrong_scores = [score for score, label in zip(scores, labels) if not label]
    auc = _rank_auc(scores, labels)
    return {
        "matched_targets": len(rows),
        "correct_targets": sum(labels),
        "wrong_targets": len(rows) - sum(labels),
        "mean_confidence_correct": mean(correct_scores) if correct_scores else None,
        "mean_confidence_wrong": mean(wrong_scores) if wrong_scores else None,
        "confidence_correctness_auc": auc,
    }


def _paired_rows(a_events: dict, b_events: dict) -> list[dict]:
    if set(a_events) != set(b_events):
        raise RuntimeError("Target universe antar-event berbeda")
    rows = []
    for key in sorted(a_events):
        a, b = a_events[key], b_events[key]
        if a.get("gt_class_id") != b.get("gt_class_id"):
            raise RuntimeError(f"GT mismatch: {key}")
        row = {
            "key": key,
            "image": a.get("image"),
            "gt": a.get("gt_class_name"),
            "a_matched": bool(a.get("matched")),
            "b_matched": bool(b.get("matched")),
            "a_correct": bool(a.get("correct")),
            "b_correct": bool(b.get("correct")),
            "a_pred": a.get("pred_class_name"),
            "b_pred": b.get("pred_class_name"),
            "a_conf": float(a["confidence"]) if a.get("confidence") is not None else None,
            "b_conf": float(b["confidence"]) if b.get("confidence") is not None else None,
        }
        row["joint_matched"] = row["a_matched"] and row["b_matched"]
        row["disagree"] = row["joint_matched"] and row["a_pred"] != row["b_pred"]
        if row["disagree"]:
            row["gap"] = abs(row["a_conf"] - row["b_conf"])
            row["higher_conf"] = "a" if row["a_conf"] >= row["b_conf"] else "b"
            row["higher_conf_correct"] = (
                row["a_correct"] if row["higher_conf"] == "a" else row["b_correct"]
            )
            row["exactly_one_correct"] = row["a_correct"] != row["b_correct"]
            row["higher_conf_picks_correct_expert"] = (
                row["higher_conf_correct"] if row["exactly_one_correct"] else None
            )
        else:
            row["gap"] = None
            row["higher_conf"] = None
            row["higher_conf_correct"] = None
            row["exactly_one_correct"] = False
            row["higher_conf_picks_correct_expert"] = None
        rows.append(row)
    return rows


def _threshold_policy(rows: list[dict], threshold: float) -> dict:
    """GT-aligned diagnostic: default A; switch to B only on joint-match disagreement and conf_B-conf_A > threshold."""
    chosen_correct = 0
    switched = 0
    beneficial = 0
    harmful = 0
    for row in rows:
        choose_b = (
            row["disagree"]
            and row["b_conf"] is not None
            and row["a_conf"] is not None
            and (row["b_conf"] - row["a_conf"]) > threshold
        )
        if choose_b:
            switched += 1
            chosen_correct += int(row["b_correct"])
            beneficial += int((not row["a_correct"]) and row["b_correct"])
            harmful += int(row["a_correct"] and (not row["b_correct"]))
        else:
            chosen_correct += int(row["a_correct"])
    return {
        "threshold": threshold,
        "switches": switched,
        "switch_rate": _safe_div(switched, len(rows)),
        "beneficial_switches": beneficial,
        "harmful_switches": harmful,
        "net_correct_delta_vs_primary": beneficial - harmful,
        "gt_aligned_object_accuracy": _safe_div(chosen_correct, len(rows)),
    }


def _gap_bin_summary(disagreements: list[dict]) -> list[dict]:
    result = []
    for low, high in GAP_BINS:
        subset = [row for row in disagreements if low <= float(row["gap"]) < high]
        resolvable = [row for row in subset if row["exactly_one_correct"]]
        result.append({
            "gap_low": low,
            "gap_high": high,
            "n_disagreements": len(subset),
            "n_exactly_one_correct": len(resolvable),
            "primary_accuracy": _safe_div(sum(row["a_correct"] for row in subset), len(subset)),
            "candidate_accuracy": _safe_div(sum(row["b_correct"] for row in subset), len(subset)),
            "higher_conf_accuracy_all_disagreements": _safe_div(
                sum(row["higher_conf_correct"] for row in subset), len(subset)
            ),
            "higher_conf_correct_expert_rate_when_resolvable": _safe_div(
                sum(bool(row["higher_conf_picks_correct_expert"]) for row in resolvable), len(resolvable)
            ),
            "oracle_accuracy": _safe_div(
                sum(row["a_correct"] or row["b_correct"] for row in subset), len(subset)
            ),
        })
    return result


def compare_pair(a_payload: dict, b_payload: dict) -> dict:
    a_name, b_name = a_payload["model"], b_payload["model"]
    rows = _paired_rows(a_payload["events"], b_payload["events"])
    joint = [row for row in rows if row["joint_matched"]]
    disagreements = [row for row in rows if row["disagree"]]
    resolvable = [row for row in disagreements if row["exactly_one_correct"]]
    both_wrong = [row for row in disagreements if not row["a_correct"] and not row["b_correct"]]
    agreement = [row for row in joint if not row["disagree"]]

    higher_correct = sum(bool(row["higher_conf_correct"]) for row in disagreements)
    higher_picks = sum(bool(row["higher_conf_picks_correct_expert"]) for row in resolvable)
    primary_on_disagreement = sum(row["a_correct"] for row in disagreements)
    candidate_on_disagreement = sum(row["b_correct"] for row in disagreements)
    oracle_disagreement = sum(row["a_correct"] or row["b_correct"] for row in disagreements)

    return {
        "primary": a_name,
        "candidate": b_name,
        "targets": len(rows),
        "joint_matched_targets": len(joint),
        "joint_matched_rate": _safe_div(len(joint), len(rows)),
        "agreement_targets": len(agreement),
        "disagreement_targets": len(disagreements),
        "disagreement_rate_among_joint_matched": _safe_div(len(disagreements), len(joint)),
        "resolvable_disagreements_exactly_one_correct": len(resolvable),
        "both_wrong_disagreements": len(both_wrong),
        "resolvable_fraction_of_disagreements": _safe_div(len(resolvable), len(disagreements)),
        "primary_accuracy_on_disagreements": _safe_div(primary_on_disagreement, len(disagreements)),
        "candidate_accuracy_on_disagreements": _safe_div(candidate_on_disagreement, len(disagreements)),
        "higher_conf_accuracy_on_all_disagreements": _safe_div(higher_correct, len(disagreements)),
        "higher_conf_correct_expert_rate_when_exactly_one_correct": _safe_div(higher_picks, len(resolvable)),
        "oracle_accuracy_on_disagreements": _safe_div(oracle_disagreement, len(disagreements)),
        "higher_conf_gain_vs_primary_on_disagreements": _safe_div(higher_correct - primary_on_disagreement, len(disagreements)),
        "primary_self_awareness": _confidence_self_awareness(a_payload["events"]),
        "candidate_self_awareness": _confidence_self_awareness(b_payload["events"]),
        "confidence_gap_bins": _gap_bin_summary(disagreements),
        "candidate_switch_threshold_sweep": [_threshold_policy(rows, value) for value in THRESHOLDS],
        "interpretation_warning": (
            "GT-anchored diagnostic only. Raw confidence scales across separately trained models may be uncalibrated. "
            "Do not report this as deployable routing accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, help="IGEM1 event JSON")
    parser.add_argument("--candidate", action="append", required=True, help="Candidate event JSON; repeatable")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    primary = _load_event(args.primary)
    candidates = [_load_event(path) for path in args.candidate]
    if primary.get("model") != "IGEM1":
        raise RuntimeError(f"Primary dikunci ke IGEM1, dapat {primary.get('model')}")
    result = {
        "protocol": PAIR_PROTOCOL,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "primary": primary["model"],
        "pairs": [compare_pair(primary, candidate) for candidate in candidates],
        "thresholds_descriptive_not_frozen_model_selection": list(THRESHOLDS),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", output)
    for pair in result["pairs"]:
        print("\nPAIR", pair["primary"], "vs", pair["candidate"])
        print("disagreements:", pair["disagreement_targets"])
        print("resolvable:", pair["resolvable_disagreements_exactly_one_correct"])
        print("higher-conf picks correct expert when resolvable:", f"{pair['higher_conf_correct_expert_rate_when_exactly_one_correct']:.2%}")
        print("higher-conf accuracy all disagreements:", f"{pair['higher_conf_accuracy_on_all_disagreements']:.2%}")
        print("primary accuracy disagreements:", f"{pair['primary_accuracy_on_disagreements']:.2%}")
        print("oracle disagreement accuracy:", f"{pair['oracle_accuracy_on_disagreements']:.2%}")


if __name__ == "__main__":
    main()
