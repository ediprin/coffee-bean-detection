"""Secondary diagnostic: CPE0 vs CIR0 on pre-frozen hard-confusion families.

This audit is post-training and exploratory. It must not overturn the frozen
Circle-CPE screening decision. The hard-family set is read from the earlier
cross-model consensus JSON that existed before Circle-CPE results.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

EVENT_PROTOCOL = "faruq-v3-validation-object-events-v1"
CONSENSUS_PROTOCOL = "faruq-v3-cross-model-hard-confusion-consensus-v1"
PROTOCOL = "faruq-v3-circle-cpe-hard-confusion-reduction-v1"
EXPECTED_FROZEN_UNDIRECTED = 17


def _load_json(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _load_event(path: str | Path, expected_model: str) -> dict:
    payload = _load_json(path)
    if payload.get("protocol") != EVENT_PROTOCOL:
        raise RuntimeError(f"Protocol event tidak kompatibel: {path}")
    if payload.get("model") != expected_model:
        raise RuntimeError(f"Model event salah: expected={expected_model}, got={payload.get('model')}")
    if int(payload.get("seed", -1)) != 42 or payload.get("evaluation_split") != "val":
        raise RuntimeError(f"Event harus seed42 val: {path}")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError(f"Event menunjukkan akses test: {path}")
    if not isinstance(payload.get("events"), dict) or not payload["events"]:
        raise RuntimeError(f"Event kosong: {path}")
    return payload


def _frozen_families(consensus: dict) -> list[str]:
    if consensus.get("protocol") != CONSENSUS_PROTOCOL:
        raise RuntimeError("Consensus protocol tidak kompatibel")
    if int(consensus.get("seed", -1)) != 42 or consensus.get("evaluation_split") != "val":
        raise RuntimeError("Consensus harus seed42 val")
    if consensus.get("test_images_accessed") is not False or consensus.get("test_opened") is not False:
        raise RuntimeError("Consensus menunjukkan akses test")
    families = [
        str(row["family"])
        for row in consensus.get("undirected_families", [])
        if row.get("frozen_consensus_hard_family") is True
    ]
    if len(families) != EXPECTED_FROZEN_UNDIRECTED:
        raise RuntimeError(f"Frozen undirected family count berubah: {len(families)}")
    return families


def _family(row: dict) -> str | None:
    if not row.get("matched") or row.get("correct"):
        return None
    gt = row.get("gt_class_name")
    pred = row.get("pred_class_name")
    if gt is None or pred is None or gt == pred:
        return None
    return " <-> ".join(sorted((str(gt), str(pred))))


def _classification_error_count(events: dict[str, dict]) -> int:
    return sum(_family(row) is not None for row in events.values())


def run(cpe0_event, cir0_event, consensus_json, output) -> dict:
    cpe = _load_event(cpe0_event, "CPE0")
    cir = _load_event(cir0_event, "CIR0")
    consensus = _load_json(consensus_json)
    frozen = _frozen_families(consensus)
    frozen_set = set(frozen)

    cpe_events = cpe["events"]
    cir_events = cir["events"]
    if set(cpe_events) != set(cir_events):
        raise RuntimeError("Target universe CPE0 dan CIR0 berbeda")

    cpe_counter: Counter[str] = Counter()
    cir_counter: Counter[str] = Counter()
    transitions = Counter()

    for key in sorted(cpe_events):
        a, b = cpe_events[key], cir_events[key]
        fa, fb = _family(a), _family(b)
        a_hard = fa in frozen_set if fa is not None else False
        b_hard = fb in frozen_set if fb is not None else False
        if a_hard:
            cpe_counter[fa] += 1
        if b_hard:
            cir_counter[fb] += 1

        if a_hard and b.get("correct"):
            transitions["cpe_hard_to_cir_correct"] += 1
        elif a_hard and b_hard:
            transitions["cpe_hard_to_cir_hard"] += 1
        elif a_hard and b.get("matched") and not b.get("correct"):
            transitions["cpe_hard_to_cir_nonhard_class_error"] += 1
        elif a_hard and not b.get("matched"):
            transitions["cpe_hard_to_cir_unmatched"] += 1

        if (not a_hard) and b_hard:
            transitions["new_cir_hard_error"] += 1
        if a.get("correct") and b_hard:
            transitions["cpe_correct_to_cir_hard"] += 1

    cpe_hard = int(sum(cpe_counter.values()))
    cir_hard = int(sum(cir_counter.values()))
    rows = []
    for family in frozen:
        before = int(cpe_counter.get(family, 0))
        after = int(cir_counter.get(family, 0))
        rows.append({
            "family": family,
            "cpe0_errors": before,
            "cir0_errors": after,
            "delta_cir0_minus_cpe0": after - before,
            "reduction": before - after,
        })
    rows.sort(key=lambda r: (r["reduction"], r["cpe0_errors"], -r["cir0_errors"]), reverse=True)

    cpe_total = _classification_error_count(cpe_events)
    cir_total = _classification_error_count(cir_events)
    result = {
        "protocol": PROTOCOL,
        "stage": "secondary_posthoc_diagnostic",
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "screening_decision_remains": "STOP_CIRCLE_CPE",
        "cannot_override_frozen_screening": True,
        "hard_family_source": {
            "protocol": CONSENSUS_PROTOCOL,
            "frozen_before_circle_results": True,
            "undirected_family_count": len(frozen),
        },
        "object_matching": cpe.get("matching"),
        "classification_errors_iou50": {
            "CPE0": cpe_total,
            "CIR0": cir_total,
            "delta_CIR0_minus_CPE0": cir_total - cpe_total,
        },
        "frozen_hard_family_errors": {
            "CPE0": cpe_hard,
            "CIR0": cir_hard,
            "delta_CIR0_minus_CPE0": cir_hard - cpe_hard,
            "absolute_reduction": cpe_hard - cir_hard,
            "relative_reduction_vs_CPE0": ((cpe_hard - cir_hard) / cpe_hard) if cpe_hard else None,
            "share_of_classification_errors_CPE0": (cpe_hard / cpe_total) if cpe_total else None,
            "share_of_classification_errors_CIR0": (cir_hard / cir_total) if cir_total else None,
        },
        "paired_transitions": dict(transitions),
        "family_summary": {
            "improved": sum(r["reduction"] > 0 for r in rows),
            "unchanged": sum(r["reduction"] == 0 for r in rows),
            "worsened": sum(r["reduction"] < 0 for r in rows),
        },
        "families": rows,
        "interpretation_rule": (
            "Descriptive only: reduction concentrated in pre-frozen families is consistent with, but does not prove, "
            "improved fine-grained decision-boundary separation. This audit cannot promote CIR0 after the frozen screen rejected it."
        ),
    }

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", destination)
    print("CLASS ERRORS:", result["classification_errors_iou50"])
    print("FROZEN HARD ERRORS:", result["frozen_hard_family_errors"])
    print("PAIRED TRANSITIONS:", result["paired_transitions"])
    print("FAMILY SUMMARY:", result["family_summary"])
    print("\nTOP REDUCTIONS")
    for row in rows[:10]:
        print(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpe0-event", required=True)
    parser.add_argument("--cir0-event", required=True)
    parser.add_argument("--consensus-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.cpe0_event, args.cir0_event, args.consensus_json, args.output)


if __name__ == "__main__":
    main()
