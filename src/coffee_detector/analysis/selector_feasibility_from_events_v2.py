"""Statistically gated wrapper for the GT-anchored selector feasibility audit.

Frozen before reading selector results:
- the raw-confidence signal is SUPPORTED only when the 95% Wilson interval for
  choosing the correct expert on resolvable disagreements has lower bound > 0.50;
- and the zero-threshold candidate-switch rule has positive net correct delta
  versus the IGEM1 primary on the full GT-anchored target universe.

This remains validation-only and GT-anchored; SUPPORTED does not mean deployable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from coffee_detector.analysis.selector_feasibility_from_events import (
    THRESHOLDS,
    _load_event,
    compare_pair,
)

PROTOCOL = "faruq-v3-gt-anchored-selector-feasibility-v2"


def _wilson95(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _add_frozen_gate(pair: dict) -> dict:
    n = int(pair["resolvable_disagreements_exactly_one_correct"])
    rate = float(pair["higher_conf_correct_expert_rate_when_exactly_one_correct"])
    successes = int(round(rate * n))
    low, high = _wilson95(successes, n)
    zero_rule = next(
        row for row in pair["candidate_switch_threshold_sweep"] if float(row["threshold"]) == 0.0
    )
    supported = low > 0.50 and int(zero_rule["net_correct_delta_vs_primary"]) > 0
    pair["higher_conf_correct_expert_successes"] = successes
    pair["higher_conf_correct_expert_wilson95_low"] = low
    pair["higher_conf_correct_expert_wilson95_high"] = high
    pair["zero_threshold_net_correct_delta_vs_primary"] = int(
        zero_rule["net_correct_delta_vs_primary"]
    )
    pair["frozen_confidence_signal_gate"] = {
        "criterion_1": "Wilson95 lower bound for correct-expert selection > 0.50",
        "criterion_1_pass": low > 0.50,
        "criterion_2": "zero-threshold candidate-switch net correct delta vs IGEM1 > 0",
        "criterion_2_pass": int(zero_rule["net_correct_delta_vs_primary"]) > 0,
        "decision": "SUPPORTED_FOR_PREDICTION_PAIR_AUDIT" if supported else "NOT_CONFIRMED",
    }
    return pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    primary = _load_event(args.primary)
    if primary.get("model") != "IGEM1":
        raise RuntimeError(f"Primary dikunci ke IGEM1, dapat {primary.get('model')}")
    candidates = [_load_event(path) for path in args.candidate]
    pairs = [_add_frozen_gate(compare_pair(primary, candidate)) for candidate in candidates]
    result = {
        "protocol": PROTOCOL,
        "seed": 42,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "primary": "IGEM1",
        "frozen_before_results": True,
        "thresholds_descriptive_not_frozen_model_selection": list(THRESHOLDS),
        "pairs": pairs,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", output)
    for pair in pairs:
        gate = pair["frozen_confidence_signal_gate"]
        print("\n", pair["primary"], "vs", pair["candidate"])
        print(
            "correct-expert rate:",
            f"{pair['higher_conf_correct_expert_rate_when_exactly_one_correct']:.2%}",
            "Wilson95:",
            f"[{pair['higher_conf_correct_expert_wilson95_low']:.2%}, {pair['higher_conf_correct_expert_wilson95_high']:.2%}]",
        )
        print("zero-threshold net correct delta:", pair["zero_threshold_net_correct_delta_vs_primary"])
        print("DECISION:", gate["decision"])


if __name__ == "__main__":
    main()
