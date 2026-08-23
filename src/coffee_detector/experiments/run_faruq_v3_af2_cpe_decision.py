from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def decide(control: dict, candidate: dict) -> dict:
    cm, ca = control["metrics"], candidate["metrics"]
    delta = {key: float(ca[key]) - float(cm[key]) for key in METRICS}
    eps = 1e-12
    safety = {
        "macro_not_below_minus_0_20pp": delta[METRICS[0]] + eps >= -0.002,
        "bottom3_drop_not_over_1pp": delta[METRICS[1]] + eps >= -0.010,
        "worst_drop_not_over_1pp": delta[METRICS[2]] + eps >= -0.010,
    }
    superiority = delta[METRICS[0]] + eps >= 0.002 and delta[METRICS[1]] + eps >= 0 and delta[METRICS[2]] + eps >= 0
    tail_pareto = (
        delta[METRICS[0]] + eps >= -0.001
        and delta[METRICS[1]] + eps >= 0.005
        and delta[METRICS[2]] + eps >= 0.010
    )
    retain = all(safety.values()) and (superiority or tail_pareto)
    return {
        "format": "coffee_detector.af2_cpe.seed42_decision.v1",
        "control": "AF2CPE0",
        "candidate": "AF2CPE5",
        "deltas_candidate_minus_control": delta,
        "safety_gates": safety,
        "routes": {"superiority": superiority, "tail_pareto": tail_pareto},
        "decision": "RETAIN" if retain else "REJECT",
        "test_images_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen AF2+CPE0 seed42 decision")
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    result = decide(load(args.control), load(args.candidate))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
