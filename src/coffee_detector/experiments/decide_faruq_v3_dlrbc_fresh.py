from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from coffee_detector.experiments.run_faruq_v3_dlrbc_fresh_arm import ARMS, METRICS


MAX_SCREEN_DROP = 0.005


def _load(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def build_fresh_dlrbc_decision(
    result_paths: list[str | Path], output: str | Path
) -> dict[str, Any]:
    loaded = [_load(path) for path in result_paths]
    results = {payload["arm"]: payload for payload in loaded}
    if set(results) != set(ARMS):
        raise RuntimeError(f"Harus tersedia tepat {ARMS}; diterima {sorted(results)}")
    for arm, payload in results.items():
        if payload.get("seed") != 42:
            raise RuntimeError(f"{arm} bukan seed 42")
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"{arm} melanggar test lock")
        if payload.get("fresh_optimizer") is not True or payload.get("coffee_parent_checkpoint") is not None:
            raise RuntimeError(f"{arm} bukan fresh protocol")

    baseline = results["B0_FRESH"]
    control = results["LRLIN_FRESH"]
    candidate = results["DLRBC_FRESH"]
    versus_control = {
        metric: float(candidate["metrics"][metric] - control["metrics"][metric])
        for metric in METRICS
    }
    versus_native = {
        metric: float(candidate["metrics"][metric] - baseline["metrics"][metric])
        for metric in METRICS
    }
    improved = sum(value > 0.0 for value in versus_control.values())
    no_large_drop = all(value >= -MAX_SCREEN_DROP for value in versus_control.values())
    decision = "PROMOTE_TO_FRESH_3_SEED" if improved >= 2 and no_large_drop else "STOP_AFTER_SEED42"
    payload = {
        "format": "coffee_detector.dlrbc_fresh.seed42_decision.v1",
        "protocol": "faruq-v3-dlrbc-fresh-seed42-v1",
        "seed": 42,
        "values": {arm: results[arm]["metrics"] for arm in ARMS},
        "dlrbc_minus_matched_linear": versus_control,
        "dlrbc_minus_native": versus_native,
        "criteria": {
            "improves_at_least_two_headline_metrics_vs_matched_linear": improved >= 2,
            "no_headline_drop_over_0_5_point_vs_matched_linear": no_large_drop,
            "all_arms_fresh_optimizer": True,
            "no_coffee_parent_checkpoint": True,
            "test_not_opened": True,
        },
        "decision": decision,
        "next": (
            "RUN_FRESH_SEEDS_123_2026_FROM_OFFICIAL_YOLO26N"
            if decision == "PROMOTE_TO_FRESH_3_SEED"
            else "DO_NOT_RUN_MORE_SEEDS_OR_AF2_FACTORIAL"
        ),
        "training_executed": False,
        "test_images_accessed": False,
        "claim_limit": "seed-42 screen; matched linear control is the causal comparator",
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide fresh DLRBC seed-42 screen")
    parser.add_argument("--results", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_fresh_dlrbc_decision(args.results, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
