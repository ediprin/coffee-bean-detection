from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
TARGET_CLASS = "kulit_tanduk_ukuran_kecil"


def run_faruq_v3_af2_rcc_decision(
    arm_result: str | Path,
    output: str | Path,
) -> dict:
    source = Path(arm_result).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("format") != "coffee_detector.af2_rcc.arm_result.v1"
        or payload.get("arm") != "AF2RCC1"
        or int(payload.get("seed", -1)) != 42
        or payload.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Result AF2RCC1 tidak memenuhi kontrak screening")
    baseline, candidate = payload["baseline_metrics"], payload["metrics"]
    deltas = {metric: float(candidate[metric]) - float(baseline[metric]) for metric in METRICS}
    baseline_by_class = baseline["map50_95_by_class"]
    candidate_by_class = candidate["map50_95_by_class"]
    if set(baseline_by_class) != set(candidate_by_class) or len(candidate_by_class) != 21:
        raise RuntimeError("Per-class validation tidak lengkap atau tidak sejajar")
    target_delta = float(candidate_by_class[TARGET_CLASS]) - float(
        baseline_by_class[TARGET_CLASS]
    )
    improved = sum(deltas[metric] > 0.0 for metric in METRICS)
    criteria = {
        "macro_drop_no_more_than_0_1_point": deltas["macro_map50_95"] >= -0.001,
        "bottom3_not_lower": deltas["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_0_5_point": deltas["worst_class_map50_95"] >= -0.005,
        "at_least_two_headline_metrics_improve": improved >= 2,
        "target_class_drop_no_more_than_0_5_point": target_delta >= -0.005,
        "all_21_classes_present": len(candidate_by_class) == 21,
        "test_not_opened": payload["test_images_accessed"] is False,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    result = {
        "format": "coffee_detector.af2_rcc.seed42_decision.v1",
        "baseline": {metric: float(baseline[metric]) for metric in METRICS},
        "candidate": {metric: float(candidate[metric]) for metric in METRICS},
        "deltas": deltas,
        "target_class": TARGET_CLASS,
        "target_class_delta": target_delta,
        "improved_headline_metrics": improved,
        "criteria": criteria,
        "decision": decision,
        "next": "AUTHORIZE_PAIRED_THREE_SEED" if decision == "PASS" else "STOP_AF2_RCC",
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2RCC1 seed-42 decision")
    parser.add_argument("--arm-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_faruq_v3_af2_rcc_decision(args.arm_result, args.output)


if __name__ == "__main__":
    main()
