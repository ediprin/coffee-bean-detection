"""Frozen confirmation decision for AF2FS vs one selective FFAB2 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SEEDS = (42, 123, 2026)
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _read(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def run_selective_decision(af2_paths, selective_paths, output: str | Path) -> dict:
    if len(af2_paths) != 3 or len(selective_paths) != 3:
        raise ValueError("Memerlukan tepat tiga seed per arm")
    controls = [_read(path) for path in af2_paths]
    candidates = [_read(path) for path in selective_paths]
    diagnostic_sha = None
    selected = None
    for seed, control, candidate in zip(SEEDS, controls, candidates):
        if control.get("format") != "coffee_detector.af2_ffa.from_start_arm_result.v1" or control.get("arm") != "AF2FS":
            raise RuntimeError("Control bukan AF2FS from-start")
        if candidate.get("format") != "coffee_detector.af2_ffa.selective_arm_result.v1" or candidate.get("arm") != "AF2FFASR1":
            raise RuntimeError("Candidate bukan AF2FFASR1")
        if int(control.get("seed")) != seed or int(candidate.get("seed")) != seed:
            raise RuntimeError("Urutan seed tidak cocok")
        if control.get("initial_d0_checkpoint_sha256") != candidate.get("initial_d0_checkpoint_sha256"):
            raise RuntimeError(f"Seed {seed} tidak mulai dari D0 identik")
        if control.get("test_images_accessed") is not False or candidate.get("test_images_accessed") is not False:
            raise RuntimeError("Test lock dilanggar")
        if diagnostic_sha is None:
            diagnostic_sha = candidate.get("diagnostic_sha256")
            selected = candidate.get("selected_candidate")
        elif diagnostic_sha != candidate.get("diagnostic_sha256") or selected != candidate.get("selected_candidate"):
            raise RuntimeError("Tiga seed tidak memakai diagnosis/candidate yang sama")

    aggregate = {}
    for metric in METRICS:
        left = [float(item["metrics"][metric]) for item in controls]
        right = [float(item["metrics"][metric]) for item in candidates]
        deltas = [b - a for a, b in zip(left, right)]
        aggregate[metric] = {
            "af2_mean": sum(left) / 3.0,
            "selective_mean": sum(right) / 3.0,
            "delta_mean": sum(deltas) / 3.0,
            "improved_seeds": sum(delta > 0.0 for delta in deltas),
            "deltas": {str(seed): delta for seed, delta in zip(SEEDS, deltas)},
        }
    macro = aggregate["macro_map50_95"]
    bottom3 = aggregate["bottom3_class_map50_95"]
    worst = aggregate["worst_class_map50_95"]
    criteria = {
        "macro_mean_gain_at_least_0_5pp": macro["delta_mean"] >= 0.005,
        "macro_improves_at_least_2_of_3": macro["improved_seeds"] >= 2,
        "bottom3_mean_gain_at_least_0_5pp": bottom3["delta_mean"] >= 0.005,
        "bottom3_improves_at_least_2_of_3": bottom3["improved_seeds"] >= 2,
        "worst_mean_not_lower": worst["delta_mean"] >= 0.0,
        "worst_improves_at_least_2_of_3": worst["improved_seeds"] >= 2,
    }
    passed = all(criteria.values())
    result = {
        "format": "coffee_detector.af2_ffa.selective_decision.v1",
        "comparison": "AF2FFASR1_vs_AF2FS",
        "seeds": list(SEEDS),
        "selected_candidate": selected,
        "diagnostic_sha256": diagnostic_sha,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": "PASS" if passed else "REJECT",
        "next": "RETAIN_SELECTIVE_FFAB2_FOR_INDEPENDENT_CONFIRMATION" if passed else "STOP_SELECTIVE_FFAB2_ROUTE",
        "claim_if_pass": "Selective FFAB2 improves AF2 under matched from-start retraining on validation; independent confirmation remains required",
        "test_opened": False,
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2 vs selective FFAB2 retraining")
    parser.add_argument("--af2", nargs=3, required=True)
    parser.add_argument("--selective", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_selective_decision(args.af2, args.selective, args.output)


if __name__ == "__main__":
    main()
