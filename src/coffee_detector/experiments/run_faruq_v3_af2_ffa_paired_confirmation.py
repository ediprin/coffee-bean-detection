"""Paired three-seed confirmation for AF2FFAB2 versus AF2FFA0."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 123, 2026)
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _read_result(path: str | Path, expected_arm: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        payload.get("format") != "coffee_detector.af2_ffa.arm_result.v1"
        or payload.get("arm") != expected_arm
        or int(payload.get("seed", -1)) not in SEEDS
    ):
        raise RuntimeError(f"Kontrak result {expected_arm} tidak valid: {source}")
    if payload.get("test_images_accessed") is not False:
        raise RuntimeError(f"Result mengakses test: {source}")
    missing = set(METRICS).difference(payload.get("metrics", {}))
    if missing:
        raise RuntimeError(f"Result kehilangan metrik {sorted(missing)}: {source}")
    return payload


def _load(paths: tuple[str | Path, ...], arm: str) -> dict[int, dict]:
    if len(paths) != 3:
        raise ValueError(f"{arm} memerlukan tepat tiga result")
    output = {}
    for path in paths:
        payload = _read_result(path, arm)
        seed = int(payload["seed"])
        if seed in output:
            raise RuntimeError(f"Seed {seed} duplikat untuk {arm}")
        output[seed] = payload
    if set(output) != set(SEEDS):
        raise RuntimeError(f"Seed {arm} tidak lengkap: {sorted(output)}")
    return output


def run_faruq_v3_af2_ffa_paired_confirmation(
    control_results: tuple[str | Path, ...],
    candidate_results: tuple[str | Path, ...],
    output: str | Path,
) -> dict:
    controls = _load(control_results, "AF2FFA0")
    candidates = _load(candidate_results, "AF2FFAB2")
    per_seed = {}
    for seed in SEEDS:
        if (
            controls[seed].get("initial_af2_checkpoint_sha256")
            != candidates[seed].get("initial_af2_checkpoint_sha256")
        ):
            raise RuntimeError(f"Pasangan seed {seed} tidak memakai AF2 yang sama")
        per_seed[str(seed)] = {
            "AF2FFA0": {
                metric: float(controls[seed]["metrics"][metric])
                for metric in METRICS
            },
            "AF2FFAB2": {
                metric: float(candidates[seed]["metrics"][metric])
                for metric in METRICS
            },
        }

    aggregate = {}
    for metric in METRICS:
        control = [per_seed[str(seed)]["AF2FFA0"][metric] for seed in SEEDS]
        candidate = [per_seed[str(seed)]["AF2FFAB2"][metric] for seed in SEEDS]
        deltas = [right - left for left, right in zip(control, candidate)]
        aggregate[metric] = {
            "control_mean": statistics.fmean(control),
            "control_std": statistics.stdev(control),
            "candidate_mean": statistics.fmean(candidate),
            "candidate_std": statistics.stdev(candidate),
            "delta_mean": statistics.fmean(deltas),
            "delta_std": statistics.stdev(deltas),
            "delta_min": min(deltas),
            "improved_seeds": sum(delta > 0.0 for delta in deltas),
            "noninferior_seeds_at_0_1_point": sum(delta >= -0.001 for delta in deltas),
            "deltas": {str(seed): delta for seed, delta in zip(SEEDS, deltas)},
        }

    criteria = {
        "macro_mean_drop_no_more_than_0_1_point": aggregate["macro_map50_95"][
            "delta_mean"
        ]
        >= -0.001,
        "macro_noninferior_at_least_2_of_3": aggregate["macro_map50_95"][
            "noninferior_seeds_at_0_1_point"
        ]
        >= 2,
        "bottom3_mean_positive": aggregate["bottom3_class_map50_95"][
            "delta_mean"
        ]
        > 0.0,
        "bottom3_improved_at_least_2_of_3": aggregate[
            "bottom3_class_map50_95"
        ]["improved_seeds"]
        >= 2,
        "worst_mean_positive": aggregate["worst_class_map50_95"]["delta_mean"]
        > 0.0,
        "worst_improved_at_least_2_of_3": aggregate["worst_class_map50_95"][
            "improved_seeds"
        ]
        >= 2,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.af2_ffa.gradient_matched_paired_confirmation.v1",
        "comparison": "AF2FFAB2_vs_AF2FFA0",
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "next": (
            "RETAIN_AF2FFAB2_AS_VALIDATED_PARETO_REFINEMENT"
            if decision == "PASS"
            else "RETAIN_ORIGINAL_AF2"
        ),
        "training_executed": True,
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirm AF2FFAB2 across three seeds")
    parser.add_argument("--control-results", nargs=3, required=True)
    parser.add_argument("--candidate-results", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_faruq_v3_af2_ffa_paired_confirmation(
        tuple(args.control_results), tuple(args.candidate_results), args.output
    )


if __name__ == "__main__":
    main()
