"""Frozen three-seed decision for AF2-parent-preserving FFAB2 residual training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (42, 123, 2026)
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")

# Frozen before the parent-preserving runs. Same upgrade gate as the direct
# from-start Stage-1 so we do not weaken the claim after seeing prior failures.
GATE = {
    "macro_mean_gain_min": 0.005,
    "macro_improved_seeds_min": 2,
    "bottom3_mean_gain_min": 0.005,
    "bottom3_improved_seeds_min": 2,
    "worst_mean_delta_min": 0.0,
    "worst_improved_seeds_min": 2,
}


def _read(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _normalize_parent(paths) -> dict[int, dict[str, Any]]:
    result = {}
    for path in paths:
        payload = _read(path)
        if payload.get("format") != "coffee_detector.af2_ffa.from_start_arm_result.v1":
            raise RuntimeError("Parent result format salah")
        if payload.get("arm") != "AF2FS":
            raise RuntimeError("Parent arm harus AF2FS")
        seed = int(payload.get("seed", -1))
        if seed not in SEEDS or seed in result:
            raise RuntimeError(f"Parent seed invalid/duplikat: {seed}")
        if payload.get("evaluation_split") != "val" or payload.get("test_images_accessed") is not False:
            raise RuntimeError("Parent harus validation-only dan test terkunci")
        result[seed] = payload
    if tuple(sorted(result)) != SEEDS:
        raise RuntimeError(f"Parent harus tepat seed {SEEDS}")
    return result


def _normalize_candidate(paths, arm: str) -> dict[int, dict[str, Any]]:
    result = {}
    for path in paths:
        payload = _read(path)
        if payload.get("format") != "coffee_detector.af2_ffa.parent_preserving_arm_result.v1":
            raise RuntimeError("Candidate result format salah")
        if payload.get("arm") != arm:
            raise RuntimeError(f"Candidate arm harus {arm}")
        if payload.get("parent_frozen") is not True or payload.get("trainable_scope") != "ffab_adapters_only":
            raise RuntimeError("Candidate tidak membuktikan frozen-parent adapter-only training")
        seed = int(payload.get("seed", -1))
        if seed not in SEEDS or seed in result:
            raise RuntimeError(f"Candidate seed invalid/duplikat: {seed}")
        if payload.get("evaluation_split") != "val" or payload.get("test_images_accessed") is not False:
            raise RuntimeError("Candidate harus validation-only dan test terkunci")
        result[seed] = payload
    if tuple(sorted(result)) != SEEDS:
        raise RuntimeError(f"Candidate harus tepat seed {SEEDS}")
    return result


def _metric_row(parent, candidate, metric: str) -> dict[str, Any]:
    parent_values = [float(parent[s]["metrics"][metric]) for s in SEEDS]
    candidate_values = [float(candidate[s]["metrics"][metric]) for s in SEEDS]
    deltas = [right - left for left, right in zip(parent_values, candidate_values)]
    return {
        "parent_mean": float(np.mean(parent_values)),
        "candidate_mean": float(np.mean(candidate_values)),
        "delta_mean": float(np.mean(deltas)),
        "parent_sd": float(np.std(parent_values, ddof=1)),
        "candidate_sd": float(np.std(candidate_values, ddof=1)),
        "improved_seeds": int(sum(delta > 0.0 for delta in deltas)),
        "deltas": {str(seed): float(delta) for seed, delta in zip(SEEDS, deltas)},
        "parent_values": {str(seed): float(value) for seed, value in zip(SEEDS, parent_values)},
        "candidate_values": {str(seed): float(value) for seed, value in zip(SEEDS, candidate_values)},
    }


def run_parent_decision(parent_paths, candidate_paths, output: str | Path, zero_paths=None) -> dict:
    parent = _normalize_parent(parent_paths)
    candidate = _normalize_candidate(candidate_paths, "AF2FFAPR1")
    for seed in SEEDS:
        expected = str(parent[seed].get("checkpoint_sha256", ""))
        if candidate[seed].get("parent_checkpoint_sha256") != expected:
            raise RuntimeError(f"Seed {seed}: candidate tidak memakai AF2FS parent yang cocok")

    aggregate = {metric: _metric_row(parent, candidate, metric) for metric in METRICS}
    macro = aggregate["macro_map50_95"]
    bottom3 = aggregate["bottom3_class_map50_95"]
    worst = aggregate["worst_class_map50_95"]
    criteria = {
        "macro_mean_gain_at_least_0_5pp": macro["delta_mean"] >= GATE["macro_mean_gain_min"],
        "macro_improves_at_least_2_of_3": macro["improved_seeds"] >= GATE["macro_improved_seeds_min"],
        "bottom3_mean_gain_at_least_0_5pp": bottom3["delta_mean"] >= GATE["bottom3_mean_gain_min"],
        "bottom3_improves_at_least_2_of_3": bottom3["improved_seeds"] >= GATE["bottom3_improved_seeds_min"],
        "worst_mean_not_lower": worst["delta_mean"] >= GATE["worst_mean_delta_min"],
        "worst_improves_at_least_2_of_3": worst["improved_seeds"] >= GATE["worst_improved_seeds_min"],
    }
    passed = all(criteria.values())

    zero_control = None
    if zero_paths:
        zero = _normalize_candidate(zero_paths, "AF2FFAPR0")
        zero_control = {}
        for metric in METRICS:
            zero_control[metric] = _metric_row(parent, zero, metric)
        for seed in SEEDS:
            if zero[seed].get("parent_checkpoint_sha256") != parent[seed].get("checkpoint_sha256"):
                raise RuntimeError(f"Seed {seed}: zero control parent mismatch")

    result = {
        "format": "coffee_detector.af2_ffa.parent_preserving_decision.v1",
        "comparison": "AF2FFAPR1_vs_frozen_AF2FS_parent",
        "seeds": list(SEEDS),
        "gate": GATE,
        "aggregate": aggregate,
        "criteria": criteria,
        "zero_control": zero_control,
        "decision": "PASS" if passed else "REJECT",
        "next": (
            "RETAIN_PARENT_PRESERVING_FFAB2_ON_DEVELOPMENT_VALIDATION"
            if passed
            else "STOP_PARENT_PRESERVING_FFAB2_ROUTE"
        ),
        "claim_if_pass": (
            "With AF2 frozen, a classification-only FFAB2 residual improves the matched AF2FS parents "
            "under the frozen three-seed development-validation protocol."
        ),
        "interpretation_boundary": (
            "This follow-up was motivated after inspecting development validation. A PASS is therefore "
            "development-validation evidence, not an independent generalization/test confirmation."
        ),
        "test_opened": False,
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide frozen-parent AF2 + FFAB2 three-seed result")
    parser.add_argument("--parent", nargs=3, required=True)
    parser.add_argument("--candidate", nargs=3, required=True)
    parser.add_argument("--zero", nargs=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_parent_decision(args.parent, args.candidate, args.output, zero_paths=args.zero)


if __name__ == "__main__":
    main()
