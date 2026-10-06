"""Aggregate the frozen three-seed DVF1 fuser confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coffee_detector.experiments.run_defectoscafeverde_dual_view_fusion import (
    FORMAT as DVF1_FORMAT,
    _json,
    _sha256,
)
from coffee_detector.experiments.run_defectoscafeverde_dvf1_confirmation import (
    FORMAT as ARM_FORMAT,
    METRICS,
    PROTOCOL,
)


FORMAT = "coffee_detector.defectoscafeverde.dvf1_confirmation.decision.v1"


def run_decision(
    seed42_result: str | Path,
    seed123_result: str | Path,
    seed2026_result: str | Path,
    output: str | Path,
) -> dict:
    paths = {
        42: Path(seed42_result).expanduser().resolve(),
        123: Path(seed123_result).expanduser().resolve(),
        2026: Path(seed2026_result).expanduser().resolve(),
    }
    rows = {seed: _json(path, f"Seed {seed} result") for seed, path in paths.items()}
    if (
        rows[42].get("format") != DVF1_FORMAT
        or rows[42].get("seed") != 42
        or rows[42].get("decision") != "PASS"
    ):
        raise RuntimeError("Seed-42 DVF1 result tidak valid")
    for seed in (123, 2026):
        row = rows[seed]
        if (
            row.get("format") != ARM_FORMAT
            or row.get("protocol") != PROTOCOL
            or row.get("seed") != seed
            or row.get("status") != "COMPLETE"
            or row.get("test_images_accessed") is not False
        ):
            raise RuntimeError(f"Seed {seed} confirmation tidak valid")
        if row["run_contract"]["dvf1_seed42_result_sha256"] != _sha256(paths[42]):
            raise RuntimeError(f"Seed {seed} tidak paired dengan seed 42")
    per_seed = {}
    for seed, row in rows.items():
        candidate = row["values"]["DVF1" if seed == 42 else "DVF1_CANDIDATE"]
        paper = row["values"]["PAPER_MAX_CONFIDENCE"]
        per_seed[str(seed)] = {
            "paper": {metric: paper[metric] for metric in METRICS},
            "DVF1": {metric: candidate[metric] for metric in METRICS},
            "delta": {metric: candidate[metric] - paper[metric] for metric in METRICS},
        }
    aggregate = {}
    for metric in METRICS:
        paper_values = np.asarray([per_seed[str(seed)]["paper"][metric] for seed in rows])
        candidate_values = np.asarray([per_seed[str(seed)]["DVF1"][metric] for seed in rows])
        deltas = candidate_values - paper_values
        aggregate[metric] = {
            "paper_mean": float(paper_values.mean()),
            "dvf1_mean": float(candidate_values.mean()),
            "dvf1_std": float(candidate_values.std(ddof=0)),
            "delta_mean": float(deltas.mean()),
            "delta_min": float(deltas.min()),
            "improved_seeds": int((deltas > 0).sum()),
            "nonnegative_seeds": int((deltas >= 0).sum()),
        }
    criteria = {
        "mean_physical_gain_at_least_0_5_point": aggregate["physical_pair_accuracy"]["delta_mean"] >= 0.005,
        "physical_improves_at_least_2_of_3": aggregate["physical_pair_accuracy"]["improved_seeds"] >= 2,
        "mean_macro_not_lower": aggregate["macro_class_accuracy"]["delta_mean"] >= 0.0,
        "macro_improves_at_least_2_of_3": aggregate["macro_class_accuracy"]["improved_seeds"] >= 2,
        "mean_bottom3_not_lower": aggregate["bottom3_class_accuracy"]["delta_mean"] >= 0.0,
        "bottom3_improves_at_least_2_of_3": aggregate["bottom3_class_accuracy"]["improved_seeds"] >= 2,
        "mean_worst_drop_no_more_than_2_points": aggregate["worst_class_accuracy"]["delta_mean"] >= -0.02,
        "all_reports_test_false": all(row.get("test_images_accessed") is False for row in rows.values()),
        "decision_training_not_executed": True,
    }
    passed = all(criteria.values())
    result = {
        "format": FORMAT,
        "protocol": PROTOCOL,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": "PASS" if passed else "FAIL",
        "next": "PROMOTE_DVF1_DUAL_VIEW_FUSION" if passed else "REPORT_DVF1_AS_EXPLORATORY_SEED42_ONLY",
        "claim_boundary": "reused-validation fuser-seed confirmation; no locked-test inference",
        "artifacts": {str(seed): {"path": str(path), "sha256": _sha256(path)} for seed, path in paths.items()},
        "training_executed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DVF1 three-seed confirmation decision")
    parser.add_argument("--seed42-result", required=True)
    parser.add_argument("--seed123-result", required=True)
    parser.add_argument("--seed2026-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_decision(
        args.seed42_result,
        args.seed123_result,
        args.seed2026_result,
        args.output,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
