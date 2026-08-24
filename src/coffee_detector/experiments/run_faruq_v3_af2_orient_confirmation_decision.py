"""Aggregate paired AF2 versus AF2_ORIENT validation results over three seeds."""

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
EPS = 1.0e-12


def _read(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def run_faruq_v3_af2_orient_confirmation_decision(
    af2_confirmation: str | Path,
    orient_seed42_result: str | Path,
    orient_seed123_result: str | Path,
    orient_seed2026_result: str | Path,
    output: str | Path,
) -> dict:
    baseline = _read(af2_confirmation, "Konfirmasi AF2")
    if (
        baseline.get("protocol")
        != "faruq-v3-af2-igem-paired-validation-confirmation-v1"
        or baseline.get("seeds") != list(SEEDS)
        or baseline.get("test_images_accessed") is not False
        or baseline.get("test_opened") is not False
        or baseline.get("decisions", {}).get("AF2", {}).get("decision") != "PASS"
    ):
        raise RuntimeError("Baseline AF2 tiga-seed tidak kompatibel")
    paths = {
        42: orient_seed42_result,
        123: orient_seed123_result,
        2026: orient_seed2026_result,
    }
    per_seed = {}
    for seed, path in paths.items():
        result = _read(path, f"AF2_ORIENT seed {seed}")
        if (
            result.get("format") != "coffee_detector.af2_iso.arm_result.v1"
            or result.get("arm") != "AF2_ORIENT"
            or int(result.get("seed", -1)) != seed
            or result.get("evaluation_split") != "val"
            or result.get("test_images_accessed") is not False
        ):
            raise RuntimeError(f"Hasil AF2_ORIENT seed {seed} tidak kompatibel")
        per_seed[str(seed)] = {
            "AF2": _metrics(baseline["per_seed"][str(seed)]["AF2"]),
            "AF2_ORIENT": _metrics(result),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "d0_checkpoint_sha256": result["initial_d0_checkpoint_sha256"],
        }

    aggregate = {}
    for metric in METRICS:
        controls = [per_seed[str(seed)]["AF2"][metric] for seed in SEEDS]
        candidates = [per_seed[str(seed)]["AF2_ORIENT"][metric] for seed in SEEDS]
        deltas = [candidate - control for control, candidate in zip(controls, candidates)]
        aggregate[metric] = {
            "af2_mean": statistics.fmean(controls),
            "af2_orient_mean": statistics.fmean(candidates),
            "delta_mean": statistics.fmean(deltas),
            "delta_std": statistics.stdev(deltas),
            "delta_min": min(deltas),
            "improved_seeds": sum(delta > 0 for delta in deltas),
            "deltas": dict(zip((str(seed) for seed in SEEDS), deltas)),
        }
    criteria = {
        "macro_mean_not_lower": aggregate["macro_map50_95"]["delta_mean"] >= -EPS,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_gain_at_least_0_5_point": aggregate["bottom3_class_map50_95"]["delta_mean"] >= 0.005 - EPS,
        "bottom3_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["improved_seeds"] >= 2,
        "worst_mean_not_lower": aggregate["worst_class_map50_95"]["delta_mean"] >= -EPS,
        "worst_improved_at_least_2_of_3": aggregate["worst_class_map50_95"]["improved_seeds"] >= 2,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.af2_orient.paired_confirmation.v1",
        "seeds": list(SEEDS),
        "evaluation_split": "val",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "next": "RETAIN_AF2_ORIENT" if decision == "PASS" else "RETAIN_ORIGINAL_AF2",
        "training_executed_by_decision": False,
        "test_images_accessed": False,
        "test_opened": False,
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide paired AF2_ORIENT confirmation")
    parser.add_argument("--af2-confirmation", required=True)
    parser.add_argument("--orient-seed42-result", required=True)
    parser.add_argument("--orient-seed123-result", required=True)
    parser.add_argument("--orient-seed2026-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_faruq_v3_af2_orient_confirmation_decision(
        args.af2_confirmation,
        args.orient_seed42_result,
        args.orient_seed123_result,
        args.orient_seed2026_result,
        args.output,
    )


if __name__ == "__main__":
    main()
