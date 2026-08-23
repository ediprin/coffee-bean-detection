"""Frozen three-seed decision for fair AF2 vs AF2+FFAB2 from-start training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
SEEDS = (42, 123, 2026)


def _read(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _metric(payload: dict, name: str) -> float:
    return float(payload["metrics"][name])


def run_faruq_v3_af2_ffa_from_start_decision(
    af2_results: list[str | Path] | tuple[str | Path, ...],
    ffab2_results: list[str | Path] | tuple[str | Path, ...],
    output: str | Path,
) -> dict:
    if len(af2_results) != 3 or len(ffab2_results) != 3:
        raise ValueError("Keputusan memerlukan tepat tiga seed per arm")
    controls = [_read(path) for path in af2_results]
    candidates = [_read(path) for path in ffab2_results]
    for seed, control, candidate in zip(SEEDS, controls, candidates):
        if control.get("format") != "coffee_detector.af2_ffa.from_start_arm_result.v1":
            raise RuntimeError("Format AF2FS salah")
        if candidate.get("format") != "coffee_detector.af2_ffa.from_start_arm_result.v1":
            raise RuntimeError("Format AF2FFAB2FS salah")
        if control.get("arm") != "AF2FS" or candidate.get("arm") != "AF2FFAB2FS":
            raise RuntimeError("Arm keputusan tidak cocok")
        if int(control.get("seed")) != seed or int(candidate.get("seed")) != seed:
            raise RuntimeError("Urutan/seed hasil tidak cocok")
        if control.get("initial_d0_checkpoint_sha256") != candidate.get("initial_d0_checkpoint_sha256"):
            raise RuntimeError(f"Seed {seed} tidak mulai dari D0 identik")
        if control.get("test_images_accessed") is not False or candidate.get("test_images_accessed") is not False:
            raise RuntimeError("Test lock dilanggar")

    aggregate = {}
    for metric in METRICS:
        control_values = [_metric(item, metric) for item in controls]
        candidate_values = [_metric(item, metric) for item in candidates]
        deltas = [b - a for a, b in zip(control_values, candidate_values)]
        aggregate[metric] = {
            "af2_mean": sum(control_values) / 3.0,
            "ffab2_mean": sum(candidate_values) / 3.0,
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
        "format": "coffee_detector.af2_ffa.from_start_decision.v1",
        "comparison": "AF2FFAB2FS_vs_AF2FS",
        "seeds": list(SEEDS),
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": "PASS" if passed else "REJECT",
        "next": "AUTHORIZE_DCT_EFFICIENCY_STAGE" if passed else "STOP_FFAB2_UPGRADE_CLAIM",
        "claim_if_pass": "FFAB2 improves AF2 under matched from-start training",
        "test_opened": False,
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide fair AF2 vs AF2+FFAB2 three-seed result")
    parser.add_argument("--af2", nargs=3, required=True)
    parser.add_argument("--ffab2", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_faruq_v3_af2_ffa_from_start_decision(args.af2, args.ffab2, args.output)


if __name__ == "__main__":
    main()
