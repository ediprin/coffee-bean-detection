"""Aggregate paired AF2 versus AF2CT30 validation results across three seeds."""

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


def run_faruq_v3_af2_continuation_decision(
    af2_confirmation: str | Path,
    seed42_evidence: str | Path,
    seed123_result: str | Path,
    seed2026_result: str | Path,
    output: str | Path,
) -> dict:
    confirmation = _read(af2_confirmation, "Konfirmasi AF2")
    if (
        confirmation.get("protocol")
        != "faruq-v3-af2-igem-paired-validation-confirmation-v1"
        or confirmation.get("seeds") != list(SEEDS)
        or confirmation.get("test_images_accessed") is not False
        or confirmation.get("test_opened") is not False
        or confirmation.get("decisions", {}).get("AF2", {}).get("decision") != "PASS"
    ):
        raise RuntimeError("Konfirmasi AF2 tidak kompatibel")
    evidence = _read(seed42_evidence, "Evidence AF2FT30 seed 42")
    if (
        evidence.get("format") != "coffee_detector.af2cal.frozen_evidence.v1"
        or evidence.get("seed") != 42
        or evidence.get("test_opened") is not False
    ):
        raise RuntimeError("Evidence seed 42 tidak kompatibel")
    frozen_seed42_af2 = _metrics(evidence["values"]["AF2"])
    confirmed_seed42_af2 = _metrics(confirmation["per_seed"]["42"]["AF2"])
    if any(
        abs(frozen_seed42_af2[name] - confirmed_seed42_af2[name]) > EPS
        for name in METRICS
    ):
        raise RuntimeError("Baseline AF2 seed 42 pada evidence dan konfirmasi berbeda")
    results = {
        123: _read(seed123_result, "Hasil seed 123"),
        2026: _read(seed2026_result, "Hasil seed 2026"),
    }
    per_seed = {
        "42": {
            "AF2": frozen_seed42_af2,
            "AF2CT30": _metrics(evidence["values"]["AF2FT30"]),
            "source": "frozen_AF2FT30_evidence",
        }
    }
    for seed, result in results.items():
        if (
            result.get("format") != "coffee_detector.af2_continuation.arm_result.v1"
            or result.get("seed") != seed
            or result.get("test_images_accessed") is not False
        ):
            raise RuntimeError(f"Hasil continuation seed {seed} tidak kompatibel")
        af2 = _metrics(confirmation["per_seed"][str(seed)]["AF2"])
        reported_baseline = _metrics(result["baseline_af2_metrics"])
        if any(abs(af2[name] - reported_baseline[name]) > EPS for name in METRICS):
            raise RuntimeError(f"Baseline AF2 seed {seed} berubah")
        per_seed[str(seed)] = {
            "AF2": af2,
            "AF2CT30": _metrics(result),
            "initial_checkpoint_sha256": result["initial_af2_checkpoint_sha256"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        }

    aggregate = {}
    for metric in METRICS:
        baseline = [per_seed[str(seed)]["AF2"][metric] for seed in SEEDS]
        candidate = [per_seed[str(seed)]["AF2CT30"][metric] for seed in SEEDS]
        deltas = [right - left for left, right in zip(baseline, candidate)]
        aggregate[metric] = {
            "af2_mean": statistics.fmean(baseline),
            "af2ct30_mean": statistics.fmean(candidate),
            "delta_mean": statistics.fmean(deltas),
            "delta_std": statistics.stdev(deltas),
            "delta_min": min(deltas),
            "improved_seeds": sum(delta > 0 for delta in deltas),
            "deltas": dict(zip((str(seed) for seed in SEEDS), deltas)),
        }
    criteria = {
        "macro_mean_gain_at_least_0_5_point": aggregate["macro_map50_95"]["delta_mean"] >= 0.005 - EPS,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_not_lower": aggregate["bottom3_class_map50_95"]["delta_mean"] >= -EPS,
        "bottom3_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["improved_seeds"] >= 2,
        "worst_mean_not_lower": aggregate["worst_class_map50_95"]["delta_mean"] >= -EPS,
        "worst_improved_at_least_2_of_3": aggregate["worst_class_map50_95"]["improved_seeds"] >= 2,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.af2_continuation.paired_confirmation.v1",
        "seeds": list(SEEDS),
        "evaluation_split": "val",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "next": "RETAIN_AF2_TWO_STAGE_PROTOCOL" if decision == "PASS" else "RETAIN_ORIGINAL_AF2",
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
    parser = argparse.ArgumentParser(description="Decide paired AF2 continuation confirmation")
    parser.add_argument("--af2-confirmation", required=True)
    parser.add_argument("--seed42-evidence", required=True)
    parser.add_argument("--seed123-result", required=True)
    parser.add_argument("--seed2026-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_faruq_v3_af2_continuation_decision(
        args.af2_confirmation,
        args.seed42_evidence,
        args.seed123_result,
        args.seed2026_result,
        args.output,
    )


if __name__ == "__main__":
    main()
