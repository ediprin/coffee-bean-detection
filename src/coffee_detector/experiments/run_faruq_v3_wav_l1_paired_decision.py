"""Combine frozen seed42 with seed123/2026 WAV_L1 results and apply the pre-frozen gate."""

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
EXPECTED_SEED42 = {
    "macro_map50_95": 0.885720537714217,
    "bottom3_class_map50_95": 0.8399334705085897,
    "worst_class_map50_95": 0.8209474694929713,
}


def _read(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {metric: float(source[metric]) for metric in METRICS}


def _validate_seed42(payload: dict) -> dict[str, float]:
    if (
        payload.get("format") != "coffee_detector.wav_l1.seed42_reference.v1"
        or payload.get("arm") != "WAV_L1"
        or int(payload.get("seed", -1)) != 42
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Frozen WAV_L1 seed42 evidence tidak kompatibel")
    metrics = _metrics(payload)
    for name, expected in EXPECTED_SEED42.items():
        if abs(metrics[name] - expected) > 1e-12:
            raise RuntimeError(f"Frozen seed42 berubah: {name}={metrics[name]} != {expected}")
    return metrics


def _validate_candidate(payload: dict, seed: int) -> dict[str, float]:
    if (
        payload.get("format") != "coffee_detector.wav_l1_confirmation.seed_result.v1"
        or payload.get("arm") != "WAV_L1"
        or int(payload.get("seed", -1)) != seed
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
    ):
        raise RuntimeError(f"WAV_L1 seed {seed} result tidak kompatibel")
    return _metrics(payload)


def _validate_reference(payload: dict) -> dict[int, dict[str, float]]:
    if (
        payload.get("seeds") != [42, 123, 2026]
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
    ):
        raise RuntimeError("AF2/IGEM reference bukan tiga-seed validation-only evidence")
    result = {}
    for seed in SEEDS:
        row = payload.get("per_seed", {}).get(str(seed), {})
        if "D0FT" not in row:
            raise RuntimeError(f"D0FT seed {seed} hilang dari reference")
        result[seed] = _metrics(row["D0FT"])
    return result


def _aggregate(d0ft: dict[int, dict], candidate: dict[int, dict]) -> dict[str, dict]:
    result = {}
    for metric in METRICS:
        control = [float(d0ft[seed][metric]) for seed in SEEDS]
        values = [float(candidate[seed][metric]) for seed in SEEDS]
        deltas = [right - left for left, right in zip(control, values)]
        result[metric] = {
            "d0ft_mean": statistics.fmean(control),
            "d0ft_std": statistics.stdev(control),
            "wav_l1_mean": statistics.fmean(values),
            "wav_l1_std": statistics.stdev(values),
            "paired_delta_mean": statistics.fmean(deltas),
            "paired_delta_std": statistics.stdev(deltas),
            "paired_delta_min": min(deltas),
            "improved_seeds": sum(delta > 0.0 for delta in deltas),
            "deltas": {str(seed): delta for seed, delta in zip(SEEDS, deltas)},
        }
    return result


def _decision(aggregate: dict[str, dict]) -> tuple[dict[str, bool], str]:
    criteria = {
        "macro_gain_at_least_0_5_point": aggregate["macro_map50_95"]["paired_delta_mean"] >= 0.005,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_not_lower": aggregate["bottom3_class_map50_95"]["paired_delta_mean"] >= 0.0,
        "bottom3_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["improved_seeds"] >= 2,
        "worst_mean_drop_no_more_than_1_point": aggregate["worst_class_map50_95"]["paired_delta_mean"] >= -0.01,
    }
    return criteria, "PASS" if all(criteria.values()) else "FAIL"


def run_decision(
    seed42_evidence: str | Path,
    seed123_result: str | Path,
    seed2026_result: str | Path,
    af2_igem_reference: str | Path,
    output_path: str | Path,
) -> dict:
    seed42_payload = _read(seed42_evidence, "WAV_L1 seed42 evidence")
    seed123_payload = _read(seed123_result, "WAV_L1 seed123 result")
    seed2026_payload = _read(seed2026_result, "WAV_L1 seed2026 result")
    reference = _read(af2_igem_reference, "AF2/IGEM D0FT reference")

    candidate = {
        42: _validate_seed42(seed42_payload),
        123: _validate_candidate(seed123_payload, 123),
        2026: _validate_candidate(seed2026_payload, 2026),
    }
    d0ft = _validate_reference(reference)
    aggregate = _aggregate(d0ft, candidate)
    criteria, decision = _decision(aggregate)

    per_seed = {
        str(seed): {"D0FT": d0ft[seed], "WAV_L1": candidate[seed]}
        for seed in SEEDS
    }
    payload = {
        "format": "coffee_detector.wav_l1_confirmation.paired_decision.v1",
        "arm": "WAV_L1",
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "evaluation_split": "val",
        "test_opened": False,
        "test_images_accessed": False,
        "next_action": (
            "freeze_new_next_step_protocol_before_any_new_experiment"
            if decision == "PASS"
            else "stop_WAV_L1_confirmation_no_retuning_or_locked_test"
        ),
        "note": "Primary decision is WAV_L1 versus seed-matched D0FT. This is not a superiority test versus two-level WAV1.",
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute frozen WAV_L1 paired three-seed decision")
    parser.add_argument("--seed42-evidence", required=True)
    parser.add_argument("--seed123-result", required=True)
    parser.add_argument("--seed2026-result", required=True)
    parser.add_argument("--af2-igem-reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_decision(
        args.seed42_evidence,
        args.seed123_result,
        args.seed2026_result,
        args.af2_igem_reference,
        args.output,
    )


if __name__ == "__main__":
    main()
