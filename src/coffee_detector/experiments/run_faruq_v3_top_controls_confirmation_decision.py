"""Aggregate the frozen three-seed confirmation of high seed-42 controls."""

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
COMPARISONS = {
    "FCT0": "STB1",
    "AF2R0": "AF2",
    "AF2R1": "AF2R0",
    "AF2CAL3": "AF2CT30",
}


def _read(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {metric: float(source[metric]) for metric in METRICS}


def _validate_common(payload: dict, label: str) -> None:
    if payload.get("test_images_accessed") is not False:
        raise RuntimeError(f"{label} tidak mempertahankan test lock")


def _load_results(
    paths: tuple[str | Path, ...], expected_format: str, expected_arm: str
) -> dict[int, dict[str, float]]:
    if len(paths) != 2:
        raise ValueError(f"{expected_arm} memerlukan tepat dua result seed 123/2026")
    results: dict[int, dict[str, float]] = {}
    for path in paths:
        payload = _read(path, f"Result {expected_arm}")
        _validate_common(payload, expected_arm)
        seed = int(payload.get("seed", -1))
        if payload.get("format") != expected_format or payload.get("arm") != expected_arm:
            raise RuntimeError(f"Kontrak result {expected_arm} tidak kompatibel: {path}")
        if seed not in (123, 2026) or seed in results:
            raise RuntimeError(f"Seed result {expected_arm} tidak valid/duplikat: {seed}")
        results[seed] = _metrics(payload)
    if set(results) != {123, 2026}:
        raise RuntimeError(f"Result {expected_arm} tidak lengkap: {sorted(results)}")
    return results


def _aggregate(per_seed: dict[str, dict[str, dict[str, float]]]) -> dict:
    output = {}
    for candidate, control in COMPARISONS.items():
        metric_rows = {}
        for metric in METRICS:
            left = [per_seed[str(seed)][control][metric] for seed in SEEDS]
            right = [per_seed[str(seed)][candidate][metric] for seed in SEEDS]
            deltas = [candidate_value - control_value for control_value, candidate_value in zip(left, right)]
            metric_rows[metric] = {
                "control": control,
                "control_mean": statistics.fmean(left),
                "control_std": statistics.stdev(left),
                "candidate_mean": statistics.fmean(right),
                "candidate_std": statistics.stdev(right),
                "delta_mean": statistics.fmean(deltas),
                "delta_std": statistics.stdev(deltas),
                "delta_min": min(deltas),
                "improved_seeds": sum(delta > 0.0 for delta in deltas),
                "deltas": dict(zip((str(seed) for seed in SEEDS), deltas)),
            }
        criteria = {
            "macro_gain_at_least_0_5_point": metric_rows["macro_map50_95"]["delta_mean"] >= 0.005,
            "macro_improved_at_least_2_of_3": metric_rows["macro_map50_95"]["improved_seeds"] >= 2,
            "bottom3_mean_not_lower": metric_rows["bottom3_class_map50_95"]["delta_mean"] >= 0.0,
            "bottom3_improved_at_least_2_of_3": metric_rows["bottom3_class_map50_95"]["improved_seeds"] >= 2,
            "worst_mean_drop_no_more_than_1_point": metric_rows["worst_class_map50_95"]["delta_mean"] >= -0.01,
        }
        output[candidate] = {
            "control": control,
            "metrics": metric_rows,
            "criteria": criteria,
            "decision": "PASS" if all(criteria.values()) else "FAIL",
        }
    return output


def run_faruq_v3_top_controls_confirmation_decision(
    stb_confirmation: str | Path,
    af2_confirmation: str | Path,
    af2_continuation_confirmation: str | Path,
    fct0_seed42_result: str | Path,
    af2r_seed42_evidence: str | Path,
    af2cal_seed42_evidence: str | Path,
    fct0_results: tuple[str | Path, ...],
    af2r0_results: tuple[str | Path, ...],
    af2r1_results: tuple[str | Path, ...],
    af2cal3_results: tuple[str | Path, ...],
    output: str | Path,
) -> dict:
    stb = _read(stb_confirmation, "Konfirmasi STB")
    af2 = _read(af2_confirmation, "Konfirmasi AF2")
    continuation = _read(af2_continuation_confirmation, "Konfirmasi continuation AF2")
    if stb.get("protocol") != "faruq-v3-stb-capacity-paired-confirmation-v1":
        raise RuntimeError("Summary STB tidak kompatibel")
    if af2.get("protocol") != "faruq-v3-af2-igem-paired-validation-confirmation-v1":
        raise RuntimeError("Summary AF2 tidak kompatibel")
    if continuation.get("format") != "coffee_detector.af2_continuation.paired_confirmation.v1":
        raise RuntimeError("Summary AF2 continuation tidak kompatibel")
    for label, payload in (("STB", stb), ("AF2", af2), ("AF2 continuation", continuation)):
        _validate_common(payload, label)
        if payload.get("test_opened") is not False or payload.get("seeds") != [42, 123, 2026]:
            raise RuntimeError(f"Summary {label} tidak memenuhi kontrak tiga-seed development")

    fct42 = _read(fct0_seed42_result, "FCT0 seed 42")
    af2r42 = _read(af2r_seed42_evidence, "Evidence AF2R seed 42")
    af2cal42 = _read(af2cal_seed42_evidence, "Evidence AF2CAL seed 42")
    if af2r42.get("format") != "coffee_detector.af2r.seed42_recovered_evidence.v1":
        raise RuntimeError("Evidence AF2R seed 42 tidak kompatibel")
    if af2cal42.get("format") != "coffee_detector.af2cal.frozen_evidence.v1":
        raise RuntimeError("Evidence AF2CAL seed 42 tidak kompatibel")
    if af2r42.get("test_opened") is not False or af2cal42.get("test_opened") is not False:
        raise RuntimeError("Evidence seed 42 tidak mempertahankan test lock")

    new = {
        "FCT0": _load_results(
            fct0_results, "coffee_detector.fct0_confirmation.arm_result.v1", "FCT0"
        ),
        "AF2R0": _load_results(
            af2r0_results, "coffee_detector.af2r.arm_result.v1", "AF2R0"
        ),
        "AF2R1": _load_results(
            af2r1_results, "coffee_detector.af2r.arm_result.v1", "AF2R1"
        ),
        "AF2CAL3": _load_results(
            af2cal3_results, "coffee_detector.af2cal.arm_result.v1", "AF2CAL3"
        ),
    }

    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed in SEEDS:
        key = str(seed)
        per_seed[key] = {
            "STB1": _metrics(stb["per_seed"][key]["STB1"]),
            "AF2": _metrics(af2["per_seed"][key]["AF2"]),
            "AF2CT30": _metrics(continuation["per_seed"][key]["AF2CT30"]),
        }
    per_seed["42"].update(
        {
            "FCT0": _metrics(fct42),
            "AF2R0": _metrics(af2r42["values"]["AF2R0"]),
            "AF2R1": _metrics(af2r42["values"]["AF2R1"]),
            "AF2CAL3": _metrics(af2cal42["values"]["AF2CAL3"]),
        }
    )
    for seed in (123, 2026):
        for arm in new:
            per_seed[str(seed)][arm] = new[arm][seed]

    decisions = _aggregate(per_seed)
    result = {
        "format": "coffee_detector.top_controls.paired_confirmation.v1",
        "seeds": list(SEEDS),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "per_seed": per_seed,
        "comparisons": decisions,
        "retained": [arm for arm, row in decisions.items() if row["decision"] == "PASS"],
        "note": "AF2FT30 is reused as AF2CT30; it is not retrained.",
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["summary"] = str(target)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide high seed-42 control confirmations")
    parser.add_argument("--stb-confirmation", required=True)
    parser.add_argument("--af2-confirmation", required=True)
    parser.add_argument("--af2-continuation-confirmation", required=True)
    parser.add_argument("--fct0-seed42-result", required=True)
    parser.add_argument("--af2r-seed42-evidence", required=True)
    parser.add_argument("--af2cal-seed42-evidence", required=True)
    parser.add_argument("--fct0-results", nargs=2, required=True)
    parser.add_argument("--af2r0-results", nargs=2, required=True)
    parser.add_argument("--af2r1-results", nargs=2, required=True)
    parser.add_argument("--af2cal3-results", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_faruq_v3_top_controls_confirmation_decision(
        args.stb_confirmation,
        args.af2_confirmation,
        args.af2_continuation_confirmation,
        args.fct0_seed42_result,
        args.af2r_seed42_evidence,
        args.af2cal_seed42_evidence,
        tuple(args.fct0_results),
        tuple(args.af2r0_results),
        tuple(args.af2r1_results),
        tuple(args.af2cal3_results),
        args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
