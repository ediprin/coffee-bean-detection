from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from coffee_detector.wav1_factorization import TRAIN_ARMS


HEADLINES = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _read(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", payload)
    if not isinstance(metrics, dict):
        raise ValueError("Payload tidak memiliki metrics")
    return metrics


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1.0e-12:
        return None
    return numerator / denominator


def _per_class_correlation(
    base: dict[str, Any], reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    b = base.get("map50_95_by_class") or {}
    w = reference.get("map50_95_by_class") or {}
    c = candidate.get("map50_95_by_class") or {}
    names = sorted(set(b) & set(w) & set(c))
    if len(names) < 2:
        return {"classes": len(names), "pearson_delta_correlation": None}
    ref_delta = np.asarray([float(w[name]) - float(b[name]) for name in names])
    cand_delta = np.asarray([float(c[name]) - float(b[name]) for name in names])
    if np.std(ref_delta) < 1.0e-12 or np.std(cand_delta) < 1.0e-12:
        correlation = None
    else:
        correlation = float(np.corrcoef(ref_delta, cand_delta)[0, 1])
    return {
        "classes": len(names),
        "pearson_delta_correlation": correlation,
        "reference_delta_by_class": {
            name: float(w[name]) - float(b[name]) for name in names
        },
        "candidate_delta_by_class": {
            name: float(c[name]) - float(b[name]) for name in names
        },
    }


def run_factorization_report(
    d0ft_result: str | Path,
    wav1_result: str | Path,
    arm_result_paths: list[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    d0_payload = _read(d0ft_result)
    wav_payload = _read(wav1_result)
    d0 = _metrics(d0_payload)
    wav = _metrics(wav_payload)
    for key in HEADLINES:
        if key not in d0 or key not in wav:
            raise ValueError(f"Reference kehilangan metric {key}")

    arms: dict[str, dict[str, Any]] = {}
    for path in arm_result_paths:
        payload = _read(path)
        if payload.get("test_images_accessed") not in (None, False):
            raise RuntimeError(f"Result mengekspos test: {path}")
        arm = str(payload.get("arm", ""))
        if arm not in TRAIN_ARMS:
            raise ValueError(f"Arm result tidak dikenal: {arm}")
        if int(payload.get("seed", 42)) != 42:
            raise ValueError("Mechanism report hanya menerima seed-42 screening")
        arms[arm] = payload
    if set(arms) != set(TRAIN_ARMS):
        raise ValueError(f"Result harus lengkap untuk {TRAIN_ARMS}; diterima {tuple(arms)}")

    reference_gain = {key: float(wav[key]) - float(d0[key]) for key in HEADLINES}
    rows = []
    for arm in TRAIN_ARMS:
        metrics = _metrics(arms[arm])
        gain = {key: float(metrics[key]) - float(d0[key]) for key in HEADLINES}
        preservation = {
            key: _safe_ratio(gain[key], reference_gain[key]) for key in HEADLINES
        }
        class_pattern = _per_class_correlation(d0, wav, metrics)
        rows.append(
            {
                "arm": arm,
                "metrics": {key: float(metrics[key]) for key in HEADLINES},
                "gain_vs_d0ft": gain,
                "wav1_gain_preservation": preservation,
                "per_class_pattern": class_pattern,
            }
        )

    payload = {
        "format": "coffee_detector.wav1_factorization.report.v1",
        "decision": "MECHANISTIC_REVIEW_REQUIRED",
        "training_authorized": False,
        "test_opened": False,
        "references": {
            "d0ft": {key: float(d0[key]) for key in HEADLINES},
            "wav1": {key: float(wav[key]) for key in HEADLINES},
            "wav1_gain_vs_d0ft": reference_gain,
        },
        "arms": rows,
        "interpretation_boundary": [
            "The report quantifies explanatory preservation; it does not declare a new best model.",
            "No arm is authorized for extra seeds by this report alone.",
            "A separate paired-confirmation protocol must be frozen before any seed-123/2026 follow-up.",
            "Faruq locked test remains closed.",
        ],
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize WAV1 mechanism-factorization screen")
    parser.add_argument("--d0ft-result", required=True)
    parser.add_argument("--wav1-result", required=True)
    parser.add_argument("--arm-result", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run_factorization_report(
        args.d0ft_result, args.wav1_result, args.arm_result, args.output
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
