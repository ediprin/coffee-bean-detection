from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.af2_refinement import TRAIN_ARMS


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
EPS = 1.0e-12


def _read(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {metric: float(source[metric]) for metric in METRICS}


def _af2_metrics(payload: dict) -> dict[str, float]:
    if "candidate" in payload and "AF2" in payload["candidate"]:
        return _metrics(payload["candidate"]["AF2"])
    if "values" in payload and "AF2" in payload["values"]:
        return _metrics(payload["values"]["AF2"])
    return _metrics(payload)


def run_af2_refinement_decision(
    output_root: str | Path,
    af2_result: str | Path,
    *,
    seed: int = 42,
) -> dict:
    if seed != 42:
        raise ValueError("Discovery decision dikunci pada seed 42")
    output_root = Path(output_root).expanduser().resolve()
    baseline_payload = _read(af2_result)
    if baseline_payload.get("test_images_accessed", False) is not False:
        raise RuntimeError("Evidence AF2 mengakses test")
    baseline = _af2_metrics(baseline_payload)

    candidates = {}
    for arm in TRAIN_ARMS:
        result = _read(output_root / "val_reports" / f"{arm}_seed{seed}_result.json")
        if (
            result.get("format") != "coffee_detector.af2_refinement.arm_result.v1"
            or result.get("arm") != arm
            or result.get("seed") != seed
            or result.get("evaluation_split") != "val"
            or result.get("test_images_accessed") is not False
        ):
            raise RuntimeError(f"Hasil {arm} tidak kompatibel")
        values = _metrics(result)
        delta = {metric: values[metric] - baseline[metric] for metric in METRICS}
        criteria = {
            # Deliberately unchanged from the original spectral-factorization
            # protocol: the follow-up does not move the success threshold after
            # seeing AF2POL/WAV1 discovery results.
            "macro_gain_at_least_0_5_point": delta["macro_map50_95"] >= 0.005 - EPS,
            "bottom3_not_lower": delta["bottom3_class_map50_95"] >= -EPS,
            "worst_drop_no_more_than_1_point": delta["worst_class_map50_95"] >= -0.01 - EPS,
        }
        candidates[arm] = {
            "metrics": values,
            "delta_vs_af2c": delta,
            "latency": result.get("latency", {}),
            "criteria": criteria,
            "tail_observation": {
                "bottom3_gain_points": 100.0 * delta["bottom3_class_map50_95"],
                "worst_gain_points": 100.0 * delta["worst_class_map50_95"],
            },
            "decision": "RETAIN" if all(criteria.values()) else "REJECT",
        }

    retained = [arm for arm, value in candidates.items() if value["decision"] == "RETAIN"]
    winner = None
    if retained:
        max_macro = max(candidates[arm]["metrics"]["macro_map50_95"] for arm in retained)
        near = [
            arm
            for arm in retained
            if candidates[arm]["metrics"]["macro_map50_95"] >= max_macro - 0.002 - EPS
        ]
        winner = sorted(
            near,
            key=lambda arm: (
                -candidates[arm]["metrics"]["bottom3_class_map50_95"],
                -candidates[arm]["metrics"]["worst_class_map50_95"],
                candidates[arm].get("latency", {}).get("median_ms", float("inf")),
                arm,
            ),
        )[0]

    decision = "PASS" if retained else "FAIL"
    result = {
        "format": "coffee_detector.af2_refinement.seed42_decision.v1",
        "seed": seed,
        "baseline": {
            "arm": "AF2C",
            "metrics": baseline,
            "source": str(Path(af2_result).expanduser().resolve()),
        },
        "candidates": candidates,
        "retained": retained,
        "winner": winner,
        "decision": decision,
        "next": (
            "AUTHORIZE_WINNER_PAIRED_CONFIRMATION"
            if decision == "PASS"
            else "KEEP_AF2C_AND_STOP"
        ),
        "test_opened": False,
    }
    path = output_root / "val_reports/af2_refinement_seed42_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2 radial/wavelet refinement")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--af2-result", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_af2_refinement_decision(
        args.output_root,
        args.af2_result,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
