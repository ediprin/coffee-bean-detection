from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
STAGE_ARMS = {
    "stage1": ("AF2WIN", "AF2ORI", "AF2POL", "AF2SOFT", "AF2LUM"),
    "stage2": ("PCG1", "WAV1"),
    "global": ("AF2WIN", "AF2ORI", "AF2POL", "AF2SOFT", "AF2LUM", "PCG1", "WAV1"),
}
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


def run_spectral_decision(
    output_root: str | Path,
    af2_result: str | Path,
    *,
    stage: str,
    seed: int = 42,
) -> dict:
    if stage not in STAGE_ARMS or seed != 42:
        raise ValueError("Decision dikunci pada stage1/stage2/global dan seed 42")
    output_root = Path(output_root).expanduser().resolve()
    baseline_payload = _read(af2_result)
    baseline = _af2_metrics(baseline_payload)
    if baseline_payload.get("test_images_accessed", False) is not False:
        raise RuntimeError("Evidence AF2 mengakses test")
    if stage in {"stage2", "global"}:
        stage1 = _read(output_root / "val_reports/stage1_seed42_decision.json")
        if stage1.get("stage") != "stage1" or stage1.get("test_opened") is not False:
            raise RuntimeError("Stage 1 belum memiliki keputusan kompatibel")
    if stage == "global":
        stage2 = _read(output_root / "val_reports/stage2_seed42_decision.json")
        if stage2.get("stage") != "stage2" or stage2.get("test_opened") is not False:
            raise RuntimeError("Stage 2 belum memiliki keputusan kompatibel")

    candidates = {}
    for arm in STAGE_ARMS[stage]:
        result = _read(output_root / "val_reports" / f"{arm}_seed{seed}_result.json")
        if (
            result.get("format") != "coffee_detector.af2_spectral.arm_result.v1"
            or result.get("arm") != arm
            or result.get("seed") != seed
            or result.get("test_images_accessed") is not False
        ):
            raise RuntimeError(f"Hasil {arm} tidak kompatibel")
        values = _metrics(result)
        delta = {metric: values[metric] - baseline[metric] for metric in METRICS}
        criteria = {
            "macro_gain_at_least_0_5_point": delta["macro_map50_95"] >= 0.005 - EPS,
            "bottom3_not_lower": delta["bottom3_class_map50_95"] >= -EPS,
            "worst_drop_no_more_than_1_point": delta["worst_class_map50_95"] >= -0.01 - EPS,
        }
        candidates[arm] = {
            "metrics": values,
            "delta_vs_af2c": delta,
            "latency": result.get("latency", {}),
            "criteria": criteria,
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
        "format": "coffee_detector.af2_spectral.seed42_decision.v1",
        "stage": stage,
        "seed": seed,
        "baseline": {"arm": "AF2C", "metrics": baseline, "source": str(Path(af2_result).resolve())},
        "candidates": candidates,
        "retained": retained,
        "winner": winner,
        "decision": decision,
        "next": (
            "AUTHORIZE_STAGE2"
            if stage == "stage1"
            else "AUTHORIZE_GLOBAL_DECISION"
            if stage == "stage2"
            else "AUTHORIZE_WINNER_PAIRED_CONFIRMATION"
            if decision == "PASS"
            else "KEEP_AF2C_AND_STOP"
        ),
        "test_opened": False,
    }
    path = output_root / "val_reports" / f"{stage}_seed{seed}_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2 spectral stage")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--af2-result", required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_ARMS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_spectral_decision(args.output_root, args.af2_result, stage=args.stage, seed=args.seed)


if __name__ == "__main__":
    main()
