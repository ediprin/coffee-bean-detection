from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
EPS = 1.0e-12


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {metric: float(source[metric]) for metric in METRICS}


def run_faruq_v3_af2cal_decision(
    output_root: str | Path,
    af2r_evidence: str | Path,
    *,
    seed: int = 42,
) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    results = {
        arm: _read(output_root / "val_reports" / f"{arm}_seed{seed}_result.json")
        for arm in ("AF2FT30", "AF2CAL3")
    }
    if any(
        result.get("test_images_accessed") is not False for result in results.values()
    ):
        raise RuntimeError("Salah satu arm mengakses test")
    evidence = _read(Path(af2r_evidence).expanduser().resolve())
    if (
        evidence.get("format")
        != "coffee_detector.af2r.seed42_recovered_evidence.v1"
        or evidence.get("seed") != seed
        or evidence.get("decision") != "FAIL"
        or evidence.get("test_opened") is not False
    ):
        raise RuntimeError("Evidence AF2R0 tidak kompatibel")
    values = {
        "AF2": _metrics(evidence["values"]["AF2"]),
        "AF2R0": _metrics(evidence["values"]["AF2R0"]),
        **{arm: _metrics(result) for arm, result in results.items()},
    }
    versus_ft = {
        metric: values["AF2CAL3"][metric] - values["AF2FT30"][metric]
        for metric in METRICS
    }
    versus_r0 = {
        metric: values["AF2CAL3"][metric] - values["AF2R0"][metric]
        for metric in METRICS
    }
    ft_versus_r0 = {
        metric: values["AF2FT30"][metric] - values["AF2R0"][metric]
        for metric in METRICS
    }
    criteria = {
        "calibration_macro_gain_at_least_0_5_vs_ft": versus_ft[
            "macro_map50_95"
        ]
        >= 0.005 - EPS,
        "calibration_bottom3_not_lower_vs_ft": versus_ft[
            "bottom3_class_map50_95"
        ]
        >= -EPS,
        "calibration_worst_drop_no_more_than_1_vs_ft": versus_ft[
            "worst_class_map50_95"
        ]
        >= -0.01 - EPS,
        "calibration_macro_within_0_5_of_af2r0": versus_r0["macro_map50_95"]
        >= -0.005 - EPS,
        "calibration_bottom3_within_1_of_af2r0": versus_r0[
            "bottom3_class_map50_95"
        ]
        >= -0.01 - EPS,
        "calibration_worst_within_1_of_af2r0": versus_r0[
            "worst_class_map50_95"
        ]
        >= -0.01 - EPS,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    optimization_reaches_r0 = (
        ft_versus_r0["macro_map50_95"] >= -0.005 - EPS
        and ft_versus_r0["bottom3_class_map50_95"] >= -0.01 - EPS
        and ft_versus_r0["worst_class_map50_95"] >= -0.01 - EPS
    )
    attribution = (
        "CHANNEL_CALIBRATION_SUPPORTED"
        if decision == "PASS"
        else (
            "CONTINUATION_OPTIMIZATION_SUFFICIENT"
            if optimization_reaches_r0
            else "AF2R0_GAIN_NOT_EXPLAINED"
        )
    )
    payload = {
        "format": "coffee_detector.af2cal.seed42_decision.v1",
        "seed": seed,
        "values": values,
        "af2cal3_minus_af2ft30": versus_ft,
        "af2cal3_minus_af2r0": versus_r0,
        "af2ft30_minus_af2r0": ft_versus_r0,
        "criteria": criteria,
        "decision": decision,
        "attribution": attribution,
        "next": (
            "AUTHORIZE_PAIRED_THREE_SEED_AF2CAL3"
            if decision == "PASS"
            else "STOP_WITHOUT_EXTRA_SEEDS_OR_TEST"
        ),
        "test_opened": False,
    }
    path = output_root / "val_reports" / f"af2cal_seed{seed}_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2 calibration screen")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--af2r-evidence", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_faruq_v3_af2cal_decision(
        args.output_root, args.af2r_evidence, seed=args.seed
    )


if __name__ == "__main__":
    main()
