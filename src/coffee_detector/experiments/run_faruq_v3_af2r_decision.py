from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {metric: float(source[metric]) for metric in METRICS}


def run_faruq_v3_af2r_decision(
    output_root: str | Path,
    af2_reference: str | Path,
    *,
    seed: int = 42,
) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    results = {
        arm: _read(output_root / "val_reports" / f"{arm}_seed{seed}_result.json")
        for arm in ("AF2R0", "AF2R1")
    }
    if any(result.get("test_images_accessed") is not False for result in results.values()):
        raise RuntimeError("Salah satu arm mengakses test")
    reference = _read(Path(af2_reference).expanduser().resolve())
    if (
        reference.get("protocol") != "faruq-v3-lfdet-afab-breadth-screening-v1"
        or int(reference.get("seed", -1)) != seed
        or reference.get("test_images_accessed") is not False
        or reference.get("decisions", {}).get("AF2", {}).get("decision") != "RETAIN"
    ):
        raise RuntimeError("Referensi AF2 bukan hasil RETAIN seed 42 yang kompatibel")
    fixed = _metrics(reference["candidate"]["AF2"])
    values = {"AF2": fixed, **{arm: _metrics(result) for arm, result in results.items()}}
    versus_control = {
        metric: values["AF2R1"][metric] - values["AF2R0"][metric] for metric in METRICS
    }
    versus_fixed = {
        metric: values["AF2R1"][metric] - values["AF2"][metric] for metric in METRICS
    }
    criteria = {
        "conditioned_macro_gain_at_least_0_5_vs_control": versus_control["macro_map50_95"] >= 0.005,
        "conditioned_bottom3_not_lower_vs_control": versus_control["bottom3_class_map50_95"] >= 0.0,
        "conditioned_worst_drop_no_more_than_1_vs_control": versus_control["worst_class_map50_95"] >= -0.01,
        "conditioned_macro_not_lower_than_fixed_af2": versus_fixed["macro_map50_95"] >= 0.0,
        "conditioned_bottom3_drop_no_more_than_1_vs_fixed_af2": versus_fixed["bottom3_class_map50_95"] >= -0.01,
        "conditioned_worst_drop_no_more_than_1_vs_fixed_af2": versus_fixed["worst_class_map50_95"] >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.af2r.seed42_decision.v1",
        "seed": seed,
        "values": values,
        "af2r1_minus_af2r0": versus_control,
        "af2r1_minus_fixed_af2": versus_fixed,
        "criteria": criteria,
        "decision": decision,
        "next": "AUTHORIZE_PAIRED_ILLUMINATION_SCREEN" if decision == "PASS" else "STOP_WITHOUT_ILLUMINATION_OR_EXTRA_SEEDS",
        "test_opened": False,
    }
    path = output_root / "val_reports" / f"af2r_seed{seed}_decision.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide adaptive residual AF2 seed-42 screen")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--af2-reference", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_faruq_v3_af2r_decision(args.output_root, args.af2_reference, seed=args.seed)


if __name__ == "__main__":
    main()
