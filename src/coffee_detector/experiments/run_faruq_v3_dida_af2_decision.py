from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("AF2FT", "AF2DG", "AF2FG", "AF2DGFG")
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_faruq_v3_dida_af2_decision(output_root: str | Path, *, seed: int = 42) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    results = {
        arm: _load(output_root / "val_reports" / f"{arm}_seed{seed}_result.json")
        for arm in ARMS
    }
    if any(result.get("test_images_accessed") is not False for result in results.values()):
        raise RuntimeError("Salah satu arm mengakses test")
    values = {
        arm: {metric: float(results[arm]["metrics"][metric]) for metric in METRICS}
        for arm in ARMS
    }
    effects = {}
    for metric in METRICS:
        ft, dg, fg, joint = (values[arm][metric] for arm in ARMS)
        effects[metric] = {
            "dg": dg - ft,
            "fg": fg - ft,
            "joint_vs_control": joint - ft,
            "joint_vs_dg": joint - dg,
            "joint_vs_fg": joint - fg,
            "interaction": joint - dg - fg + ft,
        }
    macro = effects["macro_map50_95"]
    bottom = effects["bottom3_class_map50_95"]
    worst = effects["worst_class_map50_95"]
    criteria = {
        "joint_macro_not_lower_than_control": macro["joint_vs_control"] >= 0.0,
        "joint_bottom3_higher_than_control": bottom["joint_vs_control"] > 0.0,
        "joint_worst_drop_no_more_than_1_point": worst["joint_vs_control"] >= -0.01,
        "joint_macro_gain_at_least_0_5_vs_dg": macro["joint_vs_dg"] >= 0.005,
        "joint_macro_gain_at_least_0_5_vs_fg": macro["joint_vs_fg"] >= 0.005,
        "joint_bottom3_not_lower_than_dg": bottom["joint_vs_dg"] >= 0.0,
        "joint_bottom3_not_lower_than_fg": bottom["joint_vs_fg"] >= 0.0,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.dida_af2.seed42_decision.v1",
        "seed": seed,
        "values": values,
        "effects": effects,
        "criteria": criteria,
        "decision": decision,
        "next": "AUTHORIZE_PAIRED_THREE_SEED_FACTORIAL" if decision == "PASS" else "STOP_WITHOUT_TEST_OR_EXTRA_SEEDS",
        "test_opened": False,
    }
    path = output_root / "val_reports" / f"dida_af2_factorial_seed{seed}_decision.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide DIDA-AF2 seed-42 factorial")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_faruq_v3_dida_af2_decision(args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()
