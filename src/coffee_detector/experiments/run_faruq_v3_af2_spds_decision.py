from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.af2_spds import ARMS


METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def run_af2_spds_decision(output_root: str | Path, output: str | Path) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    values = {}
    for arm in ARMS:
        path = output_root / "val_reports" / f"{arm}_seed42_result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("test_images_accessed") is not False:
            raise RuntimeError(f"Test lock gagal pada {arm}")
        metrics = row["metrics"]
        if metrics.get("classes_without_ground_truth"):
            raise RuntimeError(f"Validation kehilangan kelas pada {arm}")
        values[arm] = {metric: float(metrics[metric]) for metric in METRICS}

    def delta(candidate: str, baseline: str) -> dict[str, float]:
        return {metric: values[candidate][metric] - values[baseline][metric] for metric in METRICS}

    versus_base = delta("AF2SPDS", "AF2BASE")
    versus_rgb = delta("AF2SPDS", "AF2RGBDS")
    macro_route = (
        versus_base["macro_map50_95"] >= 0.005
        and versus_base["bottom3_class_map50_95"] >= 0.0
        and versus_base["worst_class_map50_95"] >= -0.005
    )
    tail_route = (
        versus_base["macro_map50_95"] >= -0.001
        and versus_base["bottom3_class_map50_95"] >= 0.005
        and versus_base["worst_class_map50_95"] >= 0.005
    )
    cue_wins = sum(versus_rgb[metric] > 0 for metric in METRICS)
    cue_specific = versus_rgb["macro_map50_95"] >= -0.001 and cue_wins >= 2
    criteria = {
        "strong_macro_route": macro_route,
        "lower_tail_pareto_route": tail_route,
        "af2_signal_beats_generic_rgb_on_at_least_two_metrics": cue_wins >= 2,
        "af2_signal_macro_not_more_than_0_1_point_below_rgb": versus_rgb["macro_map50_95"] >= -0.001,
        "cue_specific_evidence": cue_specific,
        "all_21_validation_classes_present": True,
        "test_not_opened": True,
    }
    passed = (macro_route or tail_route) and cue_specific
    result = {
        "format": "coffee_detector.af2_spds.seed42_decision.v1",
        "values": values,
        "spds_minus_base": versus_base,
        "spds_minus_rgb_control": versus_rgb,
        "criteria": criteria,
        "decision": "PASS" if passed else "FAIL_KILL_GATE",
        "next": "FREEZE_PAIRED_CONFIRMATION_PROTOCOL" if passed else "RETAIN_ORIGINAL_AF2",
        "test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2-SPDS seed-42 screen")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_af2_spds_decision(args.output_root, args.output)


if __name__ == "__main__":
    main()
