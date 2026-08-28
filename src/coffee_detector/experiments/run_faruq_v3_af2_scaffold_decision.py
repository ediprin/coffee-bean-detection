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


def run_af2_scaffold_decision(
    control_result: str | Path,
    candidate_result: str | Path,
    output: str | Path,
) -> dict:
    control = _read(Path(control_result).expanduser().resolve())
    candidate = _read(Path(candidate_result).expanduser().resolve())
    if candidate.get("format") != "coffee_detector.af2_scaffold.arm_result.v1":
        raise RuntimeError("Candidate result schema salah")
    if candidate.get("seed") != 42 or candidate.get("arm") != "AF2MTS1":
        raise RuntimeError("Candidate bukan AF2MTS1 seed 42")
    if control.get("seed") != 42 or control.get("arm") != "AF2CTRL":
        raise RuntimeError("Control bukan AF2CTRL seed 42")
    if control.get("test_images_accessed") is not False or candidate.get("test_images_accessed") is not False:
        raise RuntimeError("Test lock dilanggar")
    control_values = {key: float(control["metrics"][key]) for key in METRICS}
    candidate_values = {key: float(candidate["metrics"][key]) for key in METRICS}
    deltas = {key: candidate_values[key] - control_values[key] for key in METRICS}
    export_exact = (
        all(abs(float(value)) <= 1e-6 for value in candidate["native_export_deltas"].values())
        and candidate.get("native_export_state_exact") is True
    )
    criteria = {
        "macro_at_least_90_5_percent": candidate_values["macro_map50_95"] >= 0.905,
        "macro_gain_at_least_1_5_points": deltas["macro_map50_95"] >= 0.015,
        "bottom3_at_least_84_5_percent": candidate_values["bottom3_class_map50_95"] >= 0.845,
        "bottom3_not_lower_than_control": deltas["bottom3_class_map50_95"] >= 0.0,
        "worst_not_lower_than_control": deltas["worst_class_map50_95"] >= 0.0,
        "native_export_exact": export_exact,
        "all_21_classes_present": not candidate["metrics"].get("classes_without_ground_truth"),
        "test_not_opened": True,
    }
    passed = all(criteria.values())
    result = {
        "format": "coffee_detector.af2_scaffold.seed42_decision.v1",
        "seed": 42,
        "values": {"AF2CTRL": control_values, "AF2MTS1": candidate_values},
        "deltas": deltas,
        "criteria": criteria,
        "decision": "PASS_KILL_GATE" if passed else "FAIL_KILL_GATE",
        "next": "FREEZE_CANDIDATE_WITHOUT_TEST" if passed else "STOP_MULTILEVEL_SCAFFOLD",
        "test_opened": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2MTS1 seed-42 kill gate")
    parser.add_argument("--control-result", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_af2_scaffold_decision(args.control_result, args.candidate_result, args.output)


if __name__ == "__main__":
    main()
