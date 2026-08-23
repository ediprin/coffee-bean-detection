from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
PAIRS = {"saf": ("AF2SAF0", "AF2SAF1"), "igem": ("AF2IGEM0", "AF2IGEM1")}


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _headline(payload: dict) -> dict[str, float]:
    return {metric: float(payload["metrics"][metric]) for metric in METRICS}


def run_faruq_v3_af2_parent_residual_decision(
    family: str,
    control_result: str | Path,
    candidate_result: str | Path,
    output: str | Path,
) -> dict:
    if family not in PAIRS:
        raise ValueError("family harus saf atau igem")
    control, candidate = _read(control_result), _read(candidate_result)
    expected_control, expected_candidate = PAIRS[family]
    for payload, arm, conditioning in (
        (control, expected_control, "zero"),
        (candidate, expected_candidate, "feature"),
    ):
        if (
            payload.get("format") != "coffee_detector.af2_parent_residual.arm_result.v1"
            or payload.get("arm") != arm
            or payload.get("family") != family
            or payload.get("conditioning") != conditioning
            or int(payload.get("seed", -1)) != 42
            or payload.get("test_images_accessed") is not False
        ):
            raise RuntimeError(f"Result {arm} tidak memenuhi kontrak")
    if control["initial_af2_checkpoint_sha256"] != candidate["initial_af2_checkpoint_sha256"]:
        raise RuntimeError("Control dan candidate tidak berasal dari AF2 yang sama")
    af2 = {metric: float(candidate["baseline_metrics"][metric]) for metric in METRICS}
    control_values, candidate_values = _headline(control), _headline(candidate)
    versus_control = {
        metric: candidate_values[metric] - control_values[metric] for metric in METRICS
    }
    versus_af2 = {metric: candidate_values[metric] - af2[metric] for metric in METRICS}
    superiority_route = (
        versus_control["macro_map50_95"] >= 0.002
        and versus_control["bottom3_class_map50_95"] >= -0.005
        and versus_control["worst_class_map50_95"] >= -0.010
    )
    tail_pareto_route = (
        versus_control["macro_map50_95"] >= -0.001
        and versus_control["bottom3_class_map50_95"] >= 0.005
        and versus_control["worst_class_map50_95"] >= 0.010
    )
    criteria = {
        "candidate_not_materially_below_af2_macro": versus_af2["macro_map50_95"] >= -0.002,
        "candidate_not_materially_below_af2_bottom3": versus_af2["bottom3_class_map50_95"] >= -0.010,
        "candidate_not_materially_below_af2_worst": versus_af2["worst_class_map50_95"] >= -0.010,
        "superiority_route": superiority_route,
        "tail_pareto_route": tail_pareto_route,
        "all_21_classes_present": len(candidate["metrics"]["map50_95_by_class"]) == 21,
        "test_not_opened": candidate["test_images_accessed"] is False,
    }
    safety = all(
        criteria[key]
        for key in (
            "candidate_not_materially_below_af2_macro",
            "candidate_not_materially_below_af2_bottom3",
            "candidate_not_materially_below_af2_worst",
            "all_21_classes_present",
            "test_not_opened",
        )
    )
    decision = "RETAIN" if safety and (superiority_route or tail_pareto_route) else "REJECT"
    result = {
        "format": "coffee_detector.af2_parent_residual.seed42_decision.v1",
        "family": family,
        "values": {"AF2": af2, expected_control: control_values, expected_candidate: candidate_values},
        "candidate_minus_matched_control": versus_control,
        "candidate_minus_af2": versus_af2,
        "criteria": criteria,
        "decision": decision,
        "next": "AUTHORIZE_ONLY_THIS_FAMILY_MULTI_SEED" if decision == "RETAIN" else "STOP_THIS_FAMILY",
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2 parent-residual family decision")
    parser.add_argument("--family", choices=tuple(PAIRS), required=True)
    parser.add_argument("--control-result", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_faruq_v3_af2_parent_residual_decision(
        args.family, args.control_result, args.candidate_result, args.output
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
