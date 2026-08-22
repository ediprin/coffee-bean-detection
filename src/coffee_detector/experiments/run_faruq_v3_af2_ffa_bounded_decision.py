"""Pareto decision for the bounded AF2 feature-frequency adapter.

The completed AF2FFA0 and AF2FFA1 seed-42 reports are immutable historical
references.  Only AF2FFAB1 is newly trained by this refinement study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _read(path: str | Path, expected_arm: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("arm") != expected_arm or int(payload.get("seed", -1)) != 42:
        raise RuntimeError(f"Report bukan {expected_arm} seed 42: {source}")
    if payload.get("test_images_accessed") is not False:
        raise RuntimeError(f"Report mengakses test: {source}")
    missing = set(METRICS).difference(payload.get("metrics", {}))
    if missing:
        raise RuntimeError(f"Report kehilangan metrik {sorted(missing)}: {source}")
    return payload


def _values(payload: dict) -> dict[str, float]:
    return {name: float(payload["metrics"][name]) for name in METRICS}


def _delta(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {name: candidate[name] - reference[name] for name in METRICS}


def run_faruq_v3_af2_ffa_bounded_decision(
    control_result: str | Path,
    unbounded_result: str | Path,
    bounded_result: str | Path,
    output: str | Path,
) -> dict:
    reports = {
        "AF2FFA0": _read(control_result, "AF2FFA0"),
        "AF2FFA1": _read(unbounded_result, "AF2FFA1"),
        "AF2FFAB1": _read(bounded_result, "AF2FFAB1"),
    }
    source_hashes = {
        payload.get("initial_af2_checkpoint_sha256") for payload in reports.values()
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise RuntimeError("Ketiga arm tidak berasal dari checkpoint AF2 yang sama")

    values = {arm: _values(payload) for arm, payload in reports.items()}
    versus_control = _delta(values["AF2FFAB1"], values["AF2FFA0"])
    versus_unbounded = _delta(values["AF2FFAB1"], values["AF2FFA1"])
    criteria = {
        # Preserve the control's global accuracy to within 0.1 percentage point.
        "macro_drop_vs_control_no_more_than_0_1_point": versus_control[
            "macro_map50_95"
        ]
        >= -0.001,
        "bottom3_gain_vs_control_at_least_0_5_point": versus_control[
            "bottom3_class_map50_95"
        ]
        >= 0.005,
        "worst_gain_vs_control_at_least_1_point": versus_control[
            "worst_class_map50_95"
        ]
        >= 0.01,
        # The cap must recover Macro over AF2FFA1 without erasing its tail signal.
        "macro_higher_than_unbounded": versus_unbounded["macro_map50_95"] > 0.0,
        "bottom3_drop_vs_unbounded_no_more_than_0_5_point": versus_unbounded[
            "bottom3_class_map50_95"
        ]
        >= -0.005,
        "worst_drop_vs_unbounded_no_more_than_0_5_point": versus_unbounded[
            "worst_class_map50_95"
        ]
        >= -0.005,
    }
    decision = "RETAIN_PARETO" if all(criteria.values()) else "REJECT"
    payload = {
        "format": "coffee_detector.af2_ffa.bounded_seed42_decision.v1",
        "seed": 42,
        "values": values,
        "bounded_minus_control": versus_control,
        "bounded_minus_unbounded": versus_unbounded,
        "criteria": criteria,
        "decision": decision,
        "next": (
            "DEFER_MULTISEED_UNTIL_USER_AUTHORIZES"
            if decision == "RETAIN_PARETO"
            else "RETAIN_ORIGINAL_AF2_WITHOUT_EXTRA_SEEDS"
        ),
        "training_executed_for_this_study": ["AF2FFAB1"],
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide bounded AF2-FFA seed-42 screen")
    parser.add_argument("--control-result", required=True)
    parser.add_argument("--unbounded-result", required=True)
    parser.add_argument("--bounded-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_faruq_v3_af2_ffa_bounded_decision(
        args.control_result,
        args.unbounded_result,
        args.bounded_result,
        args.output,
    )


if __name__ == "__main__":
    main()
