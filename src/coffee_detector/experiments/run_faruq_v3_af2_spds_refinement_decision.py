from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.af2_spds.config import REFINEMENT_ARMS


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _result(root: Path, arm: str) -> dict:
    path = root / "val_reports" / f"{arm}_seed42_result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("arm") != arm or int(row.get("seed", -1)) != 42:
        raise RuntimeError(f"Kontrak arm/seed tidak cocok pada {path}")
    expected_format = (
        "coffee_detector.af2_spds.arm_result.v1"
        if arm in {"AF2BASE", "AF2SPDS"}
        else "coffee_detector.af2_spds_refinement.arm_result.v1"
    )
    if row.get("format") != expected_format:
        raise RuntimeError(f"Schema hasil tidak cocok pada {path}")
    if row.get("test_images_accessed") is not False:
        raise RuntimeError(f"Test lock gagal pada {arm}")
    if row["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError(f"Validation kehilangan kelas pada {arm}")
    return {metric: float(row["metrics"][metric]) for metric in METRICS}


def run_af2_spds_refinement_decision(
    original_root: str | Path, refinement_root: str | Path, output: str | Path
) -> dict:
    original_root = Path(original_root).expanduser().resolve()
    refinement_root = Path(refinement_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    values = {
        "AF2BASE": _result(original_root, "AF2BASE"),
        "AF2SPDS": _result(original_root, "AF2SPDS"),
        **{arm: _result(refinement_root, arm) for arm in REFINEMENT_ARMS},
    }
    comparisons = {}
    retained = []
    for arm in REFINEMENT_ARMS:
        versus_base = {m: values[arm][m] - values["AF2BASE"][m] for m in METRICS}
        versus_spds = {m: values[arm][m] - values["AF2SPDS"][m] for m in METRICS}
        criteria = {
            "macro_within_0_1_point_of_base": versus_base["macro_map50_95"] >= -0.001,
            "bottom3_within_0_5_point_of_spds": versus_spds["bottom3_class_map50_95"] >= -0.005,
            "worst_within_0_5_point_of_spds": versus_spds["worst_class_map50_95"] >= -0.005,
            "improves_at_least_one_metric_over_spds": any(value > 0 for value in versus_spds.values()),
        }
        passed = all(criteria.values())
        comparisons[arm] = {
            "versus_base": versus_base,
            "versus_spds": versus_spds,
            "criteria": criteria,
            "decision": "RETAIN" if passed else "REJECT",
        }
        if passed:
            retained.append(arm)
    winner = max(retained, key=lambda arm: values[arm]["macro_map50_95"]) if retained else None
    result = {
        "format": "coffee_detector.af2_spds_refinement.seed42_decision.v1",
        "values": values,
        "comparisons": comparisons,
        "retained": retained,
        "winner": winner,
        "decision": "PASS" if winner else "FAIL_KILL_GATE",
        "next": "FREEZE_PAIRED_CONFIRMATION_PROTOCOL" if winner else "RETAIN_ORIGINAL_AF2",
        "test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2-SPDS refinements")
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--refinement-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_af2_spds_refinement_decision(args.original_root, args.refinement_root, args.output)


if __name__ == "__main__":
    main()
