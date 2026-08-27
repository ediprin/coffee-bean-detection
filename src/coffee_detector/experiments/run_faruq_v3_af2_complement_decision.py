from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.af2_complement.config import ARMS


METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def run_af2_complement_decision(output_root: str | Path, output: str | Path) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    rows = {}
    for arm in ARMS:
        path = output_root / "val_reports" / f"{arm}_seed42_result.json"
        if not path.is_file():
            raise FileNotFoundError(f"Result belum lengkap: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"Test lock tidak valid: {arm}")
        rows[arm] = {metric: float(payload["metrics"][metric]) for metric in METRICS}

    control = rows["AF2CTRL"]
    comparisons, retained = {}, []
    for arm in ARMS[1:]:
        delta = {metric: rows[arm][metric] - control[metric] for metric in METRICS}
        strict = (
            delta["macro_map50_95"] >= 0.005
            and delta["bottom3_class_map50_95"] >= 0.0
            and delta["worst_class_map50_95"] >= -0.01
        )
        lower_tail_pareto = (
            delta["macro_map50_95"] >= -0.001
            and delta["bottom3_class_map50_95"] >= 0.005
            and delta["worst_class_map50_95"] >= 0.005
        )
        comparisons[arm] = {
            "deltas": delta,
            "strict_macro_gate": strict,
            "lower_tail_pareto_gate": lower_tail_pareto,
            "decision": "RETAIN" if strict or lower_tail_pareto else "REJECT",
        }
        if strict or lower_tail_pareto:
            retained.append(arm)

    winner = None
    if retained:
        winner = max(
            retained,
            key=lambda arm: (
                rows[arm]["macro_map50_95"],
                rows[arm]["bottom3_class_map50_95"],
                rows[arm]["worst_class_map50_95"],
            ),
        )
    result = {
        "format": "coffee_detector.af2_complement.seed42_decision.v1",
        "values": rows,
        "comparisons": comparisons,
        "retained": retained,
        "winner": winner,
        "decision": "PASS" if winner else "FAIL",
        "next": "FREEZE_PAIRED_CONFIRMATION_PROTOCOL" if winner else "RETAIN_AF2_WITHOUT_COMPLEMENT",
        "test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide AF2 complementary seed-42 screen")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_af2_complement_decision(args.output_root, args.output)
    print("VALUES:", result["values"])
    print("COMPARISONS:", result["comparisons"])
    print("RETAINED:", result["retained"])
    print("WINNER:", result["winner"])
    print("DECISION:", result["decision"])
    print("NEXT:", result["next"])
    print("TEST OPENED:", result["test_opened"])


if __name__ == "__main__":
    main()
