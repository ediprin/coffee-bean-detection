from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
SEEDS = (42, 123, 2026)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run_faruq_v3_af2_ffa_confirmation(output_root: str | Path) -> dict:
    root = Path(output_root).expanduser().resolve()
    rows, deltas = [], {metric: [] for metric in METRICS}
    for seed in SEEDS:
        pair = {
            arm: _read(root / "val_reports" / f"{arm}_seed{seed}_result.json")
            for arm in ("AF2FFA0", "AF2FFA1")
        }
        if any(item.get("test_images_accessed") is not False for item in pair.values()):
            raise RuntimeError("Confirmation mengakses test")
        row = {"seed": seed}
        for arm, payload in pair.items():
            row[arm] = {metric: float(payload["metrics"][metric]) for metric in METRICS}
        for metric in METRICS:
            deltas[metric].append(row["AF2FFA1"][metric] - row["AF2FFA0"][metric])
        rows.append(row)
    aggregates = {
        metric: {
            "delta_mean": sum(values) / len(values),
            "delta_min": min(values),
            "improved_seeds": sum(value > 0 for value in values),
        }
        for metric, values in deltas.items()
    }
    criteria = {
        "macro_mean_gain_at_least_0_5_point": aggregates["macro_map50_95"]["delta_mean"] >= 0.005,
        "macro_improved_at_least_2_of_3": aggregates["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_not_lower": aggregates["bottom3_class_map50_95"]["delta_mean"] >= 0.0,
        "bottom3_improved_at_least_2_of_3": aggregates["bottom3_class_map50_95"]["improved_seeds"] >= 2,
        "worst_mean_drop_no_more_than_1_point": aggregates["worst_class_map50_95"]["delta_mean"] >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.af2_ffa.paired_confirmation.v1",
        "seeds": list(SEEDS),
        "rows": rows,
        "aggregates": aggregates,
        "criteria": criteria,
        "decision": decision,
        "next": "AUTHORIZE_POSTHOC_DIAGNOSTICS" if decision == "PASS" else "RETAIN_ORIGINAL_AF2",
        "test_opened": False,
    }
    destination = root / "val_reports/af2_ffa_paired_confirmation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2-FFA paired confirmation")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    run_faruq_v3_af2_ffa_confirmation(args.output_root)


if __name__ == "__main__":
    main()
