"""Paired three-seed AF2 vs D0FT external evaluation without training."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


CHECKPOINTS = {
    "D0FT": {
        42: "faruq-v3-acmc-optimization-control-v1/D0FT_seed42/weights/best.pt",
        123: "faruq-v3-acmc-paired-confirmation-v1/D0FT/D0FT_seed123/weights/best.pt",
        2026: "faruq-v3-acmc-paired-confirmation-v1/D0FT/D0FT_seed2026/weights/best.pt",
    },
    "AF2": {
        42: "faruq-v3-breadth-screening-batch-v1/candidates/AFAB/AF2_seed42/weights/best.pt",
        123: "faruq-v3-af2-igem-paired-confirmation-v1/AF2/AF2_seed123/weights/best.pt",
        2026: "faruq-v3-af2-igem-paired-confirmation-v1/AF2/AF2_seed2026/weights/best.pt",
    },
}
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict:
    project, data, output = Path(args.project_root), Path(args.data_root), Path(args.output_root)
    worker_script = Path(__file__).with_name("run_coffee_standard_retained_external.py")
    env = os.environ.copy()
    rows = []
    for seed in (42, 123, 2026):
        row = {"seed": seed}
        for model in ("D0FT", "AF2"):
            checkpoint = project / "experiments" / CHECKPOINTS[model][seed]
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            report = output / "reports" / f"{model}_seed{seed}" / "evaluation.json"
            prior = project / "experiments/coffee-standard-v8-retained-external-v1/reports" / model / "evaluation.json"
            if seed == 42 and prior.is_file():
                print(f"REUSE PRIOR {model} seed 42", flush=True)
                row[model] = _load(prior)["metrics"]
                continue
            command = [sys.executable, str(worker_script), "--worker", "--model", f"{model}_seed{seed}",
                       "--checkpoint", str(checkpoint), "--data-root", str(data), "--output", str(report), "--device", args.device]
            print(f"EVALUATE {model} seed {seed}", flush=True)
            subprocess.run(command, env=env, check=True)
            row[model] = _load(report)["metrics"]
        for metric in METRICS:
            row[f"delta_{metric}"] = row["AF2"][metric] - row["D0FT"][metric]
        rows.append(row)
    aggregate = []
    for metric in METRICS:
        d0 = [row["D0FT"][metric] for row in rows]
        af2 = [row["AF2"][metric] for row in rows]
        deltas = [row[f"delta_{metric}"] for row in rows]
        aggregate.append({
            "metric": metric,
            "d0ft_mean": statistics.mean(d0), "d0ft_std": statistics.stdev(d0),
            "af2_mean": statistics.mean(af2), "af2_std": statistics.stdev(af2),
            "delta_mean": statistics.mean(deltas), "delta_std": statistics.stdev(deltas),
            "delta_min": min(deltas), "improved_seeds": sum(value > 0 for value in deltas),
        })
    by_metric = {row["metric"]: row for row in aggregate}
    criteria = {
        "macro_mean_positive": by_metric["macro_map50_95"]["delta_mean"] > 0,
        "macro_improved_at_least_2_of_3": by_metric["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_not_lower": by_metric["bottom3_class_map50_95"]["delta_mean"] >= 0,
        "worst_mean_drop_no_more_than_1_point": by_metric["worst_class_map50_95"]["delta_mean"] >= -0.01,
    }
    result = {
        "status": "complete", "decision": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria, "rows": rows, "aggregate": aggregate,
        "training_executed": False, "test_images_accessed": False,
        "role": "external_posthoc_three_seed_confirmation",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "coffee_standard_af2_paired_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True); parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True); parser.add_argument("--device", default="0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
