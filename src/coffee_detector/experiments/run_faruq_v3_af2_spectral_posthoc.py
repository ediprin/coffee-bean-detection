from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from coffee_detector.evaluate import evaluate


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _read(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _parse_conditions(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("condition-root harus NAME=PATH")
        name, path = value.split("=", 1)
        root = Path(path).expanduser().resolve()
        if not name or not root.is_dir() or (root / "test").exists():
            raise RuntimeError(f"Condition tidak aman: {value}")
        result[name] = root
    return result


def run_posthoc_external(
    confirmation_summary: str | Path,
    af2_checkpoints: tuple[str | Path, ...],
    candidate_checkpoints: tuple[str | Path, ...],
    condition_roots: list[str],
    output_root: str | Path,
    *,
    seeds: tuple[int, ...] = (42, 123, 2026),
    device: str = "0",
) -> dict:
    confirmation = _read(confirmation_summary)
    if (
        confirmation.get("decision") != "PASS"
        or confirmation.get("next") != "AUTHORIZE_POSTHOC_EXTERNAL_EVALUATION"
        or confirmation.get("test_opened") is not False
    ):
        raise RuntimeError("Paired confirmation belum mengotorisasi evaluasi eksternal")
    if len(af2_checkpoints) != len(seeds) or len(candidate_checkpoints) != len(seeds):
        raise ValueError("Checkpoint AF2/candidate harus tepat satu per seed")
    arm = confirmation["arm"]
    conditions = _parse_conditions(condition_roots)
    output_root = Path(output_root).expanduser().resolve()
    rows = []
    for seed, af2_value, candidate_value in zip(seeds, af2_checkpoints, candidate_checkpoints):
        for model_name, checkpoint_value in (("AF2C", af2_value), (arm, candidate_value)):
            checkpoint = Path(checkpoint_value).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for condition, data_root in conditions.items():
                report_path = output_root / "reports" / condition / f"{model_name}_seed{seed}.json"
                report = evaluate(checkpoint, data_root, report_path, split="val", device=device)
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "condition": condition,
                        "metrics": {metric: float(report["metrics"][metric]) for metric in METRICS},
                        "classes_without_ground_truth": report["metrics"].get("classes_without_ground_truth", []),
                    }
                )
    lookup = {(row["seed"], row["condition"], row["model"]): row for row in rows}
    comparisons = []
    for seed in seeds:
        for condition in conditions:
            baseline = lookup[(seed, condition, "AF2C")]["metrics"]
            candidate = lookup[(seed, condition, arm)]["metrics"]
            comparisons.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "deltas": {metric: candidate[metric] - baseline[metric] for metric in METRICS},
                }
            )
    aggregate = {
        condition: {
            metric: {
                "mean_delta": statistics.fmean(
                    row["deltas"][metric] for row in comparisons if row["condition"] == condition
                ),
                "minimum_delta": min(
                    row["deltas"][metric] for row in comparisons if row["condition"] == condition
                ),
                "improved_seeds": sum(
                    row["deltas"][metric] > 0 for row in comparisons if row["condition"] == condition
                ),
            }
            for metric in METRICS
        }
        for condition in conditions
    }
    result = {
        "format": "coffee_detector.af2_spectral.posthoc_external.v1",
        "status": "DEVELOPMENT_OR_EXTERNAL_POSTHOC_ONLY",
        "arm": arm,
        "seeds": list(seeds),
        "rows": rows,
        "comparisons": comparisons,
        "aggregate": aggregate,
        "training_executed": False,
        "test_images_accessed": False,
        "changes_confirmation_decision": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "af2_spectral_posthoc_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="No-training post-hoc AF2 spectral evaluation")
    parser.add_argument("--confirmation-summary", required=True)
    parser.add_argument("--af2-checkpoints", nargs="+", required=True)
    parser.add_argument("--candidate-checkpoints", nargs="+", required=True)
    parser.add_argument("--condition-roots", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run_posthoc_external(
        args.confirmation_summary,
        tuple(args.af2_checkpoints),
        tuple(args.candidate_checkpoints),
        args.condition_roots,
        args.output_root,
        seeds=tuple(args.seeds),
        device=args.device,
    )


if __name__ == "__main__":
    main()
