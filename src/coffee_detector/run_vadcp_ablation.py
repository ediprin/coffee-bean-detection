from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audit_dataset import audit_dataset
from .audit_vadcp import audit_vadcp_dataset
from .evaluate import evaluate
from .run_baseline import is_training_complete
from .run_visual_audit import run_visual_audit
from .train import load_experiment, train_experiment


DEFAULT_CONFIGS = {
    "A0": "configs/A0_yolo26n_real.yaml",
    "A1": "configs/A1_yolo26n_naive_copy_paste.yaml",
    "A2": "configs/A2_yolo26n_vadcp.yaml",
}


def _parse_pairs(values: list[str], label: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} harus CODE=PATH: {value}")
        code, path = value.split("=", 1)
        code = code.strip()
        if not code or not path.strip():
            raise ValueError(f"{label} tidak valid: {value}")
        if code in result:
            raise ValueError(f"{label} duplikat: {code}")
        result[code] = path.strip()
    return result


def _metric(metrics: dict, needle: str) -> float | None:
    wanted = needle.lower().replace(" ", "")
    for key, value in metrics.items():
        normalized = key.lower().replace(" ", "")
        if wanted == "map50" and "map50-95" in normalized:
            continue
        if wanted in normalized and isinstance(value, (int, float)):
            return float(value)
    return None


def _audit_arm(code: str, data_root: Path, reports: Path) -> Path:
    output = reports / f"{code}_dataset_audit.json"
    if (data_root / "metadata" / "instances_synthetic_train.json").is_file():
        report = audit_vadcp_dataset(data_root, output)
    else:
        report = audit_dataset(data_root, output, near_threshold=-1)
    if not report["safe_for_training"]:
        raise RuntimeError(f"Dataset arm {code} belum aman: {output}")
    return output


def run_vadcp_ablation(
    arms: dict[str, str | Path],
    output_root: str | Path,
    *,
    configs: dict[str, str | Path] | None = None,
    seeds: tuple[int, ...] = (42,),
    device: str | None = None,
    resume: bool = True,
    count_audit: bool = True,
    count_confidence: float = 0.25,
) -> dict:
    if "A0" not in arms:
        raise ValueError("Arm A0 real-only wajib sebagai baseline")
    if not seeds:
        raise ValueError("Minimal satu seed diperlukan")
    configs = {**DEFAULT_CONFIGS, **(configs or {})}
    missing = sorted(set(arms) - set(configs))
    if missing:
        raise ValueError("Config belum diberikan untuk arm: " + ", ".join(missing))
    output_root = Path(output_root).expanduser().resolve()
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    resolved_arms = {
        code: Path(path).expanduser().resolve() for code, path in arms.items()
    }
    audit_paths = {
        code: str(_audit_arm(code, data_root, reports))
        for code, data_root in resolved_arms.items()
    }
    runs: dict[str, dict[str, dict]] = defaultdict(dict)

    for seed in seeds:
        for code, data_root in resolved_arms.items():
            config_path = Path(configs[code]).expanduser().resolve()
            config = load_experiment(config_path)
            if str(config["code"]) != code:
                raise ValueError(
                    f"Code config {config_path} adalah {config['code']}, bukan {code}"
                )
            run_dir = output_root / f"{code}_seed{seed}"
            if not is_training_complete(run_dir):
                action = "RESUME" if (run_dir / "weights" / "last.pt").is_file() else "START"
                print(f"\n{action} TRAINING: {code} | seed {seed}", flush=True)
                run_dir = train_experiment(
                    config_path,
                    data_root,
                    output_root,
                    seed,
                    device=device,
                    resume=resume,
                )
            else:
                print(f"SKIP TRAINING: {code} seed {seed} sudah lengkap", flush=True)
            checkpoint = run_dir / "weights" / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"best.pt tidak ditemukan: {checkpoint}")
            evaluation_path = reports / f"{code}_seed{seed}_test.json"
            evaluation = evaluate(
                checkpoint, data_root, evaluation_path, split="test", device=device
            )
            count_payload = None
            if count_audit:
                count_payload = run_visual_audit(
                    checkpoint,
                    data_root,
                    output_root / "count_audit" / f"{code}_seed{seed}",
                    samples=0,
                    seed=seed,
                    device=device,
                    confidence=count_confidence,
                )
            runs[code][str(seed)] = {
                "data_root": str(data_root),
                "config": str(config_path),
                "checkpoint": str(checkpoint),
                "evaluation": str(evaluation_path),
                "metrics": evaluation["metrics"],
                "count_metrics": (
                    {
                        key: count_payload[key]
                        for key in (
                            "exact_count_match_rate",
                            "mean_absolute_count_error",
                            "mean_count_bias",
                            "over_count_rate",
                            "under_count_rate",
                        )
                        if key in count_payload
                    }
                    if count_payload
                    else None
                ),
            }

    aggregate = {}
    tracked = ("mAP50-95", "mAP50", "precision", "recall")
    for code, seed_rows in runs.items():
        aggregate[code] = {}
        for metric_name in tracked:
            values = [
                value
                for row in seed_rows.values()
                if (value := _metric(row["metrics"], metric_name)) is not None
            ]
            if values:
                aggregate[code][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }
        for count_name in (
            "exact_count_match_rate",
            "mean_absolute_count_error",
            "mean_count_bias",
            "over_count_rate",
            "under_count_rate",
        ):
            values = [
                float(row["count_metrics"][count_name])
                for row in seed_rows.values()
                if row["count_metrics"] and count_name in row["count_metrics"]
            ]
            if values:
                aggregate[code][count_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }

    comparisons = {}
    baseline = aggregate["A0"]
    for code in sorted(set(aggregate) - {"A0"}):
        comparisons[f"{code}_vs_A0"] = {}
        for metric_name in set(baseline) & set(aggregate[code]):
            comparisons[f"{code}_vs_A0"][metric_name] = (
                aggregate[code][metric_name]["mean"] - baseline[metric_name]["mean"]
            )
    payload = {
        "format": "coffee_detector.vadcp_ablation.v1",
        "arms": {code: str(path) for code, path in resolved_arms.items()},
        "audits": audit_paths,
        "seeds": list(seeds),
        "runs": dict(runs),
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    summary_path = reports / "vadcp_ablation_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(summary_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation real-only vs naive copy-paste vs VA-DCP."
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="Ulangi sebagai CODE=/path/dataset, misalnya A0=/data/real.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Override config sebagai CODE=/path/config.yaml.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-count-audit", action="store_true")
    parser.add_argument("--count-confidence", type=float, default=0.25)
    args = parser.parse_args()
    result = run_vadcp_ablation(
        _parse_pairs(args.arm, "--arm"),
        args.output_root,
        configs=_parse_pairs(args.config, "--config") if args.config else None,
        seeds=tuple(args.seeds),
        device=args.device,
        resume=not args.no_resume,
        count_audit=not args.skip_count_audit,
        count_confidence=args.count_confidence,
    )
    print("\n=== VA-DCP ABLATION ===")
    for code, metrics in result["aggregate"].items():
        print(code, json.dumps(metrics, ensure_ascii=False))
    print("DELTA", json.dumps(result["comparisons"], indent=2, ensure_ascii=False))
    print(f"SAVED: {result['summary']}")


if __name__ == "__main__":
    main()
