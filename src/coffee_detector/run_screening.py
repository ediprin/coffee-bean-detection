from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate
from .train import load_experiment, train_experiment


def _metric(metrics: dict, needle: str) -> float | None:
    for key, value in metrics.items():
        normalized = key.lower().replace(" ", "")
        wanted = needle.lower().replace(" ", "")
        if wanted == "map50" and "map50-95" in normalized:
            continue
        if wanted in normalized and isinstance(value, (int, float)):
            return float(value)
    return None


def run_screening(
    data_root: str | Path,
    output_root: str | Path,
    configs: list[str | Path],
    seed: int = 42,
    device: str | None = None,
    resume: bool = True,
) -> dict:
    output_root = Path(output_root).resolve()
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    results = {}
    for config_path in configs:
        config = load_experiment(config_path)
        code = str(config["code"])
        run_dir = output_root / f"{code}_seed{seed}"
        best = run_dir / "weights" / "best.pt"
        if not best.is_file():
            print(f"\nSTART TRAINING: {code} | seed {seed}", flush=True)
            run_dir = train_experiment(
                config_path,
                data_root,
                output_root,
                seed,
                device=device,
                resume=resume,
            )
            best = run_dir / "weights" / "best.pt"
        else:
            print(f"SKIP TRAINING: checkpoint ditemukan {best}", flush=True)
        if not best.is_file():
            raise FileNotFoundError(f"Training selesai tanpa best.pt: {best}")
        report_path = reports / f"{code}_seed{seed}_test.json"
        payload = evaluate(best, data_root, report_path, split="test", device=device)
        results[code] = payload["metrics"]

    summary = {"seed": seed, "data_root": str(Path(data_root).resolve()), "models": results}
    codes = list(results)
    if len(codes) == 2:
        baseline, candidate = codes
        summary["comparison"] = {"baseline": baseline, "candidate": candidate, "delta": {}}
        for needle in ("mAP50-95", "mAP50", "precision", "recall"):
            left = _metric(results[baseline], needle)
            right = _metric(results[candidate], needle)
            if left is not None and right is not None:
                summary["comparison"]["delta"][needle] = right - left
        left_worst = results[baseline].get("worst_class_map50_95")
        right_worst = results[candidate].get("worst_class_map50_95")
        if left_worst is not None and right_worst is not None:
            summary["comparison"]["delta"]["worst_class_map50_95"] = right_worst - left_worst
    summary_path = reports / f"screening_seed{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Screening hemat Y0 vs Y1 pada satu seed.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/Y0_yolo11n.yaml",
            "configs/Y1_yolo11n_local_hbp.yaml",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    summary = run_screening(
        args.data_root,
        args.output_root,
        args.configs,
        seed=args.seed,
        device=args.device,
        resume=not args.no_resume,
    )
    print("\n=== SCREENING Y0 vs Y1 ===")
    for code, metrics in summary["models"].items():
        print(f"{code}: {metrics}")
    if "comparison" in summary:
        print(f"DELTA: {summary['comparison']['delta']}")


if __name__ == "__main__":
    main()
