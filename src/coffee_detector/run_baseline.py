from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_dataset import audit_dataset
from .evaluate import evaluate
from .train import load_experiment, train_experiment


def run_baseline(
    data_root: str | Path,
    output_root: str | Path,
    config_path: str | Path = "configs/D0_yolo26n.yaml",
    seed: int = 42,
    device: str | None = None,
    resume: bool = True,
) -> dict:
    """Audit, train, and evaluate one locked detector baseline."""
    output_root = Path(output_root).expanduser().resolve()
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    audit_path = reports / "dataset_audit.json"
    audit = audit_dataset(data_root, audit_path)
    if not audit["safe_for_training"]:
        raise RuntimeError(
            "Dataset belum aman untuk training. Periksa "
            f"{audit_path} lalu jalankan coffee_detector.prepare_dataset."
        )

    config = load_experiment(config_path)
    code = str(config["code"])
    run_dir = output_root / f"{code}_seed{seed}"
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        print(f"START TRAINING: {code} | seed {seed}", flush=True)
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

    evaluation_path = reports / f"{code}_seed{seed}_test.json"
    evaluation = evaluate(best, data_root, evaluation_path, split="test", device=device)
    summary = {
        "code": code,
        "model": config["model"],
        "seed": seed,
        "data_root": str(Path(data_root).expanduser().resolve()),
        "checkpoint": str(best),
        "audit": str(audit_path),
        "test_report": str(evaluation_path),
        "metrics": evaluation["metrics"],
    }
    summary_path = reports / f"{code}_seed{seed}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit, train, dan evaluasi baseline YOLO26n tanpa eksperimen modifikasi."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/D0_yolo26n.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_baseline(
        args.data_root,
        args.output_root,
        args.config,
        seed=args.seed,
        device=args.device,
        resume=not args.no_resume,
    )
    print("\n=== BASELINE YOLO26n TEST ===")
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(f"SAVED: {result['summary']}")


if __name__ == "__main__":
    main()
