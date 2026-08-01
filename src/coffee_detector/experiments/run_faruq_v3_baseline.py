from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.run_baseline import is_training_complete
from coffee_detector.train import train_experiment


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/coffee_fg/D0_yolo26n_p3.yaml"


def load_faruq_grouped_summary(path: str | Path, data_root: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Summary Faruq-v3 tidak ditemukan: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "coffee_detector.faruq_grouped_development.v1":
        raise ValueError("Format summary bukan Faruq grouped development v1")
    if not payload.get("training_ready"):
        raise RuntimeError("Faruq-v3 belum melewati seluruh training gate")
    if payload.get("cross_split_parent_identities") != 0:
        raise RuntimeError("Parent identity leakage masih ditemukan")
    if payload.get("cross_split_exact_hashes") != 0:
        raise RuntimeError("Exact-hash leakage masih ditemukan")
    if payload.get("test_images_accessed") is not False:
        raise RuntimeError("Provenance test lock tidak valid")
    recorded_root = Path(payload.get("output_root", "")).expanduser().resolve()
    if recorded_root != data_root:
        raise RuntimeError(
            f"Summary berasal dari dataset berbeda: {recorded_root} != {data_root}"
        )
    return payload


def run_faruq_v3_baseline(
    data_root: str | Path,
    grouped_summary: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    grouped = load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq development tidak boleh memiliki split test")

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    audit_path = reports_root / "dataset_audit.json"
    audit = audit_dataset(data_root, audit_path, near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError(f"Audit dataset gagal: {audit_path}")

    run_dir = output_root / f"D0_seed{seed}"
    training_was_run = not is_training_complete(run_dir)
    if training_was_run:
        action = "RESUME" if (run_dir / "weights/last.pt").is_file() else "START"
        print(f"{action} FARUQ-V3 BASELINE | seed={seed}", flush=True)
        train_experiment(
            CONFIG,
            data_root,
            output_root,
            seed,
            device=device,
            resume=True,
        )
    else:
        print(f"SKIP TRAINING: D0 seed {seed} sudah lengkap", flush=True)

    checkpoint = run_dir / "weights/best.pt"
    manifest = run_dir / "experiment_manifest.json"
    if not checkpoint.is_file() or not manifest.is_file():
        raise FileNotFoundError("Training belum lengkap: best.pt/manifest belum tersedia")
    report_path = reports_root / f"D0_seed{seed}_val.json"
    report = evaluate(
        checkpoint, data_root, report_path, split="val", device=device
    )
    missing = report["metrics"].get("classes_without_ground_truth", [])
    if missing:
        raise RuntimeError("Validation kehilangan kelas: " + ", ".join(missing))

    summary = {
        "protocol": "faruq-v3-yolo26n-baseline-v1",
        "model": "D0 pinned YOLO26n P3-P5",
        "seed": seed,
        "data_root": str(data_root),
        "grouped_summary": str(Path(grouped_summary).expanduser().resolve()),
        "grouped_dataset": {
            "images_by_split": grouped["images_by_split"],
            "annotations_by_split": grouped["annotations_by_split"],
            "minimum_val_class_support": grouped["minimum_val_class_support"],
        },
        "checkpoint": str(checkpoint),
        "validation_report": str(report_path),
        "metrics": report["metrics"],
        "training_complete": True,
        "training_executed_this_call": training_was_run,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
    }
    summary_path = reports_root / f"D0_seed{seed}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the locked validation-only YOLO26n baseline on Faruq-v3."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = run_faruq_v3_baseline(
        args.data_root,
        args.grouped_summary,
        args.output_root,
        seed=args.seed,
        device=args.device,
    )
    print("\n=== FARUQ-V3 YOLO26n BASELINE ===")
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(f"SAVED: {result['summary']}")


if __name__ == "__main__":
    main()
