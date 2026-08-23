from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.experiments.run_faruq_v3_af2_cpe_arm import run_arm
from coffee_detector.experiments.run_faruq_v3_af2_cpe_decision import decide
from coffee_detector.experiments.run_faruq_v3_af2_cpe_static import run_static_audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_worker(
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("Seed-42 screening worker hanya mengizinkan seed 42")
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training")

    data_root = Path(data_root).resolve()
    grouped_summary = Path(grouped_summary).resolve()
    af2_checkpoint = Path(af2_checkpoint).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir = output_root / "val_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_root / "static_audit.json"

    print("=== AF2+CPE0 SEED42 WORKER ===", flush=True)
    print(f"DATA_ROOT      : {data_root}", flush=True)
    print(f"GROUPED_SUMMARY: {grouped_summary}", flush=True)
    print(f"AF2_CHECKPOINT : {af2_checkpoint}", flush=True)
    print(f"OUTPUT_ROOT    : {output_root}", flush=True)
    print(f"DEVICE         : {device}", flush=True)

    print("\n[PHASE 1/4] STATIC AUDIT", flush=True)
    audit = run_static_audit(af2_checkpoint, audit_path, device=device)
    print(json.dumps(audit, indent=2), flush=True)
    if audit.get("decision") != "PASS" or audit.get("training_authorized") is not True:
        raise RuntimeError(f"Static audit tidak PASS: {audit_path}")

    results: dict[str, dict] = {}
    for phase, arm in ((2, "AF2CPE0"), (3, "AF2CPE5")):
        print(f"\n[PHASE {phase}/4] TRAIN/RESUME + VAL {arm}", flush=True)
        result = run_arm(
            arm,
            data_root,
            grouped_summary,
            af2_checkpoint,
            audit_path,
            output_root,
            seed=42,
            device=device,
            authorize_training=True,
        )
        results[arm] = result
        metrics = result["metrics"]
        print(
            f"[{arm}] Macro={float(metrics['macro_map50_95']):.6f} "
            f"B3={float(metrics['bottom3_class_map50_95']):.6f} "
            f"Worst={float(metrics['worst_class_map50_95']):.6f}",
            flush=True,
        )

    print("\n[PHASE 4/4] FROZEN DECISION", flush=True)
    decision = decide(results["AF2CPE0"], results["AF2CPE5"])
    decision_path = output_root / "decision_seed42.json"
    _write_json(decision_path, decision)
    print(json.dumps(decision, indent=2), flush=True)

    summary = {
        "format": "coffee_detector.af2_cpe.seed42_worker.v1",
        "seed": 42,
        "static_audit": str(audit_path),
        "AF2CPE0": results["AF2CPE0"]["metrics"],
        "AF2CPE5": results["AF2CPE5"]["metrics"],
        "decision": decision,
        "decision_path": str(decision_path),
        "test_images_accessed": False,
    }
    worker_path = report_dir / "af2_cpe_seed42_worker.json"
    _write_json(worker_path, summary)
    print(f"WORKER SUMMARY: {worker_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen AF2+CPE0 seed42 worker")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_worker(
        args.data_root,
        args.grouped_summary,
        args.af2_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
