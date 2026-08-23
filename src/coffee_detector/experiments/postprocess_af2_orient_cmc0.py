from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from coffee_detector.af2_iso import frozen_arm_config
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_af2_orient_cmc0 import (
    ARM,
    METRICS,
    _decision,
    _load_af2_orient_parent,
    _metric_triplet,
    _sha256,
)
from coffee_detector.experiments.run_faruq_v3_af2_iso_arm import _latency
from coffee_detector.stb import STBConfig


def _history(results_csv: Path) -> dict:
    if not results_csv.is_file():
        return {
            "epochs": [],
            "duplicate_epochs": [],
            "strictly_increasing_unique": False,
            "note": "results.csv missing",
        }
    with results_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(float(row["epoch"])) for row in rows]
    counts = Counter(epochs)
    duplicate_epochs = sorted(epoch for epoch, count in counts.items() if count > 1)
    strictly_increasing_unique = all(b > a for a, b in zip(epochs, epochs[1:]))
    return {
        "epochs": epochs,
        "row_count": len(epochs),
        "first_epoch": epochs[0] if epochs else None,
        "last_epoch": epochs[-1] if epochs else None,
        "duplicate_epochs": duplicate_epochs,
        "strictly_increasing_unique": strictly_increasing_unique,
        "note": (
            "clean"
            if strictly_increasing_unique
            else "resume-history anomaly; metrics may be inspected but the run is not decision-clean"
        ),
    }


def postprocess(
    data_root: str | Path,
    d0_checkpoint: str | Path,
    af2_orient_result: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    latency_iterations: int = 50,
) -> dict:
    if seed != 42:
        raise ValueError("AF2_ORIENT+CMC0 screen dikunci seed 42")

    data_root = Path(data_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    af2_orient_result = Path(af2_orient_result).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos split test")

    run_dir = output_root / f"{ARM}_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError("Postprocess memerlukan best.pt dan last.pt dari run selesai")

    parent_payload = _load_af2_orient_parent(af2_orient_result, d0_checkpoint, seed)
    parent_metrics = _metric_triplet(parent_payload)

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"{ARM}_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    if len(report["metrics"].get("map50_95_by_class", {})) != 21:
        raise RuntimeError("Validation tidak memuat seluruh 21 kelas")

    candidate_metrics = _metric_triplet(report)
    screen = _decision(candidate_metrics, parent_metrics)
    history = _history(run_dir / "results.csv")
    decision_clean = bool(history["strictly_increasing_unique"])
    latency = _latency(best, device, iterations=latency_iterations)

    result = {
        "format": "coffee_detector.af2_orient_cmc0.postprocessed_seed42_result.v1",
        "arm": ARM,
        "seed": seed,
        "candidate": candidate_metrics,
        "parent": {"AF2_ORIENT": parent_metrics},
        "comparison": screen,
        "metrics": report["metrics"],
        "latency": latency,
        "execution_history": history,
        "decision_clean": decision_clean,
        "decision": screen["decision"] if decision_clean else "REVIEW_RESUME_HISTORY",
        "raw_screen_decision": screen["decision"],
        "next_action": (
            "AUTHORIZE_PAIRED_MULTI_SEED"
            if decision_clean and screen["decision"] == "PASS"
            else ("STOP_NO_TUNING" if decision_clean else "REVIEW_RESUME_HISTORY_NO_MULTI_SEED")
        ),
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "initial_d0_checkpoint": str(d0_checkpoint),
        "initial_d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "af2_orient_parent_result": str(af2_orient_result),
        "af2_orient_parent_result_sha256": _sha256(af2_orient_result),
        "af2_operator": frozen_arm_config("AF2_ORIENT").to_dict(),
        "cmc0": STBConfig().to_dict(),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "scientific_status": (
            "postprocessed completed checkpoint; resume history is explicitly audited and blocks "
            "confirmation if non-monotonic"
        ),
    }
    result_path = reports / f"{ARM}_seed{seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess completed AF2_ORIENT+CMC0 seed42 run")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--af2-orient-result", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--latency-iterations", type=int, default=50)
    args = parser.parse_args()
    postprocess(
        args.data_root,
        args.d0_checkpoint,
        args.af2_orient_result,
        args.output_root,
        seed=args.seed,
        device=args.device,
        latency_iterations=args.latency_iterations,
    )


if __name__ == "__main__":
    main()
