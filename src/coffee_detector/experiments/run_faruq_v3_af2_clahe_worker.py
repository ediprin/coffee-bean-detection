"""Single-seed execution worker for the frozen AF2-vs-CLAHE control.

This changes execution only, not the scientific protocol. One process trains exactly
one frozen CLAHE_LAB seed from its seed-matched D0 checkpoint and writes only that
seed's validation report. Three independent runtimes may therefore execute seeds
42, 123, and 2026 concurrently without sharing a training directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.experiments.run_faruq_v3_af2_clahe_control import (
    SEEDS,
    _checkpoint_seed,
    _metrics,
    _train_clahe,
    _validate_reference,
)
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary


def run_worker(
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_confirmation: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(f"Seed worker harus salah satu dari {SEEDS}, diterima {seed}")
    if not authorize_training:
        raise RuntimeError("Parallel CLAHE worker belum diotorisasi")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)

    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    audit = audit_dataset(
        data_root,
        reports / f"dataset_audit_seed{seed}.json",
        near_threshold=-1,
    )
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    reference = _validate_reference(af2_confirmation)
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"D0 seed {seed} tidak ditemukan: {checkpoint}")
    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError(f"Checkpoint D0 tidak cocok dengan seed {seed}")

    report, trained, recovery = _train_clahe(
        data_root,
        checkpoint,
        output_root,
        seed=seed,
        device=device,
    )
    frozen = reference["per_seed"][str(seed)]
    result = {
        "protocol": "faruq-v3-af2-vs-clahe-classical-enhancement-control-v1",
        "execution_mode": "parallel_single_seed_worker",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "D0FT": _metrics(frozen["D0FT"]),
        "AF2": _metrics(frozen["AF2"]),
        "CLAHE_LAB": _metrics(report),
        "training_executed_this_call": trained,
        "recovery": recovery,
        "report": str(reports / f"CLAHE_LAB_seed{seed}_val.json"),
    }
    worker_summary = reports / f"CLAHE_LAB_seed{seed}_worker.json"
    worker_summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["worker_summary"] = str(worker_summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel single-seed CLAHE worker")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-confirmation", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_worker(
        args.data_root,
        args.grouped_summary,
        args.af2_confirmation,
        args.d0_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
