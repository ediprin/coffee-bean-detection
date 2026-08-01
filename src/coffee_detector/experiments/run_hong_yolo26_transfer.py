from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.analysis.benchmark_coffee_fg import benchmark_checkpoint
from coffee_detector.analysis.faruq_v3_diagnostics import run_faruq_v3_diagnostics
from coffee_detector.analysis.faruq_v3_operational_audit import (
    audit_faruq_v3_fixed_operating_point,
)
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)
from coffee_detector.hong_transfer.audit import static_architecture_audit
from coffee_detector.run_baseline import is_training_complete
from coffee_detector.train import (
    recover_completed_training_manifest,
    train_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/hong/HF_yolo26n_full_hong_transfer.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
DEFAULT_WEIGHTS = REPO_ROOT / "yolo26n.pt"


def _delta(left: float, right: float) -> float:
    return float(right) - float(left)


def _decision(
    baseline_metrics: dict,
    candidate_metrics: dict,
    baseline_diagnostic: dict,
    candidate_diagnostic: dict,
    baseline_operational: dict,
    candidate_operational: dict,
    baseline_efficiency: dict,
    candidate_efficiency: dict,
) -> dict:
    deltas = {
        "macro_map50_95": _delta(
            baseline_metrics["macro_map50_95"],
            candidate_metrics["macro_map50_95"],
        ),
        "conditional_top1_accuracy": _delta(
            baseline_diagnostic["global"]["localization_conditioned_class_accuracy"],
            candidate_diagnostic["global"]["localization_conditioned_class_accuracy"],
        ),
        "bottom3_class_map50_95": _delta(
            baseline_metrics["bottom3_class_map50_95"],
            candidate_metrics["bottom3_class_map50_95"],
        ),
        "worst_class_map50_95": _delta(
            baseline_metrics["worst_class_map50_95"],
            candidate_metrics["worst_class_map50_95"],
        ),
        "proposal_accessibility": _delta(
            baseline_diagnostic["global"]["proposal_accessibility"],
            candidate_diagnostic["global"]["proposal_accessibility"],
        ),
        "operational_correct_decision_f1": _delta(
            baseline_operational["result"]["correct_decision_f1"],
            candidate_operational["result"]["correct_decision_f1"],
        ),
    }
    latency_ratio = (
        candidate_efficiency["latency_ms_per_image"]
        / baseline_efficiency["latency_ms_per_image"]
    )
    criteria = {
        "macro_gain_at_least_0_5_point": deltas["macro_map50_95"] >= 0.005,
        "conditional_top1_gain_at_least_2_points": (
            deltas["conditional_top1_accuracy"] >= 0.02
        ),
        "bottom3_drop_no_more_than_1_point": (
            deltas["bottom3_class_map50_95"] >= -0.01
        ),
        "worst_drop_no_more_than_2_points": (
            deltas["worst_class_map50_95"] >= -0.02
        ),
        "proposal_drop_no_more_than_1_point": (
            deltas["proposal_accessibility"] >= -0.01
        ),
        "operational_f1_not_lower": (
            deltas["operational_correct_decision_f1"] >= 0.0
        ),
        "latency_increase_no_more_than_25_percent": latency_ratio <= 1.25,
    }
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "deltas": deltas,
        "latency_ratio": float(latency_ratio),
        "criteria": criteria,
    }


def run_hong_yolo26_transfer(
    data_root: str | Path,
    grouped_summary: str | Path,
    baseline_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    weights: str | Path = DEFAULT_WEIGHTS,
) -> dict:
    """Run the one-seed validation-only full Hong fail-fast screen."""

    if int(seed) != 42:
        raise RuntimeError("Tahap awal Hong dikunci hanya pada seed 42")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    baseline_checkpoint = Path(baseline_checkpoint).expanduser().resolve()
    weights = Path(weights).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Faruq development tidak boleh menyediakan test")
    if not baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint D0 tidak ditemukan: {baseline_checkpoint}")
    if not weights.is_file():
        raise FileNotFoundError(f"Bobot pretrained tidak ditemukan: {weights}")
    load_faruq_grouped_summary(grouped_summary, data_root)

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(
        data_root, reports / "dataset_audit.json", near_threshold=-1
    )
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Dataset Faruq-v3 gagal audit sebelum Hong screen")

    static_path = reports / "HF_seed42_architecture_audit.json"
    static_report = static_architecture_audit(
        MODEL_YAML,
        static_path,
        nc=21,
        weights=weights,
        image_size=128,
    )
    if static_report["static_gate"] != "PASS":
        raise RuntimeError("Static architecture gate Hong gagal")

    run_dir = output_root / "HF_seed42"
    recover_completed_training_manifest(CONFIG, data_root, run_dir, 42)
    training_executed = not is_training_complete(run_dir)
    if training_executed:
        action = "RESUME" if (run_dir / "weights/last.pt").is_file() else "START"
        print(f"{action} TRAINING: HF full Hong transfer | seed=42", flush=True)
        train_experiment(
            CONFIG,
            data_root,
            output_root,
            42,
            device=device,
            resume=True,
        )
    else:
        print("SKIP TRAINING: HF seed42 lengkap", flush=True)
    candidate_checkpoint = run_dir / "weights/best.pt"
    if not candidate_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint HF tidak ditemukan: {candidate_checkpoint}")

    baseline_eval = evaluate(
        baseline_checkpoint,
        data_root,
        reports / "D0_seed42_val.json",
        split="val",
        device=device,
    )
    candidate_eval = evaluate(
        candidate_checkpoint,
        data_root,
        reports / "HF_seed42_val.json",
        split="val",
        device=device,
    )
    for label, payload in (("D0", baseline_eval), ("HF", candidate_eval)):
        missing = payload["metrics"].get("classes_without_ground_truth", [])
        if missing:
            raise RuntimeError(f"{label} validation kehilangan kelas: {missing}")

    baseline_diagnostic = run_faruq_v3_diagnostics(
        baseline_checkpoint,
        data_root,
        reports / "D0_seed42_diagnostic.json",
        split="val",
        device=device,
    )
    candidate_diagnostic = run_faruq_v3_diagnostics(
        candidate_checkpoint,
        data_root,
        reports / "HF_seed42_diagnostic.json",
        split="val",
        device=device,
    )
    baseline_operational = audit_faruq_v3_fixed_operating_point(
        baseline_checkpoint,
        data_root,
        reports / "D0_seed42_operational_fixed.json",
        device=device,
    )
    candidate_operational = audit_faruq_v3_fixed_operating_point(
        candidate_checkpoint,
        data_root,
        reports / "HF_seed42_operational_fixed.json",
        device=device,
    )
    baseline_efficiency = benchmark_checkpoint(
        baseline_checkpoint,
        image_size=640,
        batch_size=1,
        warmup=20,
        iterations=100,
        device=device,
    )
    candidate_efficiency = benchmark_checkpoint(
        candidate_checkpoint,
        image_size=640,
        batch_size=1,
        warmup=20,
        iterations=100,
        device=device,
    )
    decision = _decision(
        baseline_eval["metrics"],
        candidate_eval["metrics"],
        baseline_diagnostic,
        candidate_diagnostic,
        baseline_operational,
        candidate_operational,
        baseline_efficiency,
        candidate_efficiency,
    )
    payload = {
        "protocol": "HONG-YOLO26-TRANSFER-v1.2.0",
        "seed": 42,
        "evaluation_split": "val",
        "training_executed_this_call": training_executed,
        "test_images_accessed": False,
        "test_opened": False,
        "static_architecture_audit": str(static_path),
        "checkpoints": {
            "D0": str(baseline_checkpoint),
            "HF": str(candidate_checkpoint),
        },
        "metrics": {
            "D0": baseline_eval["metrics"],
            "HF": candidate_eval["metrics"],
        },
        "diagnostics": {
            "D0": baseline_diagnostic["global"],
            "HF": candidate_diagnostic["global"],
        },
        "operational": {
            "D0": baseline_operational["result"],
            "HF": candidate_operational["result"],
        },
        "efficiency": {
            "D0": baseline_efficiency,
            "HF": candidate_efficiency,
        },
        "decision": decision,
        "next_action": (
            "run_conditional_mechanism_controls"
            if decision["decision"] == "PASS"
            else "stop_hong_transfer_without_test_or_extra_seeds"
        ),
    }
    summary = reports / "hong_full_transfer_seed42_decision.json"
    summary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-seed validation-only full Hong transfer on Faruq-v3"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    args = parser.parse_args()
    result = run_hong_yolo26_transfer(
        args.data_root,
        args.grouped_summary,
        args.baseline_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        weights=args.weights,
    )
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
    print("NEXT:", result["next_action"])
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
