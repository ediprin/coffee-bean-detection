from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from coffee_detector.af2_rn import AF2RNConfig, make_af2rn_trainer
from coffee_detector.af2_rn.audit import sha256
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_rn/AF2RN_yolo26n.yaml"
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
EPS = 1.0e-12


def _read(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {metric: float(source[metric]) for metric in METRICS}


def _af2_metrics(payload: dict) -> dict[str, float]:
    if "candidate" in payload and "AF2" in payload["candidate"]:
        return _metrics(payload["candidate"]["AF2"])
    if "values" in payload and "AF2" in payload["values"]:
        return _metrics(payload["values"]["AF2"])
    return _metrics(payload)


def run_faruq_v3_af2rn(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    static_audit: str | Path,
    observability_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("AF2RN screening dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training AF2RN belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    load_faruq_grouped_summary(grouped_summary, data_root)
    static = _read(static_audit, "Static audit AF2RN")
    observability = _read(observability_audit, "Observability AF2RN")
    if (
        static.get("format") != "coffee_detector.af2rn.static_audit.v1"
        or static.get("decision") != "PASS"
        or static.get("d0_checkpoint_sha256") != sha256(checkpoint)
        or static.get("test_access_authorized") is not False
    ):
        raise RuntimeError("Static audit AF2RN tidak mengotorisasi checkpoint ini")
    if (
        observability.get("format") != "coffee_detector.af2rn.observability.v1"
        or observability.get("decision") != "PASS"
        or observability.get("training_authorized") is not True
        or observability.get("validation_images_accessed") is not False
        or observability.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Observability AF2RN belum mengotorisasi training")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = reports / "dataset_audit.json"
    if not dataset_audit.is_file():
        dataset = audit_dataset(data_root, dataset_audit, near_threshold=-1)
        if not dataset["safe_for_training"]:
            raise RuntimeError("Audit dataset gagal")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "AF2RN":
        raise RuntimeError("Config AF2RN tidak konsisten")
    frontend = AF2RNConfig.from_mapping(payload["af2rn"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / "AF2RN" / f"AF2RN_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "format": "coffee_detector.af2rn.run_contract.v1",
        "arm": "AF2RN",
        "seed": seed,
        "config_sha256": sha256(CONFIG),
        "d0_checkpoint_sha256": sha256(checkpoint),
        "static_audit_sha256": sha256(static_audit),
        "observability_audit_sha256": sha256(observability_audit),
        "epochs": epochs,
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _read(contract_path, "Run contract") != run_contract:
        raise RuntimeError(f"Run directory memiliki kontrak berbeda: {run_dir}")
    contract_path.write_text(json.dumps(run_contract, indent=2) + "\n", encoding="utf-8")

    training_executed = False
    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_af2rn_trainer(frontend, d0_checkpoint=checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root, lock_name=f"AF2RN_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME AF2RN dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                print(f"START AF2RN seed {seed} dari D0 seed {seed}", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root / "AF2RN"),
                    name=f"AF2RN_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=False,
                    verbose=False,
                    device=device,
                )
            model.train(trainer=trainer, **args)
        training_executed = True
    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run AF2RN belum lengkap: {run_dir}")

    report_path = reports / f"AF2RN_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    metrics = report["metrics"]
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.af2rn.arm_result.v1",
        "arm": "AF2RN",
        "seed": seed,
        "metrics": metrics,
        "checkpoint": str(best),
        "checkpoint_sha256": sha256(best),
        "initial_d0_checkpoint": str(checkpoint),
        "initial_d0_checkpoint_sha256": sha256(checkpoint),
        "config": str(CONFIG),
        "static_audit": str(Path(static_audit).expanduser().resolve()),
        "observability_audit": str(
            Path(observability_audit).expanduser().resolve()
        ),
        "training_executed_this_call": training_executed,
        "completed_epochs": epochs,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path = reports / f"AF2RN_seed{seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def run_af2rn_seed42_decision(
    af2rn_result: str | Path,
    af2_result: str | Path,
    output: str | Path,
) -> dict:
    candidate_payload = _read(af2rn_result, "Hasil AF2RN")
    baseline_payload = _read(af2_result, "Hasil AF2")
    if (
        candidate_payload.get("format") != "coffee_detector.af2rn.arm_result.v1"
        or candidate_payload.get("seed") != 42
        or candidate_payload.get("test_images_accessed") is not False
        or baseline_payload.get("test_images_accessed", False) is not False
    ):
        raise RuntimeError("Evidence AF2RN/AF2 tidak kompatibel")
    baseline = _af2_metrics(baseline_payload)
    candidate = _metrics(candidate_payload)
    deltas = {metric: candidate[metric] - baseline[metric] for metric in METRICS}
    criteria = {
        "macro_gain_at_least_0_5_point": deltas["macro_map50_95"] >= 0.005 - EPS,
        "bottom3_not_lower": deltas["bottom3_class_map50_95"] >= -EPS,
        "worst_drop_no_more_than_1_point": deltas["worst_class_map50_95"] >= -0.01 - EPS,
        "all_21_validation_classes_present": not bool(
            candidate_payload["metrics"].get("classes_without_ground_truth")
        ),
        "test_not_opened": True,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    result = {
        "format": "coffee_detector.af2rn.seed42_decision.v1",
        "seed": 42,
        "values": {"AF2C": baseline, "AF2RN": candidate},
        "deltas": deltas,
        "criteria": criteria,
        "decision": decision,
        "next": (
            "FREEZE_PAIRED_THREE_SEED_CONFIRMATION"
            if decision == "PASS"
            else "RETAIN_ORIGINAL_AF2_AND_STOP_AF2RN"
        ),
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AF2RN seed-42 screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--observability-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2rn(
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.static_audit,
        args.observability_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
