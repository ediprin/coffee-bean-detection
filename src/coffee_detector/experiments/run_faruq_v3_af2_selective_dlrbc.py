from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from coffee_detector.afab.operator import AFABConfig
from coffee_detector.af2_selective_dlrbc.audit import run_af2_selective_static_audit
from coffee_detector.af2_selective_dlrbc.model import AF2SelectiveDLRBCConfig
from coffee_detector.af2_selective_dlrbc.trainer import make_af2_selective_trainer
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_selective_dlrbc/AF2CSD1.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
PROTOCOL = "faruq-v3-af2-class-selective-dlrbc-seed42-v1"
ARM = "AF2CSD1"
SEED = 42
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _load_protocol_inputs(
    complementarity_path: str | Path,
    af2_summary_path: str | Path,
    af2_val_report_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    complementarity = _json(complementarity_path, "Complementarity audit")
    summary = _json(af2_summary_path, "AF2 direct summary")
    val_report = _json(af2_val_report_path, "AF2 validation report")
    if complementarity.get("decision") != "AUTHORIZE_AF2CSD1":
        raise RuntimeError("Train-only complementarity audit tidak mengotorisasi training")
    if complementarity.get("validation_accessed") is not False or complementarity.get("test_images_accessed") is not False:
        raise RuntimeError("Complementarity audit melanggar split lock")
    if summary.get("protocol") != "faruq-v3-af2-direct-from-pretrained-seed42-v1":
        raise RuntimeError("Baseline bukan AF2DIRECT fresh protocol")
    candidate = summary.get("candidate", {})
    if candidate.get("arm") != "AF2DIRECT" or candidate.get("seed") != SEED:
        raise RuntimeError("Baseline wajib AF2DIRECT seed 42")
    if summary.get("test_images_accessed") is not False or val_report.get("split") != "val":
        raise RuntimeError("AF2 baseline report melanggar split contract")
    return complementarity, summary, val_report


def run_faruq_v3_af2_selective_dlrbc(
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_checkpoint: str | Path,
    af2_summary: str | Path,
    af2_val_report: str | Path,
    complementarity_audit: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
    authorize_training: bool = False,
) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError("Screen pertama dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    root = Path(data_root).expanduser().resolve()
    if (root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    load_faruq_grouped_summary(Path(grouped_summary).expanduser().resolve(), root)
    parent = Path(af2_checkpoint).expanduser().resolve()
    if not parent.is_file():
        raise FileNotFoundError(parent)
    complementarity, baseline_summary, baseline_val = _load_protocol_inputs(
        complementarity_audit, af2_summary, af2_val_report
    )
    baseline = baseline_summary["candidate"]
    if _sha256(parent) != baseline.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint AF2DIRECT tidak cocok dengan summary")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    afab = AFABConfig.from_mapping(config["afab"])
    selective = AF2SelectiveDLRBCConfig.from_mapping(
        {**config["selective"], "selected_class_ids": complementarity["selected_class_ids"]}
    )
    destination = Path(output_root).expanduser().resolve()
    reports = destination / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(root, reports / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Dataset audit gagal")

    static_path = Path(static_audit).expanduser().resolve()
    static = (
        _json(static_path, "Static audit")
        if static_path.is_file()
        else run_af2_selective_static_audit(
            MODEL_YAML, parent, afab, list(selective.selected_class_ids), static_path, device="cpu"
        )
    )
    if static.get("decision") != "PASS" or static.get("selected_class_ids") != list(selective.selected_class_ids):
        raise RuntimeError("Static audit tidak cocok atau bukan PASS")

    train_args = dict(config["train"])
    maximum_epochs = int(train_args["epochs"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = reports / f"{ARM}_seed{seed}_result.json"
    contract = {
        "format": "coffee_detector.af2_selective_dlrbc.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "af2_checkpoint_sha256": _sha256(parent),
        "selected_class_ids": list(selective.selected_class_ids),
        "config_sha256": _sha256(CONFIG),
        "train": train_args,
        "frozen_parent": True,
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if contract_path.is_file() and _json(contract_path, "Run contract") != contract:
        raise RuntimeError("Run directory mempunyai kontrak berbeda")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    if result_path.is_file():
        previous = _json(result_path, "Result")
        if previous.get("run_contract") != contract:
            raise RuntimeError("Result lama mempunyai kontrak berbeda")
        return previous

    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False
    if not _run_complete(run_dir, maximum_epochs):
        from ultralytics import YOLO

        epoch, resumable = _checkpoint_state(last)
        trainer = make_af2_selective_trainer(
            af2_checkpoint=parent, afab=afab, selective=selective
        )
        with _exclusive_training_lock(destination, lock_name=f"{ARM}_seed{seed}.training.lock"):
            if last.is_file() and resumable and epoch is not None:
                print(f"RESUME {ARM} seed {seed} dari epoch {epoch + 1}", flush=True)
                model = YOLO(str(last))
                args = {"resume": True, "device": device}
            else:
                if last.is_file():
                    raise RuntimeError("last.pt ada tetapi tidak resumable")
                print(f"START {ARM} seed {seed} dari AF2DIRECT frozen parent", flush=True)
                model = YOLO(str(MODEL_YAML))
                args = dict(train_args)
                args.update(
                    data=str(root / "data.yaml"),
                    project=str(destination / ARM),
                    name=f"{ARM}_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=False,
                    verbose=False,
                    device=device,
                )
            model.train(trainer=trainer, **args)
        training_executed = True
    if not _run_complete(run_dir, maximum_epochs) or not best.is_file():
        raise RuntimeError("Training belum selesai secara valid")

    evaluation = evaluate(best, root, reports / f"{ARM}_seed{seed}_val.json", split="val", device=device)
    metrics = evaluation["metrics"]
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    baseline_metrics = baseline["metrics"]
    deltas = {name: float(metrics[name]) - float(baseline_metrics[name]) for name in METRICS}
    baseline_class = baseline_val["metrics"]["map50_95_by_class"]
    candidate_class = metrics["map50_95_by_class"]
    selected_names = complementarity["selected_class_names"]
    selected_gain = sum(candidate_class[name] - baseline_class[name] for name in selected_names) / len(selected_names)
    gates = {
        "macro_drop_no_more_than_0_1_point": deltas["macro_map50_95"] >= -0.001,
        "bottom3_not_lower": deltas["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_0_5_point": deltas["worst_class_map50_95"] >= -0.005,
        "selected_class_mean_gain_at_least_0_5_point": selected_gain >= 0.005,
        "all_21_classes_present": not metrics.get("classes_without_ground_truth"),
        "test_not_opened": True,
    }
    decision = "RETAIN_SEED42_PARETO" if all(gates.values()) else "STOP_AFTER_SEED42"
    result = {
        "format": "coffee_detector.af2_selective_dlrbc.arm_result.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "baseline": {name: float(baseline_metrics[name]) for name in METRICS},
        "candidate": {name: float(metrics[name]) for name in METRICS},
        "deltas": deltas,
        "selected_class_ids": list(selective.selected_class_ids),
        "selected_class_names": selected_names,
        "selected_class_mean_map_delta": selected_gain,
        "gates": gates,
        "decision": decision,
        "next": "AUTHORIZE_PAIRED_MULTISEED" if decision == "RETAIN_SEED42_PARETO" else "STOP_WITHOUT_TEST_OR_EXTRA_SEEDS",
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "completed_epochs": _epochs(run_dir / "results.csv"),
        "training_executed_this_call": training_executed,
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2 parent plus class-selective DLRBC residual")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--af2-summary", required=True)
    parser.add_argument("--af2-val-report", required=True)
    parser.add_argument("--complementarity-audit", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_af2_selective_dlrbc(
        args.data_root,
        args.grouped_summary,
        args.af2_checkpoint,
        args.af2_summary,
        args.af2_val_report,
        args.complementarity_audit,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps({key: result[key] for key in ("baseline", "candidate", "deltas", "selected_class_mean_map_delta", "gates", "decision", "next")}, indent=2))


if __name__ == "__main__":
    main()
