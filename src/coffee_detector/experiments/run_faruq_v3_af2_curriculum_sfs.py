from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_curriculum_sfs import (
    AF2CurriculumSFSConfig,
    make_af2_curriculum_sfs_trainer,
)
from coffee_detector.afab.operator import AFABConfig
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_curriculum_sfs/AF2CURR1_yolo26n.yaml"
ARM = "AF2CURR1"
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _control(path: str | Path, checkpoint_sha: str) -> dict:
    result = _read(path, "AF2CTRL result")
    if result.get("format") != "coffee_detector.af2_complement.arm_result.v1":
        raise RuntimeError("Control bukan result AF2 complement yang dibekukan")
    if result.get("arm") != "AF2CTRL" or int(result.get("seed", -1)) != 42:
        raise RuntimeError("Control wajib AF2CTRL seed 42")
    if result.get("initial_af2_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("AF2CTRL dan kandidat tidak memakai parent AF2 yang sama")
    if result.get("test_images_accessed") is not False:
        raise RuntimeError("AF2CTRL membuka test")
    metrics = result.get("metrics", {})
    if not all(metric in metrics for metric in METRICS):
        raise RuntimeError("AF2CTRL kehilangan headline metrics")
    return result


def _decision(control: dict, candidate: dict) -> dict:
    deltas = {
        metric: float(candidate["metrics"][metric] - control["metrics"][metric])
        for metric in METRICS
    }
    macro_route = bool(
        deltas["macro_map50_95"] >= 0.005
        and deltas["bottom3_class_map50_95"] >= 0.0
        and deltas["worst_class_map50_95"] >= -0.010
    )
    tail_route = bool(
        deltas["macro_map50_95"] >= -0.001
        and deltas["bottom3_class_map50_95"] >= 0.010
        and deltas["worst_class_map50_95"] >= 0.010
    )
    return {
        "deltas": deltas,
        "criteria": {
            "macro_route": macro_route,
            "lower_tail_pareto_route": tail_route,
            "all_21_validation_classes_present": True,
            "test_not_opened": True,
        },
        "decision": "RETAIN_SEED42" if macro_route or tail_route else "FAIL_KILL_GATE",
        "next": "FREEZE_PAIRED_CONFIRMATION" if macro_route or tail_route else "RETAIN_ORIGINAL_AF2",
    }


def run_faruq_v3_af2_curriculum_sfs(
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_checkpoint: str | Path,
    af2ctrl_result: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("Screening dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    root = Path(data_root).expanduser().resolve()
    grouped = Path(grouped_summary).expanduser().resolve()
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    audit_path = Path(static_audit).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    if not checkpoint.is_file() or not (root / "data.yaml").is_file():
        raise FileNotFoundError("Dataset atau checkpoint AF2 tidak lengkap")
    load_faruq_grouped_summary(grouped, root)
    reports = destination / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(root, reports / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Dataset audit gagal")

    checkpoint_sha = _sha256(checkpoint)
    control = _control(af2ctrl_result, checkpoint_sha)
    audit = _read(audit_path, "Static audit")
    if audit.get("format") != "coffee_detector.af2_curriculum_sfs.static_audit.v1":
        raise RuntimeError("Static audit salah schema")
    if audit.get("decision") != "PASS" or audit.get("training_authorized") is not True:
        raise RuntimeError("Static audit belum PASS")
    if audit.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("Checkpoint berbeda dari static audit")
    if audit.get("test_images_accessed") is not False:
        raise RuntimeError("Static audit tidak mempertahankan test lock")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    afab = AFABConfig.from_mapping(payload["afab"])
    curriculum = AF2CurriculumSFSConfig.from_mapping(payload["curriculum"])
    train_args = dict(payload["train"])
    epochs = int(train_args["epochs"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.af2_curriculum_sfs.arm_contract.v1",
        "protocol": "faruq-v3-af2-curriculum-sfs-seed42-v1",
        "arm": ARM,
        "seed": seed,
        "initial_af2_checkpoint_sha256": checkpoint_sha,
        "af2ctrl_checkpoint_sha256": control.get("checkpoint_sha256"),
        "config_sha256": _sha256(CONFIG),
        "train": train_args,
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _read(contract_path, "Run contract") != contract:
        raise RuntimeError("Run directory memiliki kontrak berbeda")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    result_path = reports / f"{ARM}_seed{seed}_result.json"
    if result_path.is_file():
        result = _read(result_path, "Candidate result")
        if result.get("run_contract") != contract:
            raise RuntimeError("Result lama memiliki kontrak berbeda")
    else:
        best = run_dir / "weights/best.pt"
        last = run_dir / "weights/last.pt"
        trainer = make_af2_curriculum_sfs_trainer(
            afab, curriculum, initial_checkpoint=checkpoint
        )
        training_executed = False
        if not _run_complete(run_dir, epochs):
            from ultralytics import YOLO

            epoch, resumable = _checkpoint_state(last)
            with _exclusive_training_lock(
                destination, lock_name=f"{ARM}_seed{seed}.training.lock"
            ):
                if last.is_file() and resumable and epoch is not None and epoch >= 0:
                    print(f"RESUME {ARM} seed {seed} dari epoch {epoch + 1}", flush=True)
                    model = YOLO(str(last))
                    args = {"resume": True, "device": device}
                else:
                    if last.is_file() and not resumable:
                        raise RuntimeError("last.pt ada tetapi tidak resumable")
                    print(f"START {ARM} seed {seed} dari AF2 parent", flush=True)
                    model = YOLO(str(REPO_ROOT / payload["model"]))
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
        if not _run_complete(run_dir, epochs):
            raise RuntimeError("Training belum complete/early-stopped secara valid")
        evaluation = evaluate(
            best, root, reports / f"{ARM}_seed{seed}_val.json", split="val", device=device
        )
        metrics = evaluation["metrics"]
        if metrics.get("classes_without_ground_truth"):
            raise RuntimeError("Validation kehilangan kelas")
        result = {
            "format": "coffee_detector.af2_curriculum_sfs.arm_result.v1",
            "protocol": "faruq-v3-af2-curriculum-sfs-seed42-v1",
            "arm": ARM,
            "seed": seed,
            "metrics": {metric: float(metrics[metric]) for metric in METRICS},
            "checkpoint": str(best),
            "checkpoint_sha256": _sha256(best),
            "completed_epochs": _epochs(run_dir / "results.csv"),
            "maximum_epochs": epochs,
            "training_executed_this_call": training_executed,
            "evaluation_split": "val",
            "test_images_accessed": False,
            "run_contract": contract,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    decision = _decision(control, result)
    summary = {
        "format": "coffee_detector.af2_curriculum_sfs.seed42_decision.v1",
        "protocol": "faruq-v3-af2-curriculum-sfs-seed42-v1",
        "seed": seed,
        "values": {
            "AF2CTRL": {metric: float(control["metrics"][metric]) for metric in METRICS},
            ARM: result["metrics"],
        },
        **decision,
        "test_opened": False,
    }
    summary_path = reports / "af2_curriculum_sfs_seed42_decision.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2 curriculum-SFS seed-42 screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--af2ctrl-result", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2_curriculum_sfs(
        args.data_root,
        args.grouped_summary,
        args.af2_checkpoint,
        args.af2ctrl_result,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
