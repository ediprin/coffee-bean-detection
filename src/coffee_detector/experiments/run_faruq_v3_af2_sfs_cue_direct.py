from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from coffee_detector.af2_sfs_cue import (
    AF2SFSCUEConfig,
    make_af2_sfs_cue_direct_trainer,
    run_af2_sfs_cue_direct_static_audit,
)
from coffee_detector.afab.operator import AFABConfig
from coffee_detector.analysis.faruq_v3_diagnostics import run_faruq_v3_diagnostics
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_af2_direct import (
    METRICS,
    MODEL_YAML,
    _load_json,
    _load_yaml,
    _require_official_pretrained,
)
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_sfs_cue_direct/AF2SFSCUE1_yolo26n.yaml"
ARM = "AF2SFSCUE1"
SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _historical_af2direct(path: str | Path) -> dict:
    payload = _load_json(path, "Historical AF2DIRECT summary")
    if payload.get("protocol") != "faruq-v3-af2-direct-from-pretrained-seed42-v1":
        raise RuntimeError("Pembanding bukan protokol AF2DIRECT direct-from-pretrained")
    candidate = payload.get("candidate", payload)
    if candidate.get("arm") != "AF2DIRECT" or candidate.get("seed") != 42:
        raise RuntimeError("Pembanding wajib AF2DIRECT seed 42")
    if candidate.get("evaluation_split") != "val":
        raise RuntimeError("Pembanding bukan development validation")
    if candidate.get("test_images_accessed") is not False:
        raise RuntimeError("Pembanding membuka test")
    metrics = candidate.get("metrics", {})
    if not all(name in metrics for name in METRICS):
        raise RuntimeError("Pembanding kehilangan headline metrics")
    return candidate


def _screen(historical: dict, candidate: dict) -> dict:
    deltas = {
        metric: float(candidate["metrics"][metric] - historical["metrics"][metric])
        for metric in METRICS
    }
    macro_route = bool(
        deltas["macro_map50_95"] >= 0.005
        and deltas["bottom3_class_map50_95"] >= 0.0
        and deltas["worst_class_map50_95"] >= -0.005
    )
    tail_route = bool(
        deltas["macro_map50_95"] >= -0.002
        and deltas["bottom3_class_map50_95"] >= 0.010
        and deltas["worst_class_map50_95"] >= 0.010
    )
    return {
        "historical_deltas": deltas,
        "macro_route": macro_route,
        "lower_tail_route": tail_route,
        "decision": "AUTHORIZE_MATCHED_CONTROL_AND_ABLATION" if macro_route or tail_route else "STOP_AFTER_SINGLE_ARM",
        "claim_status": "historical-matched-protocol screening; not a causal matched-runtime comparison",
    }


def run_faruq_v3_af2_sfs_cue_direct(
    data_root: str | Path,
    grouped_summary: str | Path,
    pretrained_checkpoint: str | Path,
    historical_af2direct_summary: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != SEED:
        raise ValueError("Single-arm screen dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    root = Path(data_root).expanduser().resolve()
    grouped = Path(grouped_summary).expanduser().resolve()
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    historical = _historical_af2direct(historical_af2direct_summary)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    load_faruq_grouped_summary(grouped, root)
    reports = destination / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(root, reports / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Dataset audit gagal")

    static_path = destination / "static_audit.json"
    if static_path.is_file():
        static = _load_json(static_path, "Static audit")
        if static.get("pretrained_checkpoint_sha256") != pretrained_sha:
            raise RuntimeError("Pretrained berubah dari static audit")
    else:
        static = run_af2_sfs_cue_direct_static_audit(
            checkpoint, static_path, seed=seed, device=device
        )
    if static.get("decision") != "PASS" or static.get("training_authorized") is not True:
        raise RuntimeError("Static audit bukan PASS")

    cfg = _load_yaml(CONFIG)
    train_args = dict(cfg["train"])
    afab = AFABConfig.from_mapping(cfg["afab"])
    combo = AF2SFSCUEConfig.from_mapping(cfg["sfs_cue"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.af2_sfs_cue_direct.arm_contract.v1",
        "protocol": "faruq-v3-af2-sfs-cue-direct-seed42-v1",
        "arm": ARM,
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "native_initial_state_sha256": static["native_initial_state_sha256"],
        "config_sha256": static["config_sha256"],
        "historical_af2direct_checkpoint_sha256": historical.get("checkpoint_sha256"),
        "train": train_args,
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        if _load_json(contract_path, "Run contract") != contract:
            raise RuntimeError("Run directory memiliki kontrak berbeda")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    result_path = reports / f"{ARM}_seed{seed}_result.json"
    if result_path.is_file():
        result = _load_json(result_path, "Candidate result")
        if result.get("run_contract") != contract:
            raise RuntimeError("Result lama memiliki kontrak berbeda")
    else:
        best = run_dir / "weights/best.pt"
        last = run_dir / "weights/last.pt"
        trainer = make_af2_sfs_cue_direct_trainer(
            afab,
            combo,
            model_yaml=MODEL_YAML,
            pretrained_checkpoint=checkpoint,
            seed=seed,
            expected_native_fingerprint=static["native_initial_state_sha256"],
        )
        training_executed = False
        if not _run_complete(run_dir, int(train_args["epochs"])):
            from ultralytics import YOLO

            epoch, resumable = _checkpoint_state(last)
            with _exclusive_training_lock(destination, lock_name=f"{ARM}_seed{seed}.training.lock"):
                if last.is_file() and resumable and epoch is not None and epoch >= 0:
                    print(f"RESUME {ARM} seed {seed} dari epoch {epoch + 1}", flush=True)
                    model = YOLO(str(last))
                    args = {"resume": True, "device": device}
                else:
                    if last.is_file() and not resumable:
                        raise RuntimeError("last.pt ada tetapi tidak resumable")
                    print(f"START {ARM} seed {seed} direct dari official pretrained", flush=True)
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
        if not _run_complete(run_dir, int(train_args["epochs"])):
            raise RuntimeError("Training belum complete/early-stopped secara valid")
        if not best.is_file():
            raise FileNotFoundError(best)
        evaluation = evaluate(best, root, reports / f"{ARM}_seed{seed}_val.json", split="val", device=device)
        if evaluation["metrics"].get("classes_without_ground_truth"):
            raise RuntimeError("Validation kehilangan kelas")
        diagnostic = run_faruq_v3_diagnostics(
            best, root, reports / f"{ARM}_seed{seed}_diagnostic.json", split="val", device=device
        )
        result = {
            "format": "coffee_detector.af2_sfs_cue_direct.arm_result.v1",
            "protocol": "faruq-v3-af2-sfs-cue-direct-seed42-v1",
            "arm": ARM,
            "seed": seed,
            "metrics": {name: float(evaluation["metrics"][name]) for name in METRICS},
            "diagnostic": {
                "raw_top500_proposal_accessibility": float(diagnostic["raw_candidate_sensitivity"]["500"]["proposal_accessibility"]),
                "localization_conditioned_top1": float(diagnostic["global"]["localization_conditioned_class_accuracy"]),
                "correct_decision_recall": float(diagnostic["global"]["correct_class"] / max(int(diagnostic["global"]["targets"]), 1)),
            },
            "checkpoint": str(best),
            "checkpoint_sha256": _sha256(best),
            "completed_epochs": _epochs(run_dir / "results.csv"),
            "maximum_epochs": int(train_args["epochs"]),
            "training_executed_this_call": training_executed,
            "evaluation_split": "val",
            "test_images_accessed": False,
            "run_contract": contract,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    screen = _screen(historical, result)
    summary = {
        "format": "coffee_detector.af2_sfs_cue_direct.seed42_screen.v1",
        "protocol": "faruq-v3-af2-sfs-cue-direct-seed42-v1",
        "seed": seed,
        "historical_af2direct": historical,
        "candidate": result,
        "screen": screen,
        "training_executed": bool(result.get("training_executed_this_call")),
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    summary_path = destination / "af2_sfs_cue_direct_seed42_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"values": {"AF2DIRECT": historical["metrics"], ARM: result["metrics"]}, "screen": screen, "test": False}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-arm AF2-SFS-CUE direct/fresh seed-42 screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--historical-af2direct-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2_sfs_cue_direct(
        args.data_root,
        args.grouped_summary,
        args.pretrained_checkpoint,
        args.historical_af2direct_summary,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
