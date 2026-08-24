"""Train frozen-parent AF2 + FFAB2 residual continuation arms.

The source is a completed AF2FS checkpoint, never D0. All AF2 parameters and
BatchNorm statistics remain frozen. Only FFAB adapter parameters can update.
Test is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_ffa.model import AF2FFAConfig
from coffee_detector.af2_ffa.parent_preserving import make_af2_ffa_parent_trainer
from coffee_detector.afab import AFABConfig
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "AF2FFAPR0": REPO_ROOT / "configs/af2_ffa_parent_preserving/AF2FFAPR0_yolo26n_zero_parent_residual.yaml",
    "AF2FFAPR1": REPO_ROOT / "configs/af2_ffa_parent_preserving/AF2FFAPR1_yolo26n_spectral_parent_residual.yaml",
}
SEEDS = (42, 123, 2026)


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


def _checkpoint_seed(path: Path) -> int:
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(train_args, dict) or "seed" not in train_args:
        raise RuntimeError(f"Checkpoint parent tidak merekam seed: {path}")
    return int(train_args["seed"])


def _load_config(arm: str) -> tuple[Path, dict]:
    if arm not in CONFIGS:
        raise ValueError(f"arm harus salah satu {tuple(CONFIGS)}")
    path = CONFIGS[arm]
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("code") != arm:
        raise RuntimeError(f"Config code mismatch: {path}")
    AF2FFAConfig.from_mapping(payload["af2_ffa"])
    if payload["af2_ffa"].get("fusion_mode") != "parent_residual":
        raise RuntimeError("Parent arm wajib memakai parent_residual")
    if int(payload["train"]["epochs"]) != 30:
        raise RuntimeError("Parent-preserving continuation dibekukan pada 30 epoch")
    return path, payload


def _validate_parent_result(result: dict, checkpoint: Path, seed: int) -> None:
    if result.get("format") != "coffee_detector.af2_ffa.from_start_arm_result.v1":
        raise RuntimeError("Parent result harus berasal dari AF2FS matched from-start")
    if result.get("arm") != "AF2FS" or int(result.get("seed", -1)) != seed:
        raise RuntimeError("Parent result arm/seed tidak cocok")
    if result.get("evaluation_split") != "val" or result.get("test_images_accessed") is not False:
        raise RuntimeError("Parent result harus validation-only dengan test terkunci")
    expected = str(result.get("checkpoint_sha256", ""))
    if _sha256(checkpoint) != expected:
        raise RuntimeError("Parent checkpoint SHA tidak cocok dengan AF2FS result")
    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError("Seed parent checkpoint tidak cocok")


def run_parent_arm(
    arm: str,
    data_root: str | Path,
    grouped_summary: str | Path,
    parent_result: str | Path,
    parent_checkpoint: str | Path,
    static_audit: str | Path,
    selectivity_analysis: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed not in SEEDS:
        raise ValueError(f"seed harus salah satu {SEEDS}")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    parent_checkpoint = Path(parent_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    if not parent_checkpoint.is_file():
        raise FileNotFoundError(parent_checkpoint)
    load_faruq_grouped_summary(grouped_summary, data_root)

    parent_result_path = Path(parent_result).expanduser().resolve()
    parent_payload = _read(parent_result_path, "AF2FS parent result")
    _validate_parent_result(parent_payload, parent_checkpoint, seed)

    config_path, payload = _load_config(arm)
    audit_path = Path(static_audit).expanduser().resolve()
    audit = _read(audit_path, "Parent-preserving static audit")
    if (
        audit.get("format") != "coffee_detector.af2_ffa.parent_preserving_static_audit.v1"
        or audit.get("decision") != "PASS"
        or audit.get("training_authorized") is not True
        or audit.get("test_access_authorized") is not False
        or audit.get("parent_checkpoint_sha256") != _sha256(parent_checkpoint)
    ):
        raise RuntimeError("Static audit tidak mengotorisasi parent checkpoint ini")

    selectivity_path = Path(selectivity_analysis).expanduser().resolve()
    selectivity = _read(selectivity_path, "Selectivity analysis")
    if (
        selectivity.get("format") != "coffee_detector.af2_ffa.selectivity_analysis.v1"
        or selectivity.get("decision") != "NO_RUNTIME_CANDIDATE_PASSES_GATE"
        or selectivity.get("training_authorized") is not False
        or selectivity.get("test_opened") is not False
    ):
        raise RuntimeError("Parent-preserving follow-up memerlukan diagnosis runtime yang sudah gagal")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = reports / "dataset_audit.json"
    if not dataset_audit.is_file():
        report = audit_dataset(data_root, dataset_audit, near_threshold=-1)
        if not report["safe_for_training"]:
            raise RuntimeError("Audit dataset gagal")

    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / arm / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    contract = {
        "format": "coffee_detector.af2_ffa.parent_preserving_contract.v1",
        "arm": arm,
        "seed": seed,
        "parent_arm": "AF2FS",
        "parent_checkpoint_sha256": _sha256(parent_checkpoint),
        "parent_result_sha256": _sha256(parent_result_path),
        "config_sha256": _sha256(config_path),
        "static_audit_sha256": _sha256(audit_path),
        "selectivity_analysis_sha256": _sha256(selectivity_path),
        "conditioning": payload["af2_ffa"]["conditioning"],
        "fusion_mode": payload["af2_ffa"]["fusion_mode"],
        "parent_frozen": True,
        "trainable_scope": "ffab_adapters_only",
        "epochs": epochs,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _read(contract_path, "Run contract") != contract:
        raise RuntimeError(f"Run directory memiliki kontrak berbeda: {run_dir}")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_af2_ffa_parent_trainer(
            AFABConfig.from_mapping(payload["afab"]),
            AF2FFAConfig.from_mapping(payload["af2_ffa"]),
            parent_checkpoint=parent_checkpoint,
        )
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(output_root, lock_name=f"{arm}_seed{seed}.training.lock"):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {arm} seed {seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                print(f"START {arm} seed {seed} dari completed AF2FS parent", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root / arm),
                    name=f"{arm}_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=False,
                    verbose=False,
                    device=device,
                )
            model.train(trainer=trainer, **args)

    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run belum lengkap: {run_dir}")
    if not best.is_file():
        raise FileNotFoundError(best)

    val_path = reports / f"{arm}_seed{seed}_val.json"
    val = evaluate(best, data_root, val_path, split="val", device=device)
    if val["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.af2_ffa.parent_preserving_arm_result.v1",
        "arm": arm,
        "seed": seed,
        "parent_arm": "AF2FS",
        "parent_metrics": {
            key: parent_payload["metrics"][key]
            for key in (
                "macro_map50_95",
                "bottom3_class_map50_95",
                "worst_class_map50_95",
            )
        },
        "metrics": val["metrics"],
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": _sha256(parent_checkpoint),
        "conditioning": payload["af2_ffa"]["conditioning"],
        "parent_frozen": True,
        "trainable_scope": "ffab_adapters_only",
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path = reports / f"{arm}_seed{seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen-parent AF2 + FFAB2 residual arm")
    parser.add_argument("--arm", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--parent-result", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--selectivity-analysis", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_parent_arm(
        args.arm,
        args.data_root,
        args.grouped_summary,
        args.parent_result,
        args.parent_checkpoint,
        args.static_audit,
        args.selectivity_analysis,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
