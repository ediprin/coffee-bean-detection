"""Conditional retraining of one selectivity candidate chosen by diagnosis.

The diagnostic JSON may authorize one validation-tuned candidate for a fresh
50-epoch three-seed screen from the same seed-matched D0 checkpoints. This
runner does not open test and does not itself establish a thesis upgrade claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_ffa import AF2FFAConfig, make_af2_ffa_trainer
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
BASE_CONFIG = REPO_ROOT / "configs/af2_ffa_from_start/AF2FFAB2FS_yolo26n_from_start.yaml"
ALLOWED_SEEDS = (42, 123, 2026)
ARM = "AF2FFASR1"


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
        raise RuntimeError(f"Checkpoint D0 tidak merekam seed: {path}")
    return int(train_args["seed"])


def _candidate_config(diagnostic: dict) -> tuple[dict, dict]:
    if (
        diagnostic.get("format") != "coffee_detector.af2_ffa.selectivity_analysis.v1"
        or diagnostic.get("decision") != "DIAGNOSTIC_CANDIDATE_FOUND"
        or diagnostic.get("training_authorized") is not True
        or diagnostic.get("test_opened") is not False
    ):
        raise RuntimeError("Diagnosis belum mengotorisasi selective retraining")
    candidate = dict(diagnostic.get("selected_candidate") or {})
    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")) or {}
    adapter = dict(payload["af2_ffa"])
    adapter.update(
        adapter_strength_scale=float(candidate.get("strength", 1.0)),
        active_levels=[name in set(candidate.get("active_levels", ["P3", "P4", "P5"])) for name in ("P3", "P4", "P5")],
        fusion_mode=candidate.get("fusion_mode", "replace"),
        residual_mix=float(candidate.get("residual_mix", 1.0)),
        ambiguity_gate=candidate.get("ambiguity_gate", "none"),
        ambiguity_margin=float(candidate.get("ambiguity_margin", 0.15)),
        ambiguity_temperature=float(candidate.get("ambiguity_temperature", 0.05)),
    )
    AF2FFAConfig.from_mapping(adapter)  # fail closed before training
    payload["af2_ffa"] = adapter
    return payload, candidate


def run_selective_arm(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    static_audit: str | Path,
    diagnostic_path: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed harus salah satu {ALLOWED_SEEDS}")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    d0 = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    if not d0.is_file() or _checkpoint_seed(d0) != seed:
        raise RuntimeError(f"D0 tidak valid untuk seed {seed}: {d0}")
    load_faruq_grouped_summary(grouped_summary, data_root)

    static = _read(static_audit, "Static audit")
    if (
        static.get("format") != "coffee_detector.af2_ffa.from_start_static_audit.v1"
        or static.get("decision") != "PASS"
        or static.get("training_authorized") is not True
        or static.get("test_access_authorized") is not False
        or static.get("d0_checkpoint_sha256") != _sha256(d0)
    ):
        raise RuntimeError("Static audit tidak mengotorisasi D0 ini")

    diagnostic_file = Path(diagnostic_path).expanduser().resolve()
    diagnostic = _read(diagnostic_file, "Selectivity diagnosis")
    payload, selected = _candidate_config(diagnostic)
    epochs = int(payload["train"]["epochs"])
    if epochs != 50:
        raise RuntimeError("Selective retraining dibekukan pada 50 epoch")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = reports / "dataset_audit.json"
    if not dataset_audit.is_file():
        report = audit_dataset(data_root, dataset_audit, near_threshold=-1)
        if not report["safe_for_training"]:
            raise RuntimeError("Audit dataset gagal")

    run_dir = output_root / ARM / f"{ARM}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    contract = {
        "format": "coffee_detector.af2_ffa.selective_contract.v1",
        "arm": ARM,
        "seed": seed,
        "d0_checkpoint_sha256": _sha256(d0),
        "base_config_sha256": _sha256(BASE_CONFIG),
        "diagnostic_sha256": _sha256(diagnostic_file),
        "selected_candidate": selected,
        "resolved_af2_ffa": payload["af2_ffa"],
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

        afab = AFABConfig.from_mapping(payload["afab"])
        adapter = AF2FFAConfig.from_mapping(payload["af2_ffa"])
        trainer = make_af2_ffa_trainer(afab, adapter, initial_checkpoint=d0)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(output_root, lock_name=f"{ARM}_seed{seed}.training.lock"):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {ARM} seed {seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                print(f"START {ARM} seed {seed} dari D0 seed-matched", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root / ARM),
                    name=f"{ARM}_seed{seed}",
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
    val_path = reports / f"{ARM}_seed{seed}_val.json"
    val = evaluate(best, data_root, val_path, split="val", device=device)
    if val["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.af2_ffa.selective_arm_result.v1",
        "arm": ARM,
        "seed": seed,
        "metrics": val["metrics"],
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "initial_d0_checkpoint_sha256": _sha256(d0),
        "diagnostic_sha256": _sha256(diagnostic_file),
        "selected_candidate": selected,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path = reports / f"{ARM}_seed{seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one AF2+FFAB2 selective candidate arm")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--diagnostic", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_selective_arm(
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.static_audit,
        args.diagnostic,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
