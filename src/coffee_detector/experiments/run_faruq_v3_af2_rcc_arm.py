from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_rcc import AF2RCCConfig, make_af2_rcc_trainer
from coffee_detector.afab import AFABConfig
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _exclusive_training_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_rcc/AF2RCC1_yolo26n.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_seed(path: Path) -> int:
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(args, dict) or "seed" not in args:
        raise RuntimeError(f"Checkpoint AF2 tidak merekam seed: {path}")
    return int(args["seed"])


def _complete(run_dir: Path, epochs: int, source_sha: str) -> bool:
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights/best.pt"
    if not marker.is_file() or not best.is_file():
        return False
    payload = _read(marker, "Training marker")
    return (
        payload.get("trainer_returned") is True
        and int(payload.get("epochs_requested", -1)) == epochs
        and payload.get("initial_checkpoint_sha256") == source_sha
    )


def run_faruq_v3_af2_rcc_arm(
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_checkpoint: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise RuntimeError("Protocol screening AF2-RCC dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training AF2-RCC belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    audit_path = Path(static_audit).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    if not (data_root / "data.yaml").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Dataset development atau checkpoint AF2 tidak lengkap")
    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError("Checkpoint AF2 harus seed 42")
    grouped = _read(Path(grouped_summary).expanduser().resolve(), "Grouped summary")
    if grouped.get("test_images_accessed") not in {False, None}:
        raise RuntimeError("Grouped summary tidak mempertahankan test lock")
    source_sha = _sha256(checkpoint)
    audit = _read(audit_path, "Static audit")
    if (
        audit.get("format") != "coffee_detector.af2_rcc.static_audit.v1"
        or audit.get("decision") != "PASS"
        or audit.get("training_authorized") is not True
        or audit.get("checkpoint_sha256") != source_sha
        or audit.get("test_access_authorized") is not False
    ):
        raise RuntimeError("Static audit AF2-RCC tidak mengotorisasi checkpoint ini")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "AF2RCC1":
        raise RuntimeError("Config AF2RCC1 tidak konsisten")
    afab = AFABConfig.from_mapping(payload["afab"])
    rcc = AF2RCCConfig.from_mapping(payload["af2_rcc"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / "AF2RCC1" / f"AF2RCC1_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    if not _complete(run_dir, epochs, source_sha):
        from ultralytics import YOLO

        trainer = make_af2_rcc_trainer(afab, rcc, initial_checkpoint=checkpoint)
        with _exclusive_training_lock(
            output_root, lock_name=f"AF2RCC1_seed{seed}.training.lock"
        ):
            if last.is_file():
                print("RESUME AF2RCC1 seed 42 dari last.pt", flush=True)
                model = YOLO(str(last))
                args = {"resume": True, "device": device}
            else:
                print("START AF2RCC1 seed 42 dari AF2 asli", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root / "AF2RCC1"),
                    name=f"AF2RCC1_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=False,
                    verbose=False,
                    device=device,
                )
            model.train(trainer=trainer, **args)
        training_executed = True
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "training_complete.json").write_text(
            json.dumps(
                {
                    "trainer_returned": True,
                    "epochs_requested": epochs,
                    "seed": seed,
                    "arm": "AF2RCC1",
                    "initial_checkpoint_sha256": source_sha,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not _complete(run_dir, epochs, source_sha):
        raise RuntimeError(f"Run AF2RCC1 belum lengkap: {run_dir}")

    reports = output_root / "val_reports"
    baseline = evaluate(
        checkpoint,
        data_root,
        reports / "AF2_seed42_reference_val.json",
        split="val",
        device=device,
    )
    candidate = evaluate(
        best,
        data_root,
        reports / "AF2RCC1_seed42_val.json",
        split="val",
        device=device,
    )
    for label, report in (("AF2", baseline), ("AF2RCC1", candidate)):
        if report["metrics"].get("classes_without_ground_truth"):
            raise RuntimeError(f"Validation {label} kehilangan kelas")
    result = {
        "format": "coffee_detector.af2_rcc.arm_result.v1",
        "arm": "AF2RCC1",
        "seed": seed,
        "baseline_metrics": baseline["metrics"],
        "metrics": candidate["metrics"],
        "checkpoint": str(best),
        "initial_af2_checkpoint": str(checkpoint),
        "initial_af2_checkpoint_sha256": source_sha,
        "static_audit": str(audit_path),
        "config": str(CONFIG),
        "trainable_parameters": 189,
        "training_executed": training_executed,
        "test_images_accessed": False,
    }
    result_path = reports / "AF2RCC1_seed42_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AF2RCC1 seed-42 screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2_rcc_arm(
        args.data_root,
        args.grouped_summary,
        args.af2_checkpoint,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
