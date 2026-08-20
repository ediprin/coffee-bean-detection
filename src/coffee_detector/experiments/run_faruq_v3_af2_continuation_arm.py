"""Run one validation-only AF2 continuation arm from a seed-matched AF2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.afab import AFABConfig, make_afab_trainer
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_continuation/AF2CT30_yolo26n.yaml"
ALLOWED_SEEDS = (123, 2026)
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _read(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_seed(path: Path) -> int:
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(train_args, dict) or "seed" not in train_args:
        raise RuntimeError(f"Checkpoint AF2 tidak merekam seed: {path}")
    return int(train_args["seed"])


def _baseline_metrics(confirmation: dict, seed: int) -> dict[str, float]:
    if (
        confirmation.get("protocol")
        != "faruq-v3-af2-igem-paired-validation-confirmation-v1"
        or confirmation.get("seeds") != [42, 123, 2026]
        or confirmation.get("test_images_accessed") is not False
        or confirmation.get("test_opened") is not False
        or confirmation.get("decisions", {}).get("AF2", {}).get("decision") != "PASS"
    ):
        raise RuntimeError("Konfirmasi AF2 tiga-seed tidak kompatibel")
    source = confirmation["per_seed"][str(seed)]["AF2"]
    return {name: float(source[name]) for name in METRICS}


def run_faruq_v3_af2_continuation_arm(
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_confirmation: str | Path,
    af2_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Arm baru hanya untuk seed {ALLOWED_SEEDS}; seed 42 memakai evidence lama")
    if not authorize_training:
        raise RuntimeError("Continuation AF2 belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    if not (data_root / "data.yaml").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Dataset development atau checkpoint AF2 tidak lengkap")
    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError(f"Checkpoint AF2 bukan seed {seed}: {checkpoint}")
    confirmation = _read(af2_confirmation, "Konfirmasi AF2 tiga-seed")
    baseline = _baseline_metrics(confirmation, seed)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "AF2CT30" or payload.get("mechanism") != "af2_continuation_only":
        raise RuntimeError("Config continuation AF2 tidak konsisten")
    afab = AFABConfig.from_mapping(payload["afab"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / f"AF2CT30_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False

    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_afab_trainer(afab, d0_checkpoint=checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root, lock_name=f"AF2CT30_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME AF2CT30 seed {seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model = YOLO(str(last))
                args = {"resume": True, "device": device}
            else:
                print(f"START AF2CT30 seed {seed} dari AF2 seed-matched", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root),
                    name=f"AF2CT30_seed{seed}",
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
        raise RuntimeError(f"Run continuation belum lengkap: {run_dir}")
    report_path = output_root / "val_reports" / f"AF2CT30_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    metrics = report["metrics"]
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.af2_continuation.arm_result.v1",
        "seed": seed,
        "arm": "AF2CT30",
        "initial_af2_checkpoint": str(checkpoint),
        "initial_af2_checkpoint_sha256": _sha256(checkpoint),
        "baseline_af2_metrics": baseline,
        "metrics": metrics,
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "epochs_requested": epochs,
        "training_executed_this_call": training_executed,
        "test_images_accessed": False,
    }
    result_path = output_root / "val_reports" / f"AF2CT30_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one paired AF2 continuation arm")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-confirmation", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2_continuation_arm(
        args.data_root,
        args.grouped_summary,
        args.af2_confirmation,
        args.af2_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
