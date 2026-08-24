from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_iso import ARMS, frozen_arm_config, make_af2_iso_trainer
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_ARMS = ("AF2_RADIAL", "AF2_ORIENT")
CONFIGS = {
    "AF2_RADIAL": REPO_ROOT / "configs/af2_iso/AF2_RADIAL_yolo26n.yaml",
    "AF2_ORIENT": REPO_ROOT / "configs/af2_iso/AF2_ORIENT_yolo26n.yaml",
}
ALLOWED_SEEDS = {
    "AF2_RADIAL": (42,),
    "AF2_ORIENT": (42, 123, 2026),
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_seed(path: Path) -> int:
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(train_args, dict) or "seed" not in train_args:
        raise RuntimeError(f"Checkpoint D0 tidak merekam seed: {path}")
    return int(train_args["seed"])


def _latency(checkpoint: Path, device: str, iterations: int = 50) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint)).model.eval()
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    model = model.to(torch_device)
    sample = torch.rand(1, 3, 640, 640, device=torch_device)
    warmup = min(10, iterations)
    values: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        for _ in range(iterations):
            started = time.perf_counter()
            model(sample)
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            values.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(values)
    return {
        "device": str(torch_device),
        "batch": 1,
        "image_size": 640,
        "warmup": warmup,
        "iterations": iterations,
        "median_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(int(0.95 * (len(ordered) - 1)), len(ordered) - 1)],
    }


def run_faruq_v3_af2_iso_arm(
    arm: str,
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    authorize_training: bool = False,
    latency_iterations: int = 50,
) -> dict:
    if arm not in TRAIN_ARMS:
        raise ValueError(f"arm harus salah satu {TRAIN_ARMS}")
    if seed not in ALLOWED_SEEDS[arm]:
        raise ValueError(f"{arm} hanya mengizinkan seed {ALLOWED_SEEDS[arm]}")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi; tambahkan --authorize-training")

    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos split test")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError(f"Checkpoint D0 tidak cocok dengan seed {seed}")
    load_faruq_grouped_summary(grouped_summary, data_root)

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = reports / "dataset_audit.json"
    if not dataset_audit.is_file():
        audit = audit_dataset(data_root, dataset_audit, near_threshold=-1)
        if not audit["safe_for_training"]:
            raise RuntimeError("Audit dataset gagal")

    config_path = CONFIGS[arm]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if payload.get("code") != arm:
        raise RuntimeError(f"Config {arm} tidak konsisten")
    frozen = frozen_arm_config(arm)
    yaml_operator = dict(payload["af2_iso"])
    yaml_operator["radial_boundaries"] = [
        float(v) for v in yaml_operator.get("radial_boundaries", [])
    ]
    if yaml_operator != frozen.to_dict():
        raise RuntimeError(
            f"Config drift {arm}: yaml={yaml_operator!r}, frozen={frozen.to_dict()!r}"
        )

    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / arm / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_contract = {
        "format": "coffee_detector.af2_iso.run_contract.v1",
        "arm": arm,
        "seed": seed,
        "config_sha256": _sha256(config_path),
        "d0_checkpoint_sha256": _sha256(checkpoint),
        "epochs": epochs,
        "operator": frozen.to_dict(),
        "comparison_parent": "legacy AF2",
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != run_contract:
            raise RuntimeError(f"Run directory memiliki kontrak berbeda: {run_dir}")
    contract_path.write_text(json.dumps(run_contract, indent=2) + "\n", encoding="utf-8")

    training_executed = False
    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_af2_iso_trainer(frozen, d0_checkpoint=checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root, lock_name=f"{arm}_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {arm} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                print(f"START {arm} seed {seed} dari D0 seed {seed}", flush=True)
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
        training_executed = True

    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run belum lengkap: {run_dir}")

    report_path = reports / f"{arm}_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")

    latency = _latency(best, device, latency_iterations)
    latency_path = reports / f"{arm}_seed{seed}_latency.json"
    latency_path.write_text(json.dumps(latency, indent=2) + "\n", encoding="utf-8")

    result = {
        "format": "coffee_detector.af2_iso.arm_result.v1",
        "arm": arm,
        "seed": seed,
        "metrics": report["metrics"],
        "latency": latency,
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "initial_d0_checkpoint": str(checkpoint),
        "initial_d0_checkpoint_sha256": _sha256(checkpoint),
        "config": str(config_path),
        "operator": frozen.to_dict(),
        "training_executed": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path = reports / f"{arm}_seed{seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one isolated AF2 radial/orientation arm"
    )
    parser.add_argument("--arm", choices=TRAIN_ARMS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--latency-iterations", type=int, default=50)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2_iso_arm(
        args.arm,
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
        latency_iterations=args.latency_iterations,
    )


if __name__ == "__main__":
    main()
