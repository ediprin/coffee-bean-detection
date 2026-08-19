from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)
from coffee_detector.wav1_factorization import (
    TRAIN_ARMS,
    WAV1FactorizationConfig,
    make_factorization_trainer,
)
from coffee_detector.wav1_factorization.audit import sha256


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    arm: REPO_ROOT / f"configs/wav1_factorization/{arm}_yolo26n.yaml"
    for arm in TRAIN_ARMS
}


def _read(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _latency(checkpoint: Path, device: str, iterations: int = 50) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint)).model.eval()
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    model = model.to(torch_device)
    sample = torch.rand(1, 3, 640, 640, device=torch_device)
    warmup = min(10, iterations)
    values = []
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


def run_faruq_v3_wav1_factorization_arm(
    arm: str,
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    authorize_training: bool = False,
    latency_iterations: int = 50,
) -> dict:
    if arm not in TRAIN_ARMS:
        raise ValueError(f"arm harus salah satu {TRAIN_ARMS}")
    if seed != 42:
        raise ValueError("Mechanism screening dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")

    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    load_faruq_grouped_summary(grouped_summary, data_root)

    audit = _read(static_audit, "Static audit")
    if audit.get("decision") != "PASS" or not audit.get("training_authorized"):
        raise RuntimeError("Static audit WAV1 factorization belum PASS")
    if audit.get("d0_checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("Checkpoint D0 berbeda dari static audit")
    if audit.get("wav1_ref_bitwise_equal_to_confirmed_operator") is not True:
        raise RuntimeError("WAV1 reference tidak identik dengan operator confirmed")
    if audit.get("test_access_authorized") is not False:
        raise RuntimeError("Static audit tidak mempertahankan test lock")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = reports / "dataset_audit.json"
    if not dataset_audit.is_file():
        payload = audit_dataset(data_root, dataset_audit, near_threshold=-1)
        if not payload["safe_for_training"]:
            raise RuntimeError("Audit dataset gagal")

    config_path = CONFIGS[arm]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if payload.get("code") != arm:
        raise RuntimeError(f"Config {arm} tidak konsisten")
    factorization = WAV1FactorizationConfig.from_mapping(payload["factorization"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / arm / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_contract = {
        "format": "coffee_detector.wav1_factorization.run_contract.v1",
        "arm": arm,
        "seed": seed,
        "config_sha256": sha256(config_path),
        "d0_checkpoint_sha256": sha256(checkpoint),
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

        trainer = make_factorization_trainer(factorization, d0_checkpoint=checkpoint)
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
    (reports / f"{arm}_seed{seed}_latency.json").write_text(
        json.dumps(latency, indent=2) + "\n", encoding="utf-8"
    )

    result = {
        "format": "coffee_detector.wav1_factorization.arm_result.v1",
        "arm": arm,
        "seed": seed,
        "metrics": report["metrics"],
        "latency": latency,
        "checkpoint": str(best),
        "checkpoint_sha256": sha256(best),
        "initial_d0_checkpoint": str(checkpoint),
        "initial_d0_checkpoint_sha256": sha256(checkpoint),
        "config": str(config_path),
        "static_audit": str(Path(static_audit).expanduser().resolve()),
        "training_executed": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path = reports / f"{arm}_seed{seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one WAV1 mechanism-factorization arm")
    parser.add_argument("--arm", choices=TRAIN_ARMS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--latency-iterations", type=int, default=50)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_wav1_factorization_arm(
        args.arm,
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
        latency_iterations=args.latency_iterations,
    )


if __name__ == "__main__":
    main()
