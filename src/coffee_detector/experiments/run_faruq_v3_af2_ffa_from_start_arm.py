"""Fair from-start AF2 / AF2+FFAB2 / AF2+FFAB2-DCT arm runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_ffa import AF2FFAConfig, make_af2_ffa_trainer
from coffee_detector.afab import AFABConfig, make_afab_trainer
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = ("AF2FS", "AF2FFAB2FS", "AF2FFADCTFS")
ALLOWED_SEEDS = (42, 123, 2026)
CONFIGS = {
    "AF2FS": REPO_ROOT / "configs/afab/AF2_yolo26n_chaotic_amplitude.yaml",
    "AF2FFAB2FS": REPO_ROOT / "configs/af2_ffa_from_start/AF2FFAB2FS_yolo26n_from_start.yaml",
    "AF2FFADCTFS": REPO_ROOT / "configs/af2_ffa_from_start/AF2FFADCTFS_yolo26n_from_start.yaml",
}
EXPECTED_CODES = {"AF2FS": "AF2", "AF2FFAB2FS": "AF2FFAB2FS", "AF2FFADCTFS": "AF2FFADCTFS"}


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


def _latency_and_memory(checkpoint: Path, device: str, iterations: int = 60) -> dict:
    from ultralytics import YOLO

    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    model = YOLO(str(checkpoint)).model.eval().to(torch_device)
    sample = torch.rand(1, 3, 640, 640, device=torch_device)
    warmup = min(10, iterations)
    values: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
            torch.cuda.reset_peak_memory_stats(torch_device)
        for _ in range(iterations):
            started = time.perf_counter()
            model(sample)
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            values.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(values)
    peak_mb = (
        float(torch.cuda.max_memory_allocated(torch_device)) / (1024.0 * 1024.0)
        if torch_device.type == "cuda"
        else None
    )
    return {
        "device": str(torch_device),
        "batch": 1,
        "image_size": 640,
        "warmup": warmup,
        "iterations": iterations,
        "median_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(int(0.95 * (len(ordered) - 1)), len(ordered) - 1)],
        "peak_memory_mb": peak_mb,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def _dct_authorized(path: str | Path | None) -> dict:
    if path is None:
        raise RuntimeError("AF2FFADCTFS memerlukan keputusan Stage-1 from-start")
    payload = _read(path, "Keputusan Stage-1")
    if (
        payload.get("format") != "coffee_detector.af2_ffa.from_start_decision.v1"
        or payload.get("decision") != "PASS"
        or payload.get("next") != "AUTHORIZE_DCT_EFFICIENCY_STAGE"
        or payload.get("test_opened") is not False
    ):
        raise RuntimeError("Stage-1 tidak mengotorisasi eksperimen DCT")
    return payload


def run_faruq_v3_af2_ffa_from_start_arm(
    arm: str,
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str = "0",
    stage1_decision: str | Path | None = None,
    authorize_training: bool = False,
    latency_iterations: int = 60,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"arm harus salah satu {ARMS}")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed harus salah satu {ALLOWED_SEEDS}")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    if arm == "AF2FFADCTFS":
        _dct_authorized(stage1_decision)

    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError(f"Checkpoint D0 bukan seed {seed}")
    load_faruq_grouped_summary(grouped_summary, data_root)

    audit = _read(static_audit, "Static audit from-start")
    if (
        audit.get("format") != "coffee_detector.af2_ffa.from_start_static_audit.v1"
        or audit.get("decision") != "PASS"
        or audit.get("training_authorized") is not True
        or audit.get("test_access_authorized") is not False
        or audit.get("d0_checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise RuntimeError("Static audit tidak mengotorisasi D0 ini")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = reports / "dataset_audit.json"
    if not dataset_audit.is_file():
        report = audit_dataset(data_root, dataset_audit, near_threshold=-1)
        if not report["safe_for_training"]:
            raise RuntimeError("Audit dataset gagal")

    config_path = CONFIGS[arm]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if payload.get("code") != EXPECTED_CODES[arm]:
        raise RuntimeError(f"Kode config tidak konsisten untuk {arm}")
    epochs = int(payload["train"]["epochs"])
    if epochs != 50:
        raise RuntimeError("From-start experiment dibekukan pada 50 epoch")

    run_dir = output_root / arm / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    contract = {
        "format": "coffee_detector.af2_ffa.from_start_contract.v1",
        "arm": arm,
        "seed": seed,
        "d0_checkpoint_sha256": _sha256(checkpoint),
        "config_sha256": _sha256(config_path),
        "epochs": epochs,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _read(contract_path, "Run contract") != contract:
        raise RuntimeError(f"Run directory memiliki kontrak berbeda: {run_dir}")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    training_executed = False
    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        afab = AFABConfig.from_mapping(payload["afab"])
        if arm == "AF2FS":
            trainer = make_afab_trainer(afab, d0_checkpoint=checkpoint)
        else:
            adapter = AF2FFAConfig.from_mapping(payload["af2_ffa"])
            trainer = make_af2_ffa_trainer(afab, adapter, initial_checkpoint=checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(output_root, lock_name=f"{arm}_seed{seed}.training.lock"):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {arm} seed {seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                print(f"START {arm} seed {seed} dari D0 seed-matched", flush=True)
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

    val_path = reports / f"{arm}_seed{seed}_val.json"
    val = evaluate(best, data_root, val_path, split="val", device=device)
    if val["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    efficiency = _latency_and_memory(best, device, latency_iterations)
    result = {
        "format": "coffee_detector.af2_ffa.from_start_arm_result.v1",
        "arm": arm,
        "seed": seed,
        "metrics": val["metrics"],
        "efficiency": efficiency,
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "initial_d0_checkpoint": str(checkpoint),
        "initial_d0_checkpoint_sha256": _sha256(checkpoint),
        "config": str(config_path),
        "static_audit": str(Path(static_audit).expanduser().resolve()),
        "stage1_decision": str(Path(stage1_decision).expanduser().resolve()) if stage1_decision else None,
        "training_executed": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path = reports / "val_reports" / f"{arm}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fair AF2-FFAB2 from-start arm")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--stage1-decision")
    parser.add_argument("--latency-iterations", type=int, default=60)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2_ffa_from_start_arm(
        args.arm,
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        stage1_decision=args.stage1_decision,
        authorize_training=args.authorize_training,
        latency_iterations=args.latency_iterations,
    )


if __name__ == "__main__":
    main()
