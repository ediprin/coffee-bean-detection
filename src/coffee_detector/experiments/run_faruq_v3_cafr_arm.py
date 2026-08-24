from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.cafr import VARIANTS, calibrate_patch_size, frozen_variant_config, make_cafr_trainer
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "C1": REPO_ROOT / "configs/cafr/C1_yolo26n_shared_luminance.yaml",
    "C2": REPO_ROOT / "configs/cafr/C2_yolo26n_radial_directional.yaml",
    "C3": REPO_ROOT / "configs/cafr/C3_yolo26n_soft_radial_directional.yaml",
    "C4": REPO_ROOT / "configs/cafr/C4_yolo26n_unsigned_orientation.yaml",
    "CAFR": REPO_ROOT / "configs/cafr/CAFR_yolo26n.yaml",
}


def _sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def run_faruq_v3_cafr_arm(
    arm: str,
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    labels_root: str | Path | None = None,
    seed: int = 42,
    device: str = "0",
    authorize_training: bool = False,
    latency_iterations: int = 50,
) -> dict:
    if arm not in VARIANTS:
        raise ValueError(f"arm harus salah satu {VARIANTS}")
    if seed != 42:
        raise ValueError("CAFR breadth screen dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi; tambahkan --authorize-training")

    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos split test")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
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
    epochs = int(payload["train"]["epochs"])
    imgsz = int(payload["train"]["imgsz"])

    calibration = None
    patch_size = int(payload["cafr"]["patch_size"])
    if arm == "CAFR":
        labels = Path(labels_root).expanduser().resolve() if labels_root else data_root / "train" / "labels"
        candidates = tuple(int(v) for v in payload.get("protocol", {}).get("patch_candidates", (16, 32, 64)))
        calibration = calibrate_patch_size(labels, imgsz=imgsz, candidates=candidates)
        patch_size = calibration.selected_patch_size
        calibration_path = reports / "cafr_patch_calibration.json"
        calibration_path.write_text(json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(
            f"CAFR PATCH CALIBRATION: median={calibration.median_equivalent_side_px:.2f}px "
            f"-> patch={patch_size}",
            flush=True,
        )

    cafr = frozen_variant_config(arm, patch_size=patch_size)
    expected = cafr.to_dict()
    # The final CAFR YAML contains a placeholder patch size because calibration is data-derived;
    # all other operator fields must match the frozen config exactly.
    yaml_cafr = dict(payload["cafr"])
    yaml_cafr["radial_boundaries"] = [float(v) for v in yaml_cafr.get("radial_boundaries", [])]
    for key, value in expected.items():
        if arm == "CAFR" and key == "patch_size":
            continue
        if yaml_cafr.get(key) != value:
            raise RuntimeError(f"Config drift pada {arm}.{key}: yaml={yaml_cafr.get(key)!r}, frozen={value!r}")

    run_dir = output_root / arm / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "format": "coffee_detector.cafr.run_contract.v1",
        "arm": arm,
        "seed": seed,
        "config_sha256": _sha256(config_path),
        "d0_checkpoint_sha256": _sha256(checkpoint),
        "epochs": epochs,
        "cafr": cafr.to_dict(),
        "patch_calibration": None if calibration is None else calibration.to_dict(),
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

        trainer = make_cafr_trainer(cafr, d0_checkpoint=checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(output_root, lock_name=f"{arm}_seed{seed}.training.lock"):
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
        "format": "coffee_detector.cafr.arm_result.v1",
        "arm": arm,
        "seed": seed,
        "metrics": report["metrics"],
        "latency": latency,
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "initial_d0_checkpoint": str(checkpoint),
        "initial_d0_checkpoint_sha256": _sha256(checkpoint),
        "config": str(config_path),
        "cafr": cafr.to_dict(),
        "patch_calibration": None if calibration is None else calibration.to_dict(),
        "training_executed": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path = reports / f"{arm}_seed{seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one CAFR causal-ablation arm")
    parser.add_argument("--arm", choices=VARIANTS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--labels-root", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--latency-iterations", type=int, default=50)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_cafr_arm(
        args.arm,
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.output_root,
        labels_root=args.labels_root or None,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
        latency_iterations=args.latency_iterations,
    )


if __name__ == "__main__":
    main()
