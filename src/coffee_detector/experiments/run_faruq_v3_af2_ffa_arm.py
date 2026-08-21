from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_ffa import AF2FFAConfig, make_af2_ffa_trainer
from coffee_detector.afab import AFABConfig
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _exclusive_training_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = ("AF2FFA0", "AF2FFA1")
ALLOWED_SEEDS = (42, 123, 2026)
CONFIGS = {
    "AF2FFA0": REPO_ROOT / "configs/af2_ffa/AF2FFA0_yolo26n_zero_control.yaml",
    "AF2FFA1": REPO_ROOT / "configs/af2_ffa/AF2FFA1_yolo26n_spectral_adapter.yaml",
}


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
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(train_args, dict) or "seed" not in train_args:
        raise RuntimeError(f"Checkpoint AF2 tidak merekam seed: {path}")
    return int(train_args["seed"])


def _complete(run_dir: Path, epochs: int) -> bool:
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights/best.pt"
    if not marker.is_file() or not best.is_file():
        return False
    payload = _read(marker, "Training marker")
    return bool(payload.get("trainer_returned")) and int(
        payload.get("epochs_requested", -1)
    ) == epochs


def run_faruq_v3_af2_ffa_arm(
    arm: str,
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
    if arm not in ARMS:
        raise ValueError(f"Arm harus salah satu {ARMS}")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Seed harus salah satu {ALLOWED_SEEDS}")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    audit_path = Path(static_audit).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    if not (data_root / "data.yaml").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Dataset development atau checkpoint AF2 tidak lengkap")
    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError(f"Checkpoint AF2 bukan seed {seed}: {checkpoint}")
    _read(Path(grouped_summary).expanduser().resolve(), "Grouped summary")
    audit = _read(audit_path, "Static audit")
    if audit.get("decision") != "PASS" or not audit.get("training_authorized"):
        raise RuntimeError("Static audit AF2-FFA belum PASS")
    if audit.get("checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError("Checkpoint AF2 berbeda dari static audit")
    if audit.get("test_access_authorized") is not False:
        raise RuntimeError("Static audit tidak mempertahankan test lock")

    config_path = CONFIGS[arm]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if payload.get("code") != arm:
        raise RuntimeError("Kode config tidak konsisten")
    afab = AFABConfig.from_mapping(payload["afab"])
    adapter = AF2FFAConfig.from_mapping(payload["af2_ffa"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / arm / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    if not _complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_af2_ffa_trainer(
            afab, adapter, initial_checkpoint=checkpoint
        )
        with _exclusive_training_lock(
            output_root, lock_name=f"{arm}_seed{seed}.training.lock"
        ):
            if last.is_file():
                print(f"RESUME {arm} seed {seed} dari last.pt", flush=True)
                model = YOLO(str(last))
                args = {"resume": True, "device": device}
            else:
                print(f"START {arm} seed {seed} dari AF2 seed-matched", flush=True)
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
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "training_complete.json").write_text(
            json.dumps(
                {
                    "trainer_returned": True,
                    "epochs_requested": epochs,
                    "seed": seed,
                    "arm": arm,
                    "initial_checkpoint_sha256": _sha256(checkpoint),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not _complete(run_dir, epochs):
        raise RuntimeError(f"Run belum lengkap: {run_dir}")

    report_path = output_root / "val_reports" / f"{arm}_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    metrics = report["metrics"]
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.af2_ffa.arm_result.v1",
        "arm": arm,
        "seed": seed,
        "conditioning": adapter.conditioning,
        "metrics": metrics,
        "checkpoint": str(best),
        "initial_af2_checkpoint": str(checkpoint),
        "initial_af2_checkpoint_sha256": _sha256(checkpoint),
        "config": str(config_path),
        "static_audit": str(audit_path),
        "training_executed": training_executed,
        "test_images_accessed": False,
    }
    result_path = output_root / "val_reports" / f"{arm}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AF2 feature-frequency arm")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2_ffa_arm(
        args.arm,
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
