from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_cpe import make_af2_cpe_trainer
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import _exclusive_training_lock

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("AF2CPE0", "AF2CPE5")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_arm(arm: str, data_root: str | Path, grouped_summary: str | Path,
            af2_checkpoint: str | Path, static_audit: str | Path, output_root: str | Path,
            *, seed: int = 42, device: str = "0", authorize_training: bool = False) -> dict:
    if arm not in ARMS or seed != 42:
        raise ValueError("Protocol hanya mengizinkan AF2CPE0/AF2CPE5 pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training")
    data_root, grouped_summary = Path(data_root).resolve(), Path(grouped_summary).resolve()
    checkpoint, audit_path, output_root = Path(af2_checkpoint).resolve(), Path(static_audit).resolve(), Path(output_root).resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    grouped, audit = _read(grouped_summary), _read(audit_path)
    if not grouped.get("training_ready") or not grouped.get("test_locked"):
        raise RuntimeError("Grouped development contract tidak valid")
    if audit.get("decision") != "PASS" or not audit.get("training_authorized") or audit.get("test_access_authorized") is not False:
        raise RuntimeError("Static audit belum mengotorisasi training")
    if audit.get("checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError("Checkpoint AF2 tidak sama dengan yang diaudit")
    config_path = ROOT / f"configs/af2_cpe/{arm}_yolo26n.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = output_root / arm / f"{arm}_seed42"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    marker = run_dir / "training_complete.json"
    complete = best.is_file() and marker.is_file() and _read(marker).get("trainer_returned") is True
    executed = False
    if not complete:
        from ultralytics import YOLO

        trainer = make_af2_cpe_trainer(payload["afab"], payload["cpe"], af2_checkpoint=checkpoint)
        with _exclusive_training_lock(output_root, lock_name=f"{arm}_seed42.training.lock"):
            if last.is_file():
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                model, args = YOLO(str(ROOT / payload["model"])), dict(payload["train"])
                args.update(data=str(data_root / "data.yaml"), project=str(output_root / arm),
                            name=f"{arm}_seed42", exist_ok=True, seed=42, deterministic=True,
                            plots=False, verbose=False, device=device)
            model.train(trainer=trainer, **args)
        executed = True
        marker.write_text(json.dumps({"trainer_returned": True, "epochs_requested": 50,
                                      "seed": 42, "arm": arm,
                                      "initial_af2_checkpoint_sha256": _sha256(checkpoint)}, indent=2) + "\n")
    if not best.is_file():
        raise RuntimeError(f"Checkpoint best tidak tersedia: {best}")
    report_path = output_root / "val_reports" / f"{arm}_seed42_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {"format": "coffee_detector.af2_cpe.arm_result.v1", "arm": arm, "seed": 42,
              "metrics": report["metrics"], "checkpoint": str(best), "config": str(config_path),
              "initial_af2_checkpoint": str(checkpoint), "static_audit": str(audit_path),
              "training_executed": executed, "test_images_accessed": False}
    result_path = output_root / "val_reports" / f"{arm}_seed42_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Run one frozen AF2+CPE0 seed42 arm")
    p.add_argument("--arm", choices=ARMS, required=True); p.add_argument("--data-root", required=True)
    p.add_argument("--grouped-summary", required=True); p.add_argument("--af2-checkpoint", required=True)
    p.add_argument("--static-audit", required=True); p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="0")
    p.add_argument("--authorize-training", action="store_true")
    a = p.parse_args(); print(json.dumps(run_arm(a.arm, a.data_root, a.grouped_summary, a.af2_checkpoint,
        a.static_audit, a.output_root, seed=a.seed, device=a.device, authorize_training=a.authorize_training), indent=2))


if __name__ == "__main__":
    main()
