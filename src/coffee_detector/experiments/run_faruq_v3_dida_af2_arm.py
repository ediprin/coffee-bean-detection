from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.afab.operator import AFABConfig
from coffee_detector.dida_af2 import DIDAAF2Config, make_dida_af2_trainer
from coffee_detector.evaluate import evaluate


REPO_ROOT = Path(__file__).resolve().parents[3]
ARMS = ("AF2FT", "AF2DG", "AF2FG", "AF2DGFG")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _complete(run_dir: Path, epochs: int) -> bool:
    best = run_dir / "weights/best.pt"
    csv = run_dir / "results.csv"
    marker = run_dir / "training_complete.json"
    if best.is_file() and csv.is_file() and marker.is_file():
        payload = _read_json(marker, "Training marker")
        return bool(payload.get("trainer_returned")) and int(payload.get("epochs_requested", -1)) == epochs
    if not best.is_file() or not csv.is_file():
        return False
    rows = [line for line in csv.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    return len(rows) >= epochs


def run_faruq_v3_dida_af2_arm(
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
    if seed != 42:
        raise ValueError("Screening awal DIDA-AF2 dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi; gunakan --authorize-training")
    data_root = Path(data_root).expanduser().resolve()
    grouped_summary = Path(grouped_summary).expanduser().resolve()
    af2_checkpoint = Path(af2_checkpoint).expanduser().resolve()
    static_audit = Path(static_audit).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Test terdeteksi; gunakan development archive tanpa test")
    if not (data_root / "data.yaml").is_file() or not af2_checkpoint.is_file():
        raise FileNotFoundError("Dataset development atau checkpoint AF2 tidak lengkap")
    _read_json(grouped_summary, "Grouped summary")
    audit = _read_json(static_audit, "Static audit")
    if audit.get("decision") != "PASS" or not audit.get("training_authorized"):
        raise RuntimeError("Static audit belum PASS")
    if audit.get("test_access_authorized") is not False:
        raise RuntimeError("Static audit tidak mempertahankan test lock")
    if audit.get("checkpoint_sha256") != _sha256(af2_checkpoint):
        raise RuntimeError("Checkpoint AF2 berbeda dari checkpoint yang diaudit")

    config_path = REPO_ROOT / f"configs/dida_af2/{arm}_yolo26n.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if payload.get("code") != arm:
        raise RuntimeError("Kode config tidak konsisten")
    afab = AFABConfig.from_mapping(payload["afab"])
    dida = DIDAAF2Config.from_mapping(payload["dida"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / arm / f"{arm}_seed{seed}"
    last = run_dir / "weights/last.pt"
    best = run_dir / "weights/best.pt"
    training_executed = False
    if not _complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_dida_af2_trainer(
            afab, dida, initial_checkpoint=af2_checkpoint
        )
        if last.is_file():
            print(f"RESUME {arm} seed {seed} dari {last}", flush=True)
            model = YOLO(str(last))
            args = {"resume": True, "device": device}
        else:
            print(f"START {arm} seed {seed} dari AF2 yang sama", flush=True)
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
        marker = run_dir / "training_complete.json"
        marker.write_text(
            json.dumps(
                {
                    "trainer_returned": True,
                    "epochs_requested": epochs,
                    "seed": seed,
                    "arm": arm,
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
        "format": "coffee_detector.dida_af2.arm_result.v1",
        "arm": arm,
        "seed": seed,
        "mode": dida.mode,
        "metrics": metrics,
        "checkpoint": str(best),
        "config": str(config_path),
        "static_audit": str(static_audit),
        "training_executed": training_executed,
        "test_images_accessed": False,
    }
    result_path = output_root / "val_reports" / f"{arm}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one DIDA-AF2 factorial arm")
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
    run_faruq_v3_dida_af2_arm(
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
