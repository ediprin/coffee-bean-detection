from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_parent_residual import (
    AF2ParentResidualConfig,
    make_af2_parent_residual_trainer,
)
from coffee_detector.afab import AFABConfig
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import _exclusive_training_lock


REPO_ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42, 123, 2026)
ARMS = ("AF2IGEM0", "AF2IGEM1")
CONFIGS = {code: REPO_ROOT / f"configs/af2_parent_residual/{code}.yaml" for code in ARMS}


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
        raise RuntimeError(f"Checkpoint AF2FS tidak merekam seed: {path}")
    return int(args["seed"])


def _complete(run_dir: Path, epochs: int, source_sha: str, arm: str, seed: int) -> bool:
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights/best.pt"
    if not marker.is_file() or not best.is_file():
        return False
    payload = _read(marker, "Training marker")
    return (
        payload.get("trainer_returned") is True
        and payload.get("arm") == arm
        and int(payload.get("seed", -1)) == seed
        and int(payload.get("epochs_requested", -1)) == epochs
        and payload.get("initial_checkpoint_sha256") == source_sha
    )


def run_faruq_v3_af2_igem_parent_arm(
    arm: str,
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_checkpoint: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Arm harus salah satu {ARMS}")
    if seed not in SEEDS:
        raise RuntimeError(f"Seed harus salah satu {SEEDS}")
    if not authorize_training:
        raise RuntimeError("Training IGEM frozen-parent belum diotorisasi")

    data_root = Path(data_root).expanduser().resolve()
    grouped_path = Path(grouped_summary).expanduser().resolve()
    checkpoint = Path(af2_checkpoint).expanduser().resolve()
    audit_path = Path(static_audit).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    if not (data_root / "data.yaml").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Dataset development atau checkpoint AF2FS tidak lengkap")
    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError(f"Checkpoint parent tidak seed-matched untuk seed {seed}")

    grouped = _read(grouped_path, "Grouped summary")
    if grouped.get("test_images_accessed") not in {False, None}:
        raise RuntimeError("Grouped summary tidak mempertahankan test lock")

    source_sha = _sha256(checkpoint)
    audit = _read(audit_path, "IGEM static audit")
    if (
        audit.get("format") != "coffee_detector.af2_parent_residual.igem_static_audit.v1"
        or audit.get("decision") != "PASS"
        or audit.get("training_authorized") is not True
        or audit.get("checkpoint_sha256") != source_sha
        or audit.get("test_access_authorized") is not False
    ):
        raise RuntimeError("Static audit tidak mengotorisasi checkpoint AF2FS ini")

    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    if payload.get("code") != arm:
        raise RuntimeError(f"Config {arm} tidak konsisten")
    afab = AFABConfig.from_mapping(payload["afab"])
    residual = AF2ParentResidualConfig.from_mapping(payload["parent_residual"])
    if residual.family != "igem":
        raise RuntimeError("Runner ini hanya untuk family IGEM")

    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / arm / f"{arm}_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False

    if not _complete(run_dir, epochs, source_sha, arm, seed):
        from ultralytics import YOLO

        trainer = make_af2_parent_residual_trainer(
            afab, residual, initial_checkpoint=checkpoint
        )
        with _exclusive_training_lock(
            output_root, lock_name=f"{arm}_seed{seed}.training.lock"
        ):
            if last.is_file():
                print(f"RESUME {arm} seed {seed} dari last.pt", flush=True)
                model = YOLO(str(last))
                args = {"resume": True, "device": device}
            else:
                print(f"START {arm} seed {seed} dari frozen AF2FS parent", flush=True)
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
                    save=True,
                    save_period=-1,
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
                    "initial_checkpoint_sha256": source_sha,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if not _complete(run_dir, epochs, source_sha, arm, seed):
        raise RuntimeError(f"Run {arm} seed {seed} belum lengkap; last.pt={last}")

    reports = output_root / "val_reports"
    parent_report_path = reports / f"AF2FS_seed{seed}_parent_reference_val.json"
    arm_report_path = reports / f"{arm}_seed{seed}_val.json"
    baseline = evaluate(checkpoint, data_root, parent_report_path, split="val", device=device)
    candidate = evaluate(best, data_root, arm_report_path, split="val", device=device)

    for label, report in (("AF2FS", baseline), (arm, candidate)):
        if report["metrics"].get("classes_without_ground_truth"):
            raise RuntimeError(f"Validation {label} seed {seed} kehilangan kelas")
        if len(report["metrics"].get("map50_95_by_class", {})) != 21:
            raise RuntimeError(f"Validation {label} seed {seed} tidak memuat 21 kelas")

    result = {
        "format": "coffee_detector.af2_parent_residual.igem_arm_result.v1",
        "protocol": "faruq-v3-af2fs-igem-parent-confirmation-v1",
        "arm": arm,
        "family": "igem",
        "conditioning": residual.conditioning,
        "seed": seed,
        "baseline_metrics": baseline["metrics"],
        "metrics": candidate["metrics"],
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "initial_af2_checkpoint": str(checkpoint),
        "initial_af2_checkpoint_sha256": source_sha,
        "parent_frozen": True,
        "trainable_scope": "igem_residual_only",
        "evaluation_split": "val",
        "static_audit": str(audit_path),
        "config": str(CONFIGS[arm]),
        "training_executed_this_call": training_executed,
        "test_images_accessed": False,
    }
    destination = reports / f"{arm}_seed{seed}_result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["summary"] = str(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2FS frozen-parent IGEM paired arm")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_af2_igem_parent_arm(
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
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
