from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_parent_residual import AF2ParentResidualConfig
from coffee_detector.af2_parent_residual.igem_confirmation import (
    AUDIT_REVISION,
    _parent_transfer_exact,
)
from coffee_detector.af2_parent_residual.igem_trainer import make_af2_igem_confirmation_trainer
from coffee_detector.afab import AFABConfig
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import _exclusive_training_lock


REPO_ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42, 123, 2026)
ARMS = ("AF2IGEM0", "AF2IGEM1")
CONFIGS = {code: REPO_ROOT / f"configs/af2_parent_residual/{code}.yaml" for code in ARMS}
PROTOCOL = "faruq-v3-af2fs-igem-parent-confirmation-v1"
RUN_CONTRACT_FORMAT = "coffee_detector.af2_parent_residual.igem_run_contract.v2"


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
        raise RuntimeError(f"Checkpoint tidak merekam seed: {path}")
    return int(args["seed"])


def _canonical_parent_result(path: Path, seed: int, checkpoint_sha: str) -> dict:
    payload = _read(path, "AF2FS parent result")
    if (
        payload.get("format") != "coffee_detector.af2_ffa.from_start_arm_result.v1"
        or payload.get("arm") != "AF2FS"
        or int(payload.get("seed", -1)) != seed
        or payload.get("checkpoint_sha256") != checkpoint_sha
        or payload.get("test_images_accessed") is not False
    ):
        raise RuntimeError(f"AF2FS parent result seed {seed} tidak cocok dengan checkpoint resmi")
    metrics = payload.get("metrics", {})
    if len(metrics.get("map50_95_by_class", {})) != 21:
        raise RuntimeError(f"AF2FS parent seed {seed} tidak memuat 21 kelas")
    return payload


def _epochs_recorded(run_dir: Path) -> int:
    csv = run_dir / "results.csv"
    if not csv.is_file():
        return 0
    return max(0, len(csv.read_text(encoding="utf-8", errors="replace").splitlines()) - 1)


def _complete(run_dir: Path, contract: dict, arm: str, seed: int) -> bool:
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights/best.pt"
    contract_path = run_dir / "run_contract.json"
    if not marker.is_file() or not best.is_file() or not contract_path.is_file():
        return False
    if _read(contract_path, "Run contract") != contract:
        return False
    payload = _read(marker, "Training marker")
    recorded = int(payload.get("epochs_recorded", -1))
    requested = int(contract["epochs_requested"])
    return (
        payload.get("trainer_returned") is True
        and payload.get("arm") == arm
        and int(payload.get("seed", -1)) == seed
        and int(payload.get("epochs_requested", -1)) == requested
        and 1 <= recorded <= requested
        and payload.get("initial_checkpoint_sha256") == contract["initial_af2_checkpoint_sha256"]
        and payload.get("run_contract_sha256") == _sha256(contract_path)
    )


def _verify_best_parent_exact(
    parent_checkpoint: Path,
    best_checkpoint: Path,
    *,
    expected_conditioning: str,
) -> bool:
    """Require the serialized EMA checkpoint to preserve every AF2 parent state item exactly."""

    from ultralytics import YOLO

    source = YOLO(str(parent_checkpoint)).model.eval()
    candidate = YOLO(str(best_checkpoint)).model.eval()
    head = candidate.model[-1]
    config = getattr(head, "config", None)
    if config is None or getattr(config, "family", None) != "igem":
        raise RuntimeError("best.pt bukan AF2+IGEM parent-residual checkpoint")
    if getattr(config, "conditioning", None) != expected_conditioning:
        raise RuntimeError("best.pt conditioning tidak cocok dengan arm")
    exact = bool(_parent_transfer_exact(source, candidate))
    if not exact:
        raise RuntimeError("best.pt gagal parent-preservation: frozen AF2 state tidak bitwise exact")
    return exact


def run_faruq_v3_af2_igem_parent_arm(
    arm: str,
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_checkpoint: str | Path,
    parent_result: str | Path,
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
    parent_result_path = Path(parent_result).expanduser().resolve()
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
    parent = _canonical_parent_result(parent_result_path, seed, source_sha)
    audit = _read(audit_path, "IGEM static audit")
    if (
        audit.get("format") != "coffee_detector.af2_parent_residual.igem_static_audit.v1"
        or audit.get("audit_revision") != AUDIT_REVISION
        or audit.get("decision") != "PASS"
        or audit.get("training_authorized") is not True
        or audit.get("checkpoint_sha256") != source_sha
        or audit.get("test_access_authorized") is not False
    ):
        raise RuntimeError("Static audit audited-revision tidak mengotorisasi checkpoint AF2FS ini")

    config_path = CONFIGS[arm]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if payload.get("code") != arm:
        raise RuntimeError(f"Config {arm} tidak konsisten")
    afab = AFABConfig.from_mapping(payload["afab"])
    residual = AF2ParentResidualConfig.from_mapping(payload["parent_residual"])
    if residual.family != "igem":
        raise RuntimeError("Runner ini hanya untuk family IGEM")

    epochs = int(payload["train"]["epochs"])
    if epochs != 20:
        raise RuntimeError("IGEM frozen-parent confirmation dibekukan pada maksimum 20 epoch")
    run_dir = output_root / arm / f"{arm}_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    contract = {
        "format": RUN_CONTRACT_FORMAT,
        "protocol": PROTOCOL,
        "arm": arm,
        "conditioning": residual.conditioning,
        "seed": seed,
        "initial_af2_checkpoint_sha256": source_sha,
        "parent_result_sha256": _sha256(parent_result_path),
        "config_sha256": _sha256(config_path),
        "static_audit_revision": AUDIT_REVISION,
        "static_audit_checkpoint_sha256": audit["checkpoint_sha256"],
        "epochs_requested": epochs,
        "trainable_scope": "igem_residual_only",
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    stale_training_artifacts = any(
        path.exists()
        for path in (last, best, run_dir / "results.csv", run_dir / "training_complete.json")
    )
    if contract_path.is_file():
        if _read(contract_path, "Run contract") != contract:
            raise RuntimeError(f"Run directory memiliki kontrak berbeda: {run_dir}")
    elif stale_training_artifacts:
        raise RuntimeError(f"Menolak resume artifact tanpa run_contract: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    training_executed = False
    if not _complete(run_dir, contract, arm, seed):
        from ultralytics import YOLO

        trainer = make_af2_igem_confirmation_trainer(
            afab, residual, initial_checkpoint=checkpoint
        )
        with _exclusive_training_lock(
            output_root, lock_name=f"{arm}_seed{seed}.training.lock"
        ):
            if last.is_file():
                if _checkpoint_seed(last) != seed:
                    raise RuntimeError(f"last.pt bukan seed {seed}: {last}")
                print(f"RESUME {arm} seed {seed} dari audited last.pt", flush=True)
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
        actual_epochs = _epochs_recorded(run_dir)
        if actual_epochs < 1 or actual_epochs > epochs or not best.is_file():
            raise RuntimeError(f"Trainer kembali tanpa epoch/best checkpoint valid: {run_dir}")
        (run_dir / "training_complete.json").write_text(
            json.dumps(
                {
                    "trainer_returned": True,
                    "epochs_requested": epochs,
                    "epochs_recorded": actual_epochs,
                    "early_stopped": actual_epochs < epochs,
                    "seed": seed,
                    "arm": arm,
                    "initial_checkpoint_sha256": source_sha,
                    "run_contract_sha256": _sha256(contract_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if not _complete(run_dir, contract, arm, seed):
        raise RuntimeError(f"Run {arm} seed {seed} belum lengkap; last.pt={last}")

    final_parent_exact = _verify_best_parent_exact(
        checkpoint,
        best,
        expected_conditioning=residual.conditioning,
    )

    reports = output_root / "val_reports"
    arm_report_path = reports / f"{arm}_seed{seed}_val.json"
    candidate = evaluate(best, data_root, arm_report_path, split="val", device=device)
    if candidate["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError(f"Validation {arm} seed {seed} kehilangan kelas")
    if len(candidate["metrics"].get("map50_95_by_class", {})) != 21:
        raise RuntimeError(f"Validation {arm} seed {seed} tidak memuat 21 kelas")

    result = {
        "format": "coffee_detector.af2_parent_residual.igem_arm_result.v2",
        "protocol": PROTOCOL,
        "arm": arm,
        "family": "igem",
        "conditioning": residual.conditioning,
        "seed": seed,
        "baseline_metrics": parent["metrics"],
        "baseline_source": "canonical_AF2FS_parent_result",
        "metrics": candidate["metrics"],
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "initial_af2_checkpoint": str(checkpoint),
        "initial_af2_checkpoint_sha256": source_sha,
        "parent_result": str(parent_result_path),
        "parent_result_sha256": _sha256(parent_result_path),
        "parent_frozen": True,
        "final_parent_state_bitwise_exact": final_parent_exact,
        "trainable_scope": "igem_residual_only",
        "evaluation_split": "val",
        "static_audit": str(audit_path),
        "static_audit_revision": AUDIT_REVISION,
        "run_contract": str(contract_path),
        "run_contract_sha256": _sha256(contract_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "training_executed_this_call": training_executed,
        "epochs_recorded": _epochs_recorded(run_dir),
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
    parser.add_argument("--parent-result", required=True)
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
        args.parent_result,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
