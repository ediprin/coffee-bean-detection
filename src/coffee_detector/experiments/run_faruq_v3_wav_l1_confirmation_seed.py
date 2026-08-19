"""Train exactly one frozen WAV_L1 confirmation seed (123 or 2026).

Seed 42 is immutable repository evidence from the completed Stage-1 screen.
This runner is deliberately isolated per seed so two Colab accounts can run
123 and 2026 in parallel without sharing writable run directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    WAV1FactorizationConfig,
    WAV1FactorizationEnhancer,
    make_factorization_trainer,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/wav1_factorization/WAV_L1_yolo26n.yaml"
PROTOCOL = REPO_ROOT / "docs/FARUQ_V3_WAV_L1_PAIRED_CONFIRMATION_PROTOCOL_2026-08-19.md"
SEED42_EVIDENCE = REPO_ROOT / "docs/evidence/FARUQ_V3_WAV_L1_SEED42_RESULT_2026-08-19.json"
ALLOWED_SEEDS = (123, 2026)


def sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
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
        raise RuntimeError(f"Checkpoint tidak merekam seed: {path}")
    return int(train_args["seed"])


def _checkpoint_nc(path: Path) -> int:
    from ultralytics import YOLO

    model = YOLO(str(path)).model.cpu().eval()
    return int(getattr(model.model[-1], "nc", -1))


def _validate_frozen_contract() -> tuple[dict, WAV1FactorizationConfig]:
    if not PROTOCOL.is_file() or not SEED42_EVIDENCE.is_file():
        raise FileNotFoundError("Protocol/evidence WAV_L1 belum dibekukan")
    protocol_text = PROTOCOL.read_text(encoding="utf-8")
    if "frozen before seed-123/2026 WAV_L1 training" not in protocol_text:
        raise RuntimeError("Protocol WAV_L1 tidak memiliki frozen confirmation status")

    seed42 = _read(SEED42_EVIDENCE, "WAV_L1 seed42 evidence")
    if (
        seed42.get("format") != "coffee_detector.wav_l1.seed42_reference.v1"
        or seed42.get("arm") != "WAV_L1"
        or int(seed42.get("seed", -1)) != 42
        or seed42.get("evaluation_split") != "val"
        or seed42.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Evidence seed42 WAV_L1 tidak kompatibel")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "WAV_L1":
        raise RuntimeError("Config confirmation bukan WAV_L1")
    factorization = WAV1FactorizationConfig.from_mapping(payload.get("factorization"))
    if factorization.arm != "WAV_L1":
        raise RuntimeError("Factorization arm berubah dari WAV_L1")

    frontend = WAV1FactorizationEnhancer(factorization)
    if frontend.state_dict() or any(p.requires_grad for p in frontend.parameters()):
        raise RuntimeError("WAV_L1 harus tetap parameter/state-free")

    torch.manual_seed(20260819)
    probe = torch.rand(1, 3, 65, 63, dtype=torch.float32, requires_grad=True)
    first = frontend(probe)
    second = frontend(probe.detach())
    repeatable = torch.equal(first.detach(), second)
    finite = bool(torch.isfinite(first).all())
    active = not torch.equal(first.detach(), probe.detach())
    first.square().mean().backward()
    finite_grad = probe.grad is not None and bool(torch.isfinite(probe.grad).all())
    constant = torch.full((1, 3, 64, 64), 0.4, dtype=torch.float32)
    constant_identity = torch.equal(frontend(constant), constant)
    if not all((repeatable, finite, active, finite_grad, constant_identity)):
        raise RuntimeError(
            "Frozen WAV_L1 static contract gagal: "
            f"repeatable={repeatable}, finite={finite}, active={active}, "
            f"finite_grad={finite_grad}, constant_identity={constant_identity}"
        )

    return {
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(PROTOCOL),
        "seed42_evidence_sha256": sha256(SEED42_EVIDENCE),
        "parameter_free": True,
        "state_free": True,
        "cpu_bitwise_repeatable": repeatable,
        "finite_output": finite,
        "finite_gradient": finite_grad,
        "constant_image_identity": constant_identity,
    }, factorization


def run_confirmation_seed(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Confirmation hanya mengizinkan seed {ALLOWED_SEEDS}; seed42 harus direuse")
    if not authorize_training:
        raise RuntimeError("Training WAV_L1 confirmation belum diotorisasi")

    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos locked test")

    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError(f"D0 checkpoint bukan seed {seed}: {checkpoint}")
    if _checkpoint_nc(checkpoint) != 21:
        raise RuntimeError("D0 checkpoint bukan detector 21 kelas")

    load_faruq_grouped_summary(grouped_summary, data_root)
    contract, factorization = _validate_frozen_contract()
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    epochs = int(payload["train"]["epochs"])
    if epochs != 50:
        raise RuntimeError("Frozen confirmation schedule harus 50 epoch")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    run_dir = output_root / "WAV_L1" / f"WAV_L1_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_contract = {
        "format": "coffee_detector.wav_l1_confirmation.run_contract.v1",
        "arm": "WAV_L1",
        "seed": seed,
        "config_sha256": contract["config_sha256"],
        "protocol_sha256": contract["protocol_sha256"],
        "seed42_evidence_sha256": contract["seed42_evidence_sha256"],
        "d0_checkpoint_sha256": sha256(checkpoint),
        "epochs": epochs,
        "evaluation_split": "val",
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
            output_root, lock_name=f"WAV_L1_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME WAV_L1 seed {seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                print(f"START WAV_L1 seed {seed} dari D0 seed {seed}", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root / "WAV_L1"),
                    name=f"WAV_L1_seed{seed}",
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
        raise RuntimeError(f"WAV_L1 seed {seed} belum lengkap: {run_dir}")

    validation_path = reports / f"WAV_L1_seed{seed}_val.json"
    validation = evaluate(best, data_root, validation_path, split="val", device=device)
    if validation["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError(f"Validation WAV_L1 seed {seed} kehilangan kelas")

    result = {
        "format": "coffee_detector.wav_l1_confirmation.seed_result.v1",
        "arm": "WAV_L1",
        "seed": seed,
        "metrics": validation["metrics"],
        "checkpoint": str(best),
        "checkpoint_sha256": sha256(best),
        "initial_d0_checkpoint": str(checkpoint),
        "initial_d0_checkpoint_sha256": sha256(checkpoint),
        "config": str(CONFIG),
        "config_sha256": contract["config_sha256"],
        "protocol": str(PROTOCOL),
        "protocol_sha256": contract["protocol_sha256"],
        "seed42_evidence_sha256": contract["seed42_evidence_sha256"],
        "training_executed": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    result_path = reports / f"WAV_L1_seed{seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen WAV_L1 confirmation seed")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_confirmation_seed(
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
