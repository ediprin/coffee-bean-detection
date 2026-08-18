"""Paired validation confirmation for standalone WAV1.

Seed 42 is reused from the completed spectral Stage-2 run. Only seeds 123 and
2026 are newly trained from the corresponding existing D0 checkpoints. The
completed AF2/IGEM paired-confirmation artifact supplies D0FT controls and
three-seed AF2/IGEM references. Faruq locked test is never restored or read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_spectral import SpectralFrontendConfig, SpectralInputEnhancer, make_spectral_trainer
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _epochs,
    _exclusive_training_lock,
    _recover_from_best,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_spectral/WAV1_yolo26n.yaml"
SEED42 = 42
CONFIRMATION_SEEDS = (123, 2026)
ALL_SEEDS = (SEED42, *CONFIRMATION_SEEDS)
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
EXPECTED_SEED42 = {
    "macro_map50_95": 0.8841052369918866,
    "bottom3_class_map50_95": 0.8327607439278027,
    "worst_class_map50_95": 0.8203489485589485,
}
REFERENCE_MEANS = {
    "AF2": {
        "macro_map50_95": 0.8793765273831853,
        "bottom3_class_map50_95": 0.7937036279638393,
        "worst_class_map50_95": 0.7815268371194097,
        "status": "PASS_vs_D0FT",
    },
    "IGEM1": {
        "macro_map50_95": 0.8771367301594344,
        "bottom3_class_map50_95": 0.7926573397931215,
        "worst_class_map50_95": 0.7773973469475308,
        "status": "PASS_vs_D0FT",
    },
    "STB1": {
        "macro_map50_95": 0.8781987645071222,
        "bottom3_class_map50_95": 0.8049539441492847,
        "worst_class_map50_95": 0.7835956356579805,
        "status": "paired_validation_reference_spatial_causal_FAIL",
    },
    "ACMC1": {
        "macro_map50_95": 0.8762,
        "bottom3_class_map50_95": 0.7913,
        "worst_class_map50_95": 0.7630,
        "status": "rounded_master_log_reference_locked_test_NOT_CONFIRMED",
    },
}


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


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


def _validate_frontend_contract() -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "WAV1":
        raise RuntimeError("Config WAV1 berubah/code tidak konsisten")
    spectral = SpectralFrontendConfig.from_mapping(payload["spectral"])
    if spectral.arm != "WAV1" or spectral.wavelet_levels != 2:
        raise RuntimeError("Kontrak WAV1 bukan arm WAV1 dua-level yang dibekukan")
    frontend = SpectralInputEnhancer(spectral)
    if any(parameter.requires_grad for parameter in frontend.parameters()):
        raise RuntimeError("WAV1 tidak lagi parameter-free")
    if frontend.state_dict():
        raise RuntimeError("WAV1 memperoleh persistent trainable/stateful buffer")
    torch.manual_seed(20260819)
    probe = torch.rand(1, 3, 65, 63, dtype=torch.float32, requires_grad=True)
    first = frontend(probe)
    second = frontend(probe)
    repeatable = torch.equal(first.detach(), second.detach())
    finite = bool(torch.isfinite(first).all().item())
    first.mean().backward()
    finite_gradient = probe.grad is not None and bool(torch.isfinite(probe.grad).all().item())
    if not repeatable or not finite or not finite_gradient:
        raise RuntimeError(
            f"Static WAV1 contract gagal: repeatable={repeatable}, finite={finite}, finite_gradient={finite_gradient}"
        )
    return {
        "config_sha256": _sha256(CONFIG),
        "arm": spectral.arm,
        "wavelet_levels": spectral.wavelet_levels,
        "parameter_free": True,
        "no_persistent_state": True,
        "repeatable_cpu_bitwise": repeatable,
        "finite_output": finite,
        "finite_gradient": finite_gradient,
    }


def _validate_seed42(path: str | Path) -> tuple[dict, dict]:
    payload = _load_json(path, "WAV1 seed42")
    if (
        payload.get("format") != "coffee_detector.af2_spectral.arm_result.v1"
        or payload.get("arm") != "WAV1"
        or int(payload.get("seed", -1)) != 42
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
    ):
        raise RuntimeError("WAV1 seed42 bukan artifact spectral validation-only yang kompatibel")
    values = _metrics(payload)
    for metric, expected in EXPECTED_SEED42.items():
        if abs(values[metric] - expected) > 1e-12:
            raise RuntimeError(
                f"WAV1 seed42 berbeda dari hasil yang dibekukan: {metric}={values[metric]} != {expected}"
            )
    return values, payload


def _validate_reference(path: str | Path) -> dict:
    payload = _load_json(path, "AF2/IGEM paired reference")
    if (
        payload.get("seeds") != [42, 123, 2026]
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
    ):
        raise RuntimeError("Reference AF2/IGEM bukan evidence tiga-seed validation-only")
    for seed in ALL_SEEDS:
        row = payload.get("per_seed", {}).get(str(seed), {})
        _metrics(row["D0FT"])
        _metrics(row["AF2"])
        _metrics(row["IGEM1"])
    return payload


def _recover_if_corrupt(run_dir: Path) -> dict | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        return None
    try:
        _epochs(csv_path)
    except RuntimeError:
        result = _recover_from_best(run_dir)
        print(f"RECOVERY {run_dir.name}: {result}", flush=True)
        return result
    return None


def _train_seed(
    data_root: Path,
    checkpoint: Path,
    output_root: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[dict, bool, dict | None]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    spectral = SpectralFrontendConfig.from_mapping(payload["spectral"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / "WAV1" / f"WAV1_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.wav1_paired.run_contract.v1",
        "arm": "WAV1",
        "seed": seed,
        "config_sha256": _sha256(CONFIG),
        "d0_checkpoint_sha256": _sha256(checkpoint),
        "epochs": epochs,
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _load_json(contract_path, "Run contract") != contract:
        raise RuntimeError(f"Run directory memiliki kontrak berbeda: {run_dir}")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    recovery = _recover_if_corrupt(run_dir)
    trained = False
    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_spectral_trainer(spectral, d0_checkpoint=checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(output_root, lock_name=f"WAV1_seed{seed}.training.lock"):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME WAV1 seed {seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True}
            else:
                print(f"START WAV1 seed {seed} dari D0 seed {seed}", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root / "WAV1"),
                    name=f"WAV1_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=False,
                    verbose=False,
                )
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        trained = True

    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"WAV1 seed {seed} belum lengkap: {run_dir}")
    report_path = output_root / "val_reports" / f"WAV1_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError(f"Validation WAV1 seed {seed} kehilangan kelas")
    return report, trained, recovery


def _aggregate(per_seed: dict[str, dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for metric in METRICS:
        control = [float(per_seed[str(seed)]["D0FT"][metric]) for seed in ALL_SEEDS]
        candidate = [float(per_seed[str(seed)]["WAV1"][metric]) for seed in ALL_SEEDS]
        deltas = [right - left for left, right in zip(control, candidate)]
        result[metric] = {
            "d0ft_mean": statistics.fmean(control),
            "d0ft_std": statistics.stdev(control),
            "wav1_mean": statistics.fmean(candidate),
            "wav1_std": statistics.stdev(candidate),
            "paired_delta_mean": statistics.fmean(deltas),
            "paired_delta_std": statistics.stdev(deltas),
            "paired_delta_min": min(deltas),
            "improved_seeds": sum(delta > 0.0 for delta in deltas),
            "deltas": dict(zip((str(seed) for seed in ALL_SEEDS), deltas)),
        }
    return result


def _primary_decision(aggregate: dict[str, dict]) -> tuple[dict[str, bool], str]:
    criteria = {
        "macro_gain_at_least_0_5_point": aggregate["macro_map50_95"]["paired_delta_mean"] >= 0.005,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_not_lower": aggregate["bottom3_class_map50_95"]["paired_delta_mean"] >= 0.0,
        "bottom3_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["improved_seeds"] >= 2,
        "worst_mean_drop_no_more_than_1_point": aggregate["worst_class_map50_95"]["paired_delta_mean"] >= -0.01,
    }
    return criteria, "PASS" if all(criteria.values()) else "FAIL"


def _descriptive_reference_table(aggregate: dict[str, dict]) -> dict:
    wav1 = {metric: aggregate[metric]["wav1_mean"] for metric in METRICS}
    rows = {"WAV1": {**wav1, "status": "current_candidate"}, **REFERENCE_MEANS}
    for name, row in rows.items():
        row["delta_macro_vs_WAV1"] = float(row["macro_map50_95"] - wav1["macro_map50_95"])
        row["delta_bottom3_vs_WAV1"] = float(row["bottom3_class_map50_95"] - wav1["bottom3_class_map50_95"])
        row["delta_worst_vs_WAV1"] = float(row["worst_class_map50_95"] - wav1["worst_class_map50_95"])
    order = sorted(rows, key=lambda name: rows[name]["macro_map50_95"], reverse=True)
    return {
        "rows": rows,
        "macro_rank_order": order,
        "note": "descriptive only; primary confirmatory decision is WAV1 versus seed-matched D0FT",
    }


def run_wav1_paired_confirmation(
    data_root: str | Path,
    grouped_summary: str | Path,
    wav1_seed42_result: str | Path,
    af2_igem_reference: str | Path,
    d0_seed123: str | Path,
    d0_seed2026: str | Path,
    output_root: str | Path,
    *,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if not authorize_training:
        raise RuntimeError("WAV1 paired confirmation belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    load_faruq_grouped_summary(grouped_summary, data_root)
    dataset_audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    frontend_contract = _validate_frontend_contract()
    seed42, seed42_payload = _validate_seed42(wav1_seed42_result)
    reference = _validate_reference(af2_igem_reference)

    checkpoints = {123: Path(d0_seed123).expanduser().resolve(), 2026: Path(d0_seed2026).expanduser().resolve()}
    for seed, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"D0 seed {seed} tidak ditemukan: {checkpoint}")
        if _checkpoint_seed(checkpoint) != seed:
            raise RuntimeError(f"Checkpoint D0 tidak cocok dengan seed {seed}")
        if _checkpoint_nc(checkpoint) != 21:
            raise RuntimeError(f"D0 seed {seed} bukan detector 21 kelas")
        expected_sha = reference["per_seed"][str(seed)].get("d0_checkpoint_sha256")
        if expected_sha and _sha256(checkpoint) != expected_sha:
            raise RuntimeError(f"SHA D0 seed {seed} berbeda dari paired reference")

    per_seed: dict[str, dict] = {
        "42": {
            "D0FT": _metrics(reference["per_seed"]["42"]["D0FT"]),
            "WAV1": seed42,
            "source": str(Path(wav1_seed42_result).expanduser().resolve()),
        }
    }
    execution: dict[str, dict] = {}
    for seed in CONFIRMATION_SEEDS:
        print(f"\n=== WAV1 PAIRED CONFIRMATION | SEED {seed} ===", flush=True)
        report, trained, recovery = _train_seed(
            data_root, checkpoints[seed], output_root, seed=seed, device=device
        )
        per_seed[str(seed)] = {
            "D0FT": _metrics(reference["per_seed"][str(seed)]["D0FT"]),
            "WAV1": _metrics(report),
            "d0_checkpoint_sha256": _sha256(checkpoints[seed]),
        }
        execution[str(seed)] = {
            "training_executed_this_call": trained,
            "recovery": recovery,
            "report": str(reports / f"WAV1_seed{seed}_val.json"),
        }

    aggregate = _aggregate(per_seed)
    criteria, decision = _primary_decision(aggregate)
    descriptive = _descriptive_reference_table(aggregate)
    result = {
        "format": "coffee_detector.wav1_paired_confirmation.v1",
        "protocol": "faruq-v3-wav1-paired-confirmation-v1",
        "seeds": list(ALL_SEEDS),
        "newly_trained_seeds": list(CONFIRMATION_SEEDS),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "frontend_contract": frontend_contract,
        "seed42_latency": seed42_payload.get("latency"),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "descriptive_reference_comparison": descriptive,
        "next_action": (
            "REPORT_WAV1_VALIDATION_ROBUSTNESS_WITHOUT_REOPENING_TEST"
            if decision == "PASS"
            else "STOP_WAV1_WITHOUT_RETUNE_OR_TEST"
        ),
        "execution": execution,
    }
    path = reports / "wav1_paired_confirmation.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="WAV1 paired multiseed confirmation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--wav1-seed42-result", required=True)
    parser.add_argument("--af2-igem-reference", required=True)
    parser.add_argument("--d0-seed123", required=True)
    parser.add_argument("--d0-seed2026", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_wav1_paired_confirmation(
        args.data_root,
        args.grouped_summary,
        args.wav1_seed42_result,
        args.af2_igem_reference,
        args.d0_seed123,
        args.d0_seed2026,
        args.output_root,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
