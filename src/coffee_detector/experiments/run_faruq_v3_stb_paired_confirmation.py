"""Paired three-seed confirmation of STB1 against its CMC0 capacity control."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    METRICS,
    _checkpoint_state,
    _epochs,
    _exclusive_training_lock,
    _recover_from_best,
    _run_complete,
)
from coffee_detector.stb import STBConfig, make_stb_trainer
from coffee_detector.stb_control import (
    make_stb_control_trainer,
    static_stb_capacity_control_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIGS = {
    "STB1": REPO_ROOT / "configs/stb/STB1_yolo26n_swin_classification.yaml",
    "CMC0": REPO_ROOT / "configs/stb_control/CMC0_yolo26n_channel_capacity.yaml",
}
SEED42 = 42
CONFIRMATION_SEEDS = (123, 2026)


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


def _validate_seed42_result(path: str | Path) -> dict:
    payload = _load_json(path, "Hasil seed-42 STB capacity control")
    if (
        payload.get("protocol") != "faruq-v3-stb-capacity-causal-control-v1"
        or int(payload.get("seed", -1)) != SEED42
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
        or payload.get("decision") != "PASS"
    ):
        raise RuntimeError("Hasil seed 42 bukan PASS validation-only yang kompatibel")
    for arm in ("STB1", "CMC0"):
        _metrics(payload["models"][arm])
    return payload


def _checkpoint_seed(path: Path) -> int:
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(train_args, dict) or "seed" not in train_args:
        raise RuntimeError(f"Checkpoint tidak merekam seed: {path}")
    return int(train_args["seed"])


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


def _train_arm(
    arm: str,
    data_root: Path,
    d0_checkpoint: Path,
    output_root: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[dict, bool, dict | None]:
    if arm not in CONFIGS:
        raise ValueError(f"Arm tidak dikenal: {arm}")
    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    config = STBConfig.from_mapping(payload["stb"])
    epochs = int(payload["train"]["epochs"])
    arm_root = output_root / arm
    run_dir = arm_root / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    recovery = _recover_if_corrupt(run_dir)
    training_executed = False

    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = (
            make_stb_trainer(config, d0_checkpoint=d0_checkpoint)
            if arm == "STB1"
            else make_stb_control_trainer(config, d0_checkpoint=d0_checkpoint)
        )
        epoch, resumable = _checkpoint_state(last)
        lock_name = f"{arm}_seed{seed}.training.lock"
        with _exclusive_training_lock(output_root, lock_name=lock_name):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {arm} seed {seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True}
            else:
                print(f"START {arm} seed {seed} dari D0 seed yang sama", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(arm_root),
                    name=f"{arm}_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=True,
                    verbose=True,
                )
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        training_executed = True

    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run {arm} seed {seed} belum lengkap: {run_dir}")
    report_path = output_root / "val_reports" / f"{arm}_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    missing = report["metrics"].get("classes_without_ground_truth", [])
    if missing:
        raise RuntimeError(f"Validation {arm} seed {seed} kehilangan kelas: {missing}")
    return report, training_executed, recovery


def _aggregate(per_seed: dict[str, dict]) -> dict[str, dict]:
    seeds = (SEED42, *CONFIRMATION_SEEDS)
    result: dict[str, dict] = {}
    for metric in METRICS:
        control = [float(per_seed[str(seed)]["CMC0"][metric]) for seed in seeds]
        candidate = [float(per_seed[str(seed)]["STB1"][metric]) for seed in seeds]
        deltas = [right - left for left, right in zip(control, candidate)]
        result[metric] = {
            "cmc0_mean": statistics.fmean(control),
            "cmc0_std": statistics.stdev(control),
            "stb1_mean": statistics.fmean(candidate),
            "stb1_std": statistics.stdev(candidate),
            "spatial_delta_mean": statistics.fmean(deltas),
            "spatial_delta_std": statistics.stdev(deltas),
            "spatial_delta_min": min(deltas),
            "spatial_improved_seeds": sum(delta > 0.0 for delta in deltas),
            "deltas": dict(zip((str(seed) for seed in seeds), deltas)),
        }
    return result


def _decision(aggregate: dict[str, dict]) -> tuple[dict[str, bool], str]:
    criteria = {
        "macro_spatial_gain_at_least_0_5_point": aggregate["macro_map50_95"]["spatial_delta_mean"] >= 0.005,
        "macro_spatial_improved_at_least_2_of_3": aggregate["macro_map50_95"]["spatial_improved_seeds"] >= 2,
        "bottom3_spatial_mean_not_lower": aggregate["bottom3_class_map50_95"]["spatial_delta_mean"] >= 0.0,
        "bottom3_spatial_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["spatial_improved_seeds"] >= 2,
        "worst_spatial_mean_drop_no_more_than_1_point": aggregate["worst_class_map50_95"]["spatial_delta_mean"] >= -0.01,
    }
    return criteria, "PASS" if all(criteria.values()) else "FAIL"


def run_faruq_v3_stb_paired_confirmation(
    data_root: str | Path,
    grouped_summary: str | Path,
    seed42_result: str | Path,
    d0_checkpoints: tuple[str | Path, ...],
    output_root: str | Path,
    *,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    frozen_seeds = tuple(int(seed) for seed in seeds)
    if frozen_seeds != CONFIRMATION_SEEDS:
        raise ValueError(f"Konfirmasi dikunci pada seed {CONFIRMATION_SEEDS}")
    if len(d0_checkpoints) != len(CONFIRMATION_SEEDS):
        raise ValueError("Harus tersedia tepat dua checkpoint D0 untuk seed 123/2026")
    if not authorize_training:
        raise RuntimeError("Konfirmasi paired STB/CMC0 belum diotorisasi")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    seed42 = _validate_seed42_result(seed42_result)
    per_seed: dict[str, dict] = {
        str(SEED42): {
            "source": str(Path(seed42_result).expanduser().resolve()),
            "CMC0": _metrics(seed42["models"]["CMC0"]),
            "STB1": _metrics(seed42["models"]["STB1"]),
        }
    }
    execution: dict[str, dict] = {}
    static_root = output_root / "static_audits"
    static_root.mkdir(parents=True, exist_ok=True)

    for seed, checkpoint_value in zip(frozen_seeds, d0_checkpoints):
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"D0 seed {seed} tidak ditemukan: {checkpoint}")
        recorded_seed = _checkpoint_seed(checkpoint)
        if recorded_seed != seed:
            raise RuntimeError(f"Checkpoint seed mismatch: diminta {seed}, ditemukan {recorded_seed}")
        static_path = static_root / f"seed{seed}_static_audit.json"
        static = static_stb_capacity_control_audit(
            MODEL_YAML, checkpoint, static_path, image_size=128
        )
        if static["decision"] != "PASS":
            raise RuntimeError(f"Static audit seed {seed} gagal: {static_path}")

        print(f"\n=== PAIRED STB/CMC0 CONFIRMATION | SEED {seed} ===", flush=True)
        per_seed[str(seed)] = {"d0_checkpoint_sha256": _sha256(checkpoint)}
        execution[str(seed)] = {}
        for arm in ("CMC0", "STB1"):
            report, trained, recovery = _train_arm(
                arm, data_root, checkpoint, output_root, seed=seed, device=device
            )
            per_seed[str(seed)][arm] = _metrics(report)
            execution[str(seed)][arm] = {
                "training_executed_this_call": trained,
                "recovery": recovery,
                "report": str(reports / f"{arm}_seed{seed}_val.json"),
            }

    aggregate = _aggregate(per_seed)
    criteria, decision = _decision(aggregate)
    result = {
        "protocol": "faruq-v3-stb-capacity-paired-confirmation-v1",
        "seeds": [SEED42, *CONFIRMATION_SEEDS],
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "FREEZE_STB1_AS_VALIDATION_CONFIRMED_PRIMARY_CANDIDATE"
            if decision == "PASS"
            else "STOP_STB_CAUSAL_CLAIM_WITHOUT_TEST"
        ),
        "execution": execution,
    }
    summary = reports / "stb_capacity_paired_confirmation.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired multi-seed STB/CMC0 confirmation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--seed42-result", required=True)
    parser.add_argument("--d0-checkpoints", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIRMATION_SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_stb_paired_confirmation(
        args.data_root,
        args.grouped_summary,
        args.seed42_result,
        tuple(args.d0_checkpoints),
        args.output_root,
        seeds=tuple(args.seeds),
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
