"""Paired validation confirmation for the retained AF2 and IGEM1 candidates.

Seed 42 is reused from the frozen breadth screen.  Only seed 123 and 2026
candidates are trained, each from the corresponding existing D0 checkpoint.
The already completed D0FT paired confirmation supplies the optimization-
matched control for all three seeds.  Test data is neither restored nor read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from coffee_detector.afab import AFABConfig, make_afab_trainer
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _epochs,
    _exclusive_training_lock,
    _recover_from_best,
    _run_complete,
)
from coffee_detector.igem import IGEMConfig, make_igem_trainer


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "AF2": REPO_ROOT / "configs/afab/AF2_yolo26n_chaotic_amplitude.yaml",
    "IGEM1": REPO_ROOT / "configs/igem/IGEM1_yolo26n_classification_guidance.yaml",
}
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
SEED42 = 42
CONFIRMATION_SEEDS = (123, 2026)
ARMS = ("AF2", "IGEM1")


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


def _validate_seed42_results(af2_path: str | Path, igem_path: str | Path) -> dict:
    af2 = _load_json(af2_path, "Hasil seed-42 AF2")
    igem = _load_json(igem_path, "Hasil seed-42 IGEM1")
    if (
        af2.get("protocol") != "faruq-v3-lfdet-afab-breadth-screening-v1"
        or int(af2.get("seed", -1)) != SEED42
        or af2.get("evaluation_split") != "val"
        or af2.get("test_images_accessed") is not False
        or af2.get("test_opened") is not False
        or af2.get("decisions", {}).get("AF2", {}).get("decision") != "RETAIN"
    ):
        raise RuntimeError("Hasil AF2 seed 42 bukan RETAIN validation-only yang kompatibel")
    if (
        igem.get("protocol")
        != "faruq-v3-igem-classification-guidance-screening-v1"
        or int(igem.get("seed", -1)) != SEED42
        or igem.get("evaluation_split") != "val"
        or igem.get("test_images_accessed") is not False
        or igem.get("test_opened") is not False
        or igem.get("decision") != "RETAIN"
    ):
        raise RuntimeError("Hasil IGEM1 seed 42 bukan RETAIN validation-only yang kompatibel")

    af2_control = _metrics(af2["controls"]["D0FT"])
    igem_control = _metrics(igem["controls"]["D0FT"])
    if any(abs(af2_control[name] - igem_control[name]) > 1e-12 for name in METRICS):
        raise RuntimeError("Kontrol D0FT seed 42 AF2 dan IGEM1 tidak identik")
    return {
        "AF2": _metrics(af2["candidate"]["AF2"]),
        "IGEM1": _metrics(igem["candidate"]["IGEM1"]),
        "D0FT": af2_control,
        "sources": {"AF2": str(Path(af2_path).resolve()), "IGEM1": str(Path(igem_path).resolve())},
    }


def _validate_d0ft_confirmation(path: str | Path, seed42_control: dict) -> dict:
    payload = _load_json(path, "Konfirmasi paired D0FT")
    if (
        payload.get("protocol") != "faruq-v3-acmc-paired-optimization-confirmation-v1"
        or payload.get("seeds") != [42, 123, 2026]
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
        or payload.get("decision") != "PASS"
    ):
        raise RuntimeError("Konfirmasi D0FT bukan PASS validation-only yang kompatibel")
    for seed in (SEED42, *CONFIRMATION_SEEDS):
        _metrics(payload["per_seed"][str(seed)]["results"]["D0FT"])
    paired_seed42 = _metrics(payload["per_seed"][str(SEED42)]["results"]["D0FT"])
    if any(abs(paired_seed42[name] - seed42_control[name]) > 1e-12 for name in METRICS):
        raise RuntimeError("D0FT seed 42 pada breadth screen dan paired control berbeda")
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


def _config(arm: str) -> tuple[dict, object]:
    if arm not in CONFIGS:
        raise ValueError(f"Arm tidak dikenal: {arm}")
    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    if payload.get("code") != arm:
        raise RuntimeError(f"Config {arm} tidak konsisten")
    frozen = (
        AFABConfig.from_mapping(payload["afab"])
        if arm == "AF2"
        else IGEMConfig.from_mapping(payload["igem"])
    )
    return payload, frozen


def _trainer(arm: str, frozen: object, d0_checkpoint: Path):
    if arm == "AF2":
        return make_afab_trainer(frozen, d0_checkpoint=d0_checkpoint)
    return make_igem_trainer(frozen, d0_checkpoint=d0_checkpoint)


def _train_arm(
    arm: str,
    data_root: Path,
    d0_checkpoint: Path,
    output_root: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[dict, bool, dict | None]:
    payload, frozen = _config(arm)
    epochs = int(payload["train"]["epochs"])
    arm_root = output_root / arm
    run_dir = arm_root / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    recovery = _recover_if_corrupt(run_dir)
    training_executed = False

    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = _trainer(arm, frozen, d0_checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root, lock_name=f"{arm}_seed{seed}.training.lock"
        ):
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


def _aggregate(per_seed: dict[str, dict], arm: str) -> dict[str, dict]:
    seeds = (SEED42, *CONFIRMATION_SEEDS)
    result: dict[str, dict] = {}
    for metric in METRICS:
        control = [float(per_seed[str(seed)]["D0FT"][metric]) for seed in seeds]
        candidate = [float(per_seed[str(seed)][arm][metric]) for seed in seeds]
        deltas = [right - left for left, right in zip(control, candidate)]
        result[metric] = {
            "d0ft_mean": statistics.fmean(control),
            "d0ft_std": statistics.stdev(control),
            "candidate_mean": statistics.fmean(candidate),
            "candidate_std": statistics.stdev(candidate),
            "head_delta_mean": statistics.fmean(deltas),
            "head_delta_std": statistics.stdev(deltas),
            "head_delta_min": min(deltas),
            "head_improved_seeds": sum(delta > 0.0 for delta in deltas),
            "deltas": dict(zip((str(seed) for seed in seeds), deltas)),
        }
    return result


def _decision(aggregate: dict[str, dict]) -> tuple[dict[str, bool], str]:
    criteria = {
        "macro_gain_at_least_0_5_point": aggregate["macro_map50_95"]["head_delta_mean"] >= 0.005,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"]["head_improved_seeds"] >= 2,
        "bottom3_mean_not_lower": aggregate["bottom3_class_map50_95"]["head_delta_mean"] >= 0.0,
        "bottom3_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["head_improved_seeds"] >= 2,
        "worst_mean_drop_no_more_than_1_point": aggregate["worst_class_map50_95"]["head_delta_mean"] >= -0.01,
    }
    return criteria, "PASS" if all(criteria.values()) else "FAIL"


def run_faruq_v3_af2_igem_paired_confirmation(
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_seed42_result: str | Path,
    igem_seed42_result: str | Path,
    d0ft_confirmation: str | Path,
    d0_checkpoints: tuple[str | Path, ...],
    output_root: str | Path,
    *,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    models: tuple[str, ...] = ARMS,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    frozen_seeds = tuple(int(seed) for seed in seeds)
    frozen_models = tuple(models)
    if frozen_seeds != CONFIRMATION_SEEDS:
        raise ValueError(f"Konfirmasi dikunci pada seed {CONFIRMATION_SEEDS}")
    if frozen_models != ARMS:
        raise ValueError(f"Konfirmasi dikunci pada model {ARMS}")
    if len(d0_checkpoints) != len(CONFIRMATION_SEEDS):
        raise ValueError("Harus tersedia tepat dua checkpoint D0 seed 123/2026")
    if not authorize_training:
        raise RuntimeError("Konfirmasi paired AF2/IGEM1 belum diotorisasi")

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

    seed42 = _validate_seed42_results(af2_seed42_result, igem_seed42_result)
    d0ft = _validate_d0ft_confirmation(d0ft_confirmation, seed42["D0FT"])
    per_seed: dict[str, dict] = {
        str(SEED42): {
            "D0FT": seed42["D0FT"],
            "AF2": seed42["AF2"],
            "IGEM1": seed42["IGEM1"],
            "sources": seed42["sources"],
        }
    }
    execution: dict[str, dict] = {}

    for seed, checkpoint_value in zip(frozen_seeds, d0_checkpoints):
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"D0 seed {seed} tidak ditemukan: {checkpoint}")
        if _checkpoint_seed(checkpoint) != seed:
            raise RuntimeError(f"Checkpoint D0 tidak cocok dengan seed {seed}")
        per_seed[str(seed)] = {
            "D0FT": _metrics(d0ft["per_seed"][str(seed)]["results"]["D0FT"]),
            "d0_checkpoint_sha256": _sha256(checkpoint),
        }
        execution[str(seed)] = {}
        for arm in frozen_models:
            print(f"\n=== PAIRED CONFIRMATION | {arm} | SEED {seed} ===", flush=True)
            report, trained, recovery = _train_arm(
                arm, data_root, checkpoint, output_root, seed=seed, device=device
            )
            per_seed[str(seed)][arm] = _metrics(report)
            execution[str(seed)][arm] = {
                "training_executed_this_call": trained,
                "recovery": recovery,
                "report": str(reports / f"{arm}_seed{seed}_val.json"),
            }

    aggregates, decisions = {}, {}
    for arm in frozen_models:
        aggregate = _aggregate(per_seed, arm)
        criteria, decision = _decision(aggregate)
        aggregates[arm] = aggregate
        decisions[arm] = {"decision": decision, "criteria": criteria}
    passed = [arm for arm in frozen_models if decisions[arm]["decision"] == "PASS"]
    status = (
        "BOTH_PASS"
        if len(passed) == 2
        else f"{passed[0]}_ONLY_PASS"
        if len(passed) == 1
        else "BOTH_FAIL"
    )
    result = {
        "protocol": "faruq-v3-af2-igem-paired-validation-confirmation-v1",
        "seeds": [SEED42, *CONFIRMATION_SEEDS],
        "models": list(frozen_models),
        "evaluation_split": "val",
        "training_data": "faruq-v3-grouped-development-only",
        "test_images_accessed": False,
        "test_opened": False,
        "per_seed": per_seed,
        "aggregate": aggregates,
        "decisions": decisions,
        "status": status,
        "next_action": "REPORT_VALIDATION_ROBUSTNESS_WITHOUT_FURTHER_TEST_TUNING",
        "execution": execution,
    }
    summary = reports / "af2_igem_paired_confirmation.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired multi-seed AF2 and IGEM1 confirmation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-seed42-result", required=True)
    parser.add_argument("--igem-seed42-result", required=True)
    parser.add_argument("--d0ft-confirmation", required=True)
    parser.add_argument("--d0-checkpoints", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIRMATION_SEEDS))
    parser.add_argument("--models", nargs="+", default=list(ARMS))
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_af2_igem_paired_confirmation(
        args.data_root,
        args.grouped_summary,
        args.af2_seed42_result,
        args.igem_seed42_result,
        args.d0ft_confirmation,
        tuple(args.d0_checkpoints),
        args.output_root,
        seeds=tuple(args.seeds),
        models=tuple(args.models),
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
