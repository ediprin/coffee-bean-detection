"""Frozen three-seed classical-enhancement control for AF2.

Scientific question:
    Is AF2's validation gain more than a generic local-contrast enhancement effect?

The CLAHE arm is fixed before results are observed: RGB -> LAB, CLAHE on L only,
clipLimit=3.0, tileGridSize=8x8.  Every CLAHE run starts from the seed-matched
D0 checkpoint and uses the same 50-epoch schedule as AF2.  Existing AF2 and
D0FT validation results are read only as frozen references; test data is forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.classical_enhancement import CLAHEConfig, make_clahe_trainer
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
CONFIG = REPO_ROOT / "configs/classical/CLAHE_LAB_yolo26n.yaml"
SEEDS = (42, 123, 2026)
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


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


def _validate_reference(path: str | Path) -> dict:
    payload = _load_json(path, "AF2 paired confirmation")
    if (
        payload.get("protocol") != "faruq-v3-af2-igem-paired-validation-confirmation-v1"
        or payload.get("seeds") != list(SEEDS)
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
        or payload.get("decisions", {}).get("AF2", {}).get("decision") != "PASS"
    ):
        raise RuntimeError("Reference AF2 bukan frozen three-seed PASS validation-only yang kompatibel")
    for seed in SEEDS:
        row = payload["per_seed"][str(seed)]
        _metrics(row["D0FT"])
        _metrics(row["AF2"])
    return payload


def _config() -> tuple[dict, CLAHEConfig]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "CLAHE_LAB":
        raise RuntimeError("Config CLAHE_LAB tidak konsisten")
    frozen = CLAHEConfig.from_mapping(payload["clahe"])
    if frozen.clip_limit != 3.0 or frozen.tile_grid_size != (8, 8):
        raise RuntimeError("Primary CLAHE control harus tetap clipLimit=3.0 dan tileGrid=8x8")
    return payload, frozen


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


def _train_clahe(
    data_root: Path,
    d0_checkpoint: Path,
    output_root: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[dict, bool, dict | None]:
    payload, frozen = _config()
    epochs = int(payload["train"]["epochs"])
    arm_root = output_root / "CLAHE_LAB"
    run_dir = arm_root / f"CLAHE_LAB_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    recovery = _recover_if_corrupt(run_dir)
    training_executed = False

    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_clahe_trainer(frozen, d0_checkpoint=d0_checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root, lock_name=f"CLAHE_LAB_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME CLAHE seed {seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True}
            else:
                print(f"START CLAHE seed {seed} dari D0 seed yang sama", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(arm_root),
                    name=f"CLAHE_LAB_seed{seed}",
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
        raise RuntimeError(f"Run CLAHE seed {seed} belum lengkap: {run_dir}")

    report_path = output_root / "val_reports" / f"CLAHE_LAB_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    missing = report["metrics"].get("classes_without_ground_truth", [])
    if missing:
        raise RuntimeError(f"Validation CLAHE seed {seed} kehilangan kelas: {missing}")
    return report, training_executed, recovery


def _paired_summary(per_seed: dict[str, dict], left: str, right: str) -> dict[str, dict]:
    """Return right-minus-left paired summaries."""

    result: dict[str, dict] = {}
    for metric in METRICS:
        left_values = [float(per_seed[str(seed)][left][metric]) for seed in SEEDS]
        right_values = [float(per_seed[str(seed)][right][metric]) for seed in SEEDS]
        deltas = [right_value - left_value for left_value, right_value in zip(left_values, right_values)]
        result[metric] = {
            "left_mean": statistics.fmean(left_values),
            "left_std": statistics.stdev(left_values),
            "right_mean": statistics.fmean(right_values),
            "right_std": statistics.stdev(right_values),
            "delta_mean": statistics.fmean(deltas),
            "delta_std": statistics.stdev(deltas),
            "delta_min": min(deltas),
            "right_wins": sum(delta > 0.0 for delta in deltas),
            "deltas": dict(zip((str(seed) for seed in SEEDS), deltas)),
        }
    return result


def _generic_enhancement_decision(clahe_vs_d0ft: dict[str, dict]) -> tuple[dict[str, bool], str]:
    criteria = {
        "macro_gain_at_least_0_5_point": clahe_vs_d0ft["macro_map50_95"]["delta_mean"] >= 0.005,
        "macro_improved_at_least_2_of_3": clahe_vs_d0ft["macro_map50_95"]["right_wins"] >= 2,
        "bottom3_mean_not_lower": clahe_vs_d0ft["bottom3_class_map50_95"]["delta_mean"] >= 0.0,
        "worst_mean_drop_no_more_than_1_point": clahe_vs_d0ft["worst_class_map50_95"]["delta_mean"] >= -0.01,
    }
    return criteria, "PASS" if all(criteria.values()) else "FAIL"


def _af2_specific_decision(af2_vs_clahe: dict[str, dict]) -> tuple[dict[str, bool], str]:
    criteria = {
        "af2_macro_advantage_at_least_0_5_point": af2_vs_clahe["macro_map50_95"]["delta_mean"] >= 0.005,
        "af2_macro_wins_at_least_2_of_3": af2_vs_clahe["macro_map50_95"]["right_wins"] >= 2,
        "af2_bottom3_mean_not_lower": af2_vs_clahe["bottom3_class_map50_95"]["delta_mean"] >= 0.0,
        "af2_worst_mean_not_lower": af2_vs_clahe["worst_class_map50_95"]["delta_mean"] >= 0.0,
    }
    return criteria, "PASS" if all(criteria.values()) else "FAIL"


def _clahe_superiority_decision(clahe_vs_af2: dict[str, dict]) -> tuple[dict[str, bool], str]:
    criteria = {
        "clahe_macro_advantage_at_least_0_5_point": clahe_vs_af2["macro_map50_95"]["delta_mean"] >= 0.005,
        "clahe_macro_wins_at_least_2_of_3": clahe_vs_af2["macro_map50_95"]["right_wins"] >= 2,
        "clahe_bottom3_mean_not_lower": clahe_vs_af2["bottom3_class_map50_95"]["delta_mean"] >= 0.0,
        "clahe_worst_mean_not_lower": clahe_vs_af2["worst_class_map50_95"]["delta_mean"] >= 0.0,
    }
    return criteria, "PASS" if all(criteria.values()) else "FAIL"


def run_faruq_v3_af2_clahe_control(
    data_root: str | Path,
    grouped_summary: str | Path,
    af2_confirmation: str | Path,
    d0_checkpoints: tuple[str | Path, ...],
    output_root: str | Path,
    *,
    seeds: tuple[int, ...] = SEEDS,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if tuple(int(seed) for seed in seeds) != SEEDS:
        raise ValueError(f"Control dikunci pada seed {SEEDS}")
    if len(d0_checkpoints) != len(SEEDS):
        raise ValueError("Harus tersedia tepat tiga checkpoint D0 seed 42/123/2026")
    if not authorize_training:
        raise RuntimeError("AF2-vs-CLAHE control belum diotorisasi")

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

    reference = _validate_reference(af2_confirmation)
    per_seed: dict[str, dict] = {}
    execution: dict[str, dict] = {}

    for seed, checkpoint_value in zip(SEEDS, d0_checkpoints):
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"D0 seed {seed} tidak ditemukan: {checkpoint}")
        if _checkpoint_seed(checkpoint) != seed:
            raise RuntimeError(f"Checkpoint D0 tidak cocok dengan seed {seed}")
        frozen_row = reference["per_seed"][str(seed)]
        per_seed[str(seed)] = {
            "D0FT": _metrics(frozen_row["D0FT"]),
            "AF2": _metrics(frozen_row["AF2"]),
            "d0_checkpoint_sha256": _sha256(checkpoint),
        }
        print(f"\n=== AF2-vs-CLAHE CONTROL | CLAHE_LAB | SEED {seed} ===", flush=True)
        report, trained, recovery = _train_clahe(
            data_root, checkpoint, output_root, seed=seed, device=device
        )
        per_seed[str(seed)]["CLAHE_LAB"] = _metrics(report)
        execution[str(seed)] = {
            "training_executed_this_call": trained,
            "recovery": recovery,
            "report": str(reports / f"CLAHE_LAB_seed{seed}_val.json"),
        }

    clahe_vs_d0ft = _paired_summary(per_seed, "D0FT", "CLAHE_LAB")
    af2_vs_clahe = _paired_summary(per_seed, "CLAHE_LAB", "AF2")
    clahe_vs_af2 = _paired_summary(per_seed, "AF2", "CLAHE_LAB")

    generic_criteria, generic_decision = _generic_enhancement_decision(clahe_vs_d0ft)
    af2_criteria, af2_decision = _af2_specific_decision(af2_vs_clahe)
    clahe_criteria, clahe_decision = _clahe_superiority_decision(clahe_vs_af2)

    if af2_decision == "PASS":
        interpretation = "AF2_SPECIFIC_ADVANTAGE_SUPPORTED"
    elif clahe_decision == "PASS":
        interpretation = "CLAHE_SUPERIOR_UNDER_FROZEN_CONTROL"
    else:
        interpretation = "NO_DIRECTIONAL_SUPERIORITY_ESTABLISHED"

    result = {
        "protocol": "faruq-v3-af2-vs-clahe-classical-enhancement-control-v1",
        "seeds": list(SEEDS),
        "models": ["D0FT", "CLAHE_LAB", "AF2"],
        "clahe": {
            "color_space": "LAB",
            "channel": "L",
            "clip_limit": 3.0,
            "tile_grid_size": [8, 8],
            "source": "Guruprakash et al. 2026 DeeppestNet",
        },
        "evaluation_split": "val",
        "training_data": "faruq-v3-grouped-development-only",
        "test_images_accessed": False,
        "test_opened": False,
        "per_seed": per_seed,
        "comparisons": {
            "CLAHE_minus_D0FT": clahe_vs_d0ft,
            "AF2_minus_CLAHE": af2_vs_clahe,
            "CLAHE_minus_AF2": clahe_vs_af2,
        },
        "decisions": {
            "generic_clahe_effect": {"decision": generic_decision, "criteria": generic_criteria},
            "af2_beyond_clahe": {"decision": af2_decision, "criteria": af2_criteria},
            "clahe_superior_to_af2": {"decision": clahe_decision, "criteria": clahe_criteria},
        },
        "interpretation": interpretation,
        "claim_boundary": (
            "Validation-only three-seed control. A PASS for AF2 beyond CLAHE supports an advantage "
            "over this frozen classical local-contrast baseline, not universal superiority over all enhancement methods."
        ),
        "execution": execution,
    }
    summary = reports / "af2_vs_clahe_classical_enhancement_control.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen AF2-vs-CLAHE three-seed validation control")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--af2-confirmation", required=True)
    parser.add_argument("--d0-checkpoints", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_af2_clahe_control(
        args.data_root,
        args.grouped_summary,
        args.af2_confirmation,
        tuple(args.d0_checkpoints),
        args.output_root,
        seeds=tuple(args.seeds),
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
