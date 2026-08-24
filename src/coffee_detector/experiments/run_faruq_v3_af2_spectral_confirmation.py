from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from coffee_detector.af2_spectral import SpectralFrontendConfig, make_spectral_trainer
from coffee_detector.af2_spectral.audit import sha256
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_af2_spectral_arm import CONFIGS
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
SEEDS = (42, 123, 2026)
EPS = 1.0e-12


def _read(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _checkpoint_seed(path: Path) -> int:
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(args, dict) or "seed" not in args:
        raise RuntimeError(f"Checkpoint tidak merekam seed: {path}")
    return int(args["seed"])


def run_confirmation_arm(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    global_decision: str | Path,
    output_root: str | Path,
    *,
    seed: int,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed not in {123, 2026} or not authorize_training:
        raise ValueError("Confirmation arm memerlukan otorisasi dan seed 123/2026")
    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    load_faruq_grouped_summary(grouped_summary, data_root)
    if _checkpoint_seed(checkpoint) != seed:
        raise RuntimeError(f"Checkpoint D0 tidak cocok dengan seed {seed}")
    decision = _read(global_decision)
    if (
        decision.get("stage") != "global"
        or decision.get("decision") != "PASS"
        or decision.get("next") != "AUTHORIZE_WINNER_PAIRED_CONFIRMATION"
        or decision.get("test_opened") is not False
    ):
        raise RuntimeError("Global decision belum mengotorisasi confirmation")
    arm = decision["winner"]
    config_path = CONFIGS[arm]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    spectral = SpectralFrontendConfig.from_mapping(payload["spectral"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / "confirmation" / arm / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_spectral_trainer(spectral, d0_checkpoint=checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root, lock_name=f"confirmation_{arm}_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {arm} seed {seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                print(f"START {arm} seed {seed} dari D0 seed {seed}", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root / "confirmation" / arm),
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
    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Confirmation belum lengkap: {run_dir}")
    report_path = output_root / "val_reports" / f"confirmation_{arm}_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.af2_spectral.confirmation_arm.v1",
        "arm": arm,
        "seed": seed,
        "metrics": report["metrics"],
        "checkpoint": str(best),
        "checkpoint_sha256": sha256(best),
        "initial_d0_checkpoint_sha256": sha256(checkpoint),
        "training_executed": training_executed,
        "test_images_accessed": False,
    }
    result_path = output_root / "val_reports" / f"confirmation_{arm}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {metric: float(source[metric]) for metric in METRICS}


def run_confirmation_decision(
    output_root: str | Path,
    global_decision: str | Path,
    af2_confirmation: str | Path,
) -> dict:
    output_root = Path(output_root).expanduser().resolve()
    global_payload = _read(global_decision)
    if global_payload.get("decision") != "PASS" or global_payload.get("test_opened") is not False:
        raise RuntimeError("Global seed-42 decision tidak kompatibel")
    arm = global_payload["winner"]
    seed42 = _read(output_root / "val_reports" / f"{arm}_seed42_result.json")
    candidate = {42: _metrics(seed42)}
    for seed in (123, 2026):
        result = _read(output_root / "val_reports" / f"confirmation_{arm}_seed{seed}_result.json")
        if result.get("arm") != arm or result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Confirmation {seed} tidak kompatibel")
        candidate[seed] = _metrics(result)
    af2_payload = _read(af2_confirmation)
    if af2_payload.get("test_images_accessed") is not False or af2_payload.get("seeds") != [42, 123, 2026]:
        raise RuntimeError("Evidence AF2 three-seed tidak kompatibel")
    af2 = {
        seed: _metrics(af2_payload["per_seed"][str(seed)]["AF2"])
        for seed in SEEDS
    }
    aggregate = {}
    for metric in METRICS:
        left = [af2[seed][metric] for seed in SEEDS]
        right = [candidate[seed][metric] for seed in SEEDS]
        deltas = [b - a for a, b in zip(left, right)]
        aggregate[metric] = {
            "af2_mean": statistics.fmean(left),
            "af2_std": statistics.stdev(left),
            "candidate_mean": statistics.fmean(right),
            "candidate_std": statistics.stdev(right),
            "delta_mean": statistics.fmean(deltas),
            "delta_std": statistics.stdev(deltas),
            "delta_min": min(deltas),
            "improved_seeds": sum(delta > 0 for delta in deltas),
            "deltas": dict(zip((str(seed) for seed in SEEDS), deltas)),
        }
    criteria = {
        "macro_gain_at_least_0_5_point": aggregate["macro_map50_95"]["delta_mean"] >= 0.005 - EPS,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_not_lower": aggregate["bottom3_class_map50_95"]["delta_mean"] >= -EPS,
        "bottom3_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["improved_seeds"] >= 2,
        "worst_mean_drop_no_more_than_1_point": aggregate["worst_class_map50_95"]["delta_mean"] >= -0.01 - EPS,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    result = {
        "format": "coffee_detector.af2_spectral.paired_confirmation.v1",
        "arm": arm,
        "seeds": list(SEEDS),
        "per_seed": {
            str(seed): {"AF2C": af2[seed], arm: candidate[seed]} for seed in SEEDS
        },
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "next": "AUTHORIZE_POSTHOC_EXTERNAL_EVALUATION" if decision == "PASS" else "KEEP_AF2C_AND_STOP",
        "test_opened": False,
    }
    path = output_root / "val_reports/af2_spectral_paired_confirmation.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2 spectral paired confirmation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    arm = subparsers.add_parser("arm")
    arm.add_argument("--data-root", required=True)
    arm.add_argument("--grouped-summary", required=True)
    arm.add_argument("--d0-checkpoint", required=True)
    arm.add_argument("--global-decision", required=True)
    arm.add_argument("--output-root", required=True)
    arm.add_argument("--seed", type=int, choices=(123, 2026), required=True)
    arm.add_argument("--device", default="0")
    arm.add_argument("--authorize-training", action="store_true")
    decision = subparsers.add_parser("decision")
    decision.add_argument("--output-root", required=True)
    decision.add_argument("--global-decision", required=True)
    decision.add_argument("--af2-confirmation", required=True)
    args = parser.parse_args()
    if args.command == "arm":
        run_confirmation_arm(
            args.data_root,
            args.grouped_summary,
            args.d0_checkpoint,
            args.global_decision,
            args.output_root,
            seed=args.seed,
            device=args.device,
            authorize_training=args.authorize_training,
        )
    else:
        run_confirmation_decision(args.output_root, args.global_decision, args.af2_confirmation)


if __name__ == "__main__":
    main()
