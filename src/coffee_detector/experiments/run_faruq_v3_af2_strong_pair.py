"""Seed-42 direct pairing of historical AF2 with retained strong YOLO26 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_pairs import make_af2_pair_trainer, run_af2_pair_static_audit
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
ARMS = {
    "AF2STB1": REPO_ROOT / "configs/af2_pairs/AF2STB1.yaml",
    "AF2IGEM1": REPO_ROOT / "configs/af2_pairs/AF2IGEM1.yaml",
    "AF2SAF1": REPO_ROOT / "configs/af2_pairs/AF2SAF1.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def _config(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Arm harus salah satu {tuple(ARMS)}")
    payload = yaml.safe_load(ARMS[arm].read_text(encoding="utf-8")) or {}
    required = {"code", "standalone", "model", "af2", "strong", "train"}
    if payload.get("code") != arm or not required <= set(payload):
        raise RuntimeError(f"Config {arm} tidak lengkap")
    return payload


def _decision(delta: dict[str, float]) -> tuple[dict[str, bool], str]:
    strict = {
        "macro_not_lower": delta["macro_map50_95"] >= 0.0,
        "bottom3_not_lower": delta["bottom3_class_map50_95"] >= 0.0,
        "worst_not_lower": delta["worst_class_map50_95"] >= 0.0,
        "at_least_one_gain_0_2_point": max(delta.values()) >= 0.002,
    }
    pareto = {
        "macro_drop_no_more_than_0_1_point": delta["macro_map50_95"] >= -0.001,
        "bottom3_not_lower": delta["bottom3_class_map50_95"] >= 0.0,
        "worst_not_lower": delta["worst_class_map50_95"] >= 0.0,
        "material_tail_gain": (
            delta["bottom3_class_map50_95"] >= 0.005
            or delta["worst_class_map50_95"] >= 0.010
        ),
    }
    criteria = {f"strict_{key}": value for key, value in strict.items()}
    criteria.update({f"pareto_{key}": value for key, value in pareto.items()})
    if all(strict.values()):
        return criteria, "RETAIN_STRICT_SUPERIOR"
    if all(pareto.values()):
        return criteria, "RETAIN_PARETO"
    return criteria, "REJECT"


def run_faruq_v3_af2_strong_pair(
    arm: str,
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    standalone_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("Screening pasangan langsung dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Tambahkan --authorize-training setelah static gate PASS")
    payload = _config(arm)
    data_root = Path(data_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    standalone_checkpoint = Path(standalone_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    for path in (d0_checkpoint, standalone_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    dataset_audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    model_yaml = REPO_ROOT / payload["model"]
    static_path = output_root / "static_audit.json"
    static = run_af2_pair_static_audit(
        arm, model_yaml, d0_checkpoint, standalone_checkpoint,
        payload["strong"], static_path, image_size=128,
    )
    if static["decision"] != "PASS":
        raise RuntimeError(f"Static gate gagal: {static_path}")

    standalone_report_path = reports / f"{payload['standalone']}_paired_reference_val.json"
    standalone_report = evaluate(
        standalone_checkpoint, data_root, standalone_report_path,
        split="val", device=device,
    )
    if standalone_report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Reference validation kehilangan kelas")

    run_dir = output_root / f"{arm}_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    epochs = int(payload["train"]["epochs"])
    training_executed = False
    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_af2_pair_trainer(
            arm, payload["strong"], af2=payload["af2"], d0_checkpoint=d0_checkpoint,
        )
        checkpoint_epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(output_root, lock_name=f"{arm}_seed{seed}.training.lock"):
            if last.is_file() and resumable and checkpoint_epoch is not None:
                print(f"RESUME {arm}: epoch {checkpoint_epoch + 1}/{epochs}", flush=True)
                model = YOLO(str(last))
                args = {"resume": True}
            else:
                print(f"START {arm}: D0 seed 42 -> {epochs} epoch", flush=True)
                model = YOLO(str(model_yaml))
                args = dict(payload["train"])
                args.update({
                    "data": str(data_root / "data.yaml"),
                    "project": str(output_root),
                    "name": f"{arm}_seed{seed}",
                    "exist_ok": True,
                    "seed": seed,
                    "deterministic": True,
                    "plots": False,
                    "verbose": False,
                    "save": True,
                    # Ultralytics updates last.pt every epoch. Avoid retaining
                    # fifty redundant epoch*.pt files on the shared Drive.
                    "save_period": -1,
                })
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        training_executed = True

    if not _run_complete(run_dir, epochs) or not best.is_file():
        raise RuntimeError(f"Run {arm} belum lengkap; last.pt tetap tersedia di {last}")
    candidate_report_path = reports / f"{arm}_seed{seed}_val.json"
    candidate_report = evaluate(best, data_root, candidate_report_path, split="val", device=device)
    if candidate_report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Candidate validation kehilangan kelas")

    standalone = _metrics(standalone_report)
    candidate = _metrics(candidate_report)
    delta = {metric: candidate[metric] - standalone[metric] for metric in METRICS}
    criteria, decision = _decision(delta)
    result = {
        "format": "coffee_detector.af2_pairs.seed42_screening.v1",
        "protocol": "faruq-v3-direct-af2-strong-pairs-v1",
        "arm": arm,
        "standalone": payload["standalone"],
        "seed": seed,
        "evaluation_split": "val",
        "values": {payload["standalone"]: standalone, arm: candidate},
        "delta_pair_minus_standalone": delta,
        "criteria": criteria,
        "decision": decision,
        "next": "COMPARE_THREE_AF2_PAIRS" if decision.startswith("RETAIN") else "STOP_THIS_PAIR",
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "standalone_checkpoint_sha256": _sha256(standalone_checkpoint),
        "candidate_checkpoint": str(best),
        "static_audit": str(static_path),
        "training_executed_this_call": training_executed,
        "test_images_accessed": False,
        "test_opened": False,
    }
    summary = reports / f"{arm}_seed42_decision.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct AF2 + retained strong-model screening")
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--standalone-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_af2_strong_pair(
        args.arm, args.data_root, args.grouped_summary, args.d0_checkpoint,
        args.standalone_checkpoint, args.output_root, seed=args.seed,
        device=args.device, authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
