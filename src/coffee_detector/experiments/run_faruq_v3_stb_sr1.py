"""Seed-42 screening runner for STB-SR1.

The locked test is intentionally unsupported. Stage A compares one fresh
STB-SR1 run with frozen CMC0/STB1 seed-42 validation references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)
from coffee_detector.stb import STBConfig
from coffee_detector.stb_sr1 import make_stb_sr1_trainer, static_stb_sr1_audit


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/stb_sr1/STBSR1_yolo26n_cmc_spatial_residual.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
REFERENCE = REPO_ROOT / "docs/evidence/FARUQ_V3_STB_CAPACITY_PAIRED_CONFIRMATION_2026-08-14.json"
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def _references() -> tuple[dict[str, float], dict[str, float]]:
    payload = json.loads(REFERENCE.read_text(encoding="utf-8"))
    if payload.get("test_opened") is not False or payload.get("evaluation_split") != "val":
        raise RuntimeError("Frozen STB/CMC evidence violates development-only contract")
    seed42 = payload["per_seed"]["42"]
    return _metrics(seed42["CMC0"]), _metrics(seed42["STB1"])


def _delta(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {name: candidate[name] - reference[name] for name in METRICS}


def _decision(candidate: dict[str, float], cmc0: dict[str, float], stb1: dict[str, float]) -> dict:
    vs_cmc = _delta(candidate, cmc0)
    vs_stb = _delta(candidate, stb1)
    cmc_criteria = {
        "macro_gain_at_least_0_5_point": vs_cmc["macro_map50_95"] >= 0.005,
        "bottom3_not_lower": vs_cmc["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_1_point": vs_cmc["worst_class_map50_95"] >= -0.01,
    }
    stb_retention = {
        "macro_drop_no_more_than_0_5_point": vs_stb["macro_map50_95"] >= -0.005,
        "bottom3_drop_no_more_than_1_point": vs_stb["bottom3_class_map50_95"] >= -0.01,
        "worst_drop_no_more_than_1_point": vs_stb["worst_class_map50_95"] >= -0.01,
    }
    advancement = {
        "macro_plus_0_2": vs_stb["macro_map50_95"] >= 0.002,
        "bottom3_plus_0_5": vs_stb["bottom3_class_map50_95"] >= 0.005,
        "worst_plus_0_5": vs_stb["worst_class_map50_95"] >= 0.005,
    }
    passed = all(cmc_criteria.values()) and all(stb_retention.values()) and any(advancement.values())
    return {
        "delta_vs_CMC0": vs_cmc,
        "delta_vs_STB1": vs_stb,
        "cmc_improvement_criteria": cmc_criteria,
        "stb_retention_criteria": stb_retention,
        "stb_advancement_signals": advancement,
        "decision": "PASS" if passed else "FAIL",
    }


def run_stb_sr1(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("STB-SR1 Stage A dikunci ke seed 42")
    if stage not in {"static", "train"}:
        raise ValueError("stage harus static atau train")

    data_root = Path(data_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)

    if stage == "static":
        return static_stb_sr1_audit(
            MODEL_YAML,
            d0_checkpoint,
            output_root / "static_audit.json",
        )

    if not authorize_training:
        raise RuntimeError("Training STB-SR1 belum diotorisasi")
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    static_path = output_root / "static_audit.json"
    if not static_path.is_file():
        raise RuntimeError("Jalankan stage static sebelum training")
    static = json.loads(static_path.read_text(encoding="utf-8"))
    if static.get("decision") != "PASS" or static.get("test_opened") is not False:
        raise RuntimeError("Static audit belum PASS atau test contract invalid")

    dataset_audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    config = STBConfig.from_mapping(payload["stb"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / f"STBSR1_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False

    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_stb_sr1_trainer(config, d0_checkpoint=d0_checkpoint)
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root,
            lock_name=f"STBSR1_seed{seed}.training.lock",
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME STB-SR1 dari epoch {epoch + 1}/{epochs}", flush=True)
                model = YOLO(str(last))
                args = {"resume": True}
            else:
                print("START STB-SR1 dari native seed-42 D0", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root),
                    name=f"STBSR1_seed{seed}",
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
        raise RuntimeError(f"Run STB-SR1 belum lengkap: {run_dir}")

    report = evaluate(
        best,
        data_root,
        reports / f"STBSR1_seed{seed}_val.json",
        split="val",
        device=device,
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation STB-SR1 kehilangan kelas")

    candidate = _metrics(report)
    cmc0, stb1 = _references()
    gate = _decision(candidate, cmc0, stb1)
    result = {
        "protocol": "faruq-v3-stb-sr1-seed42-screen-v1",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "models": {"CMC0_frozen": cmc0, "STB1_frozen": stb1, "STBSR1": candidate},
        "static_capacity": static.get("models", {}),
        "parameter_overhead_vs_cmc0": static.get("overhead_vs_cmc0"),
        "parameter_overhead_vs_stb1": static.get("overhead_vs_stb1"),
        "comparison": gate,
        "training_executed_this_call": training_executed,
        "checkpoint": str(best),
        "decision": gate["decision"],
        "next_action": (
            "FREEZE_PAIRED_MULTI_SEED_CONFIRMATION_PROTOCOL"
            if gate["decision"] == "PASS"
            else "STOP_STB_SR1_WITHOUT_EXTRA_SEEDS_OR_TEST"
        ),
    }
    summary = reports / "stb_sr1_seed42_decision.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 STB-SR1 seed42 screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("static", "train"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_stb_sr1(
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.output_root,
        stage=args.stage,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
