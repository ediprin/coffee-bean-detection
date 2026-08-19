"""Frozen seed-42 S2 runner for STB1-guided WAV-L1 YOLO26.

Only S2/crosskd is executable here. S3 AF2 robustness training is deliberately
not exposed until S2 passes a separately frozen multiseed confirmation.
"""

from __future__ import annotations

import argparse
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
from coffee_detector.stb_guided import (
    STBGuidedConfig,
    audit_stb_guided_checkpoint,
    make_stb_guided_trainer,
    sha256,
    static_stb_guided_audit,
)
from coffee_detector.wav1_factorization import WAV1FactorizationConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/stb_guided/S2_WAVL1_STB1_CROSSKD.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
WAV_L1_REFERENCE = REPO_ROOT / "docs/evidence/FARUQ_V3_WAV_L1_SEED42_RESULT_2026-08-19.json"
STB1_REFERENCE = REPO_ROOT / "docs/evidence/FARUQ_V3_STB_CAPACITY_PAIRED_CONFIRMATION_2026-08-14.json"
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def _references() -> tuple[dict[str, float], dict[str, float]]:
    wav = json.loads(WAV_L1_REFERENCE.read_text(encoding="utf-8"))
    if wav.get("evaluation_split") != "val" or wav.get("test_images_accessed") is not False:
        raise RuntimeError("WAV-L1 reference violates development-only contract")
    stb = json.loads(STB1_REFERENCE.read_text(encoding="utf-8"))
    if stb.get("evaluation_split") != "val" or stb.get("test_opened") is not False:
        raise RuntimeError("STB1 reference violates development-only contract")
    return _metrics(wav), _metrics(stb["per_seed"]["42"]["STB1"])


def _delta(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {name: candidate[name] - reference[name] for name in METRICS}


def _decision(candidate: dict[str, float], wav_l1: dict[str, float]) -> dict:
    delta = _delta(candidate, wav_l1)
    retention = {
        "macro_drop_no_more_than_0_5_point": delta["macro_map50_95"] >= -0.005,
        "bottom3_drop_no_more_than_1_point": delta["bottom3_class_map50_95"] >= -0.010,
        "worst_drop_no_more_than_1_point": delta["worst_class_map50_95"] >= -0.010,
    }
    advancement = {
        "macro_plus_0_2": delta["macro_map50_95"] >= 0.002,
        "bottom3_plus_0_5": delta["bottom3_class_map50_95"] >= 0.005,
        "worst_plus_0_5": delta["worst_class_map50_95"] >= 0.005,
    }
    passed = all(retention.values()) and any(advancement.values())
    return {
        "delta_vs_WAV_L1": delta,
        "retention": retention,
        "advancement": advancement,
        "decision": "PASS" if passed else "FAIL",
    }


def run_stb_guided(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    teacher_checkpoint: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("S2 Stage A dikunci ke seed 42")
    if stage not in {"static", "train"}:
        raise ValueError("stage harus static atau train")

    data_root = Path(data_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    teacher_checkpoint = Path(teacher_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)

    if stage == "static":
        return static_stb_guided_audit(
            MODEL_YAML,
            d0_checkpoint,
            teacher_checkpoint,
            output_root / "static_audit.json",
        )

    if not authorize_training:
        raise RuntimeError("S2 training belum diotorisasi")
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")

    static_path = output_root / "static_audit.json"
    if not static_path.is_file():
        raise RuntimeError("Jalankan stage static sebelum S2 training")
    static = json.loads(static_path.read_text(encoding="utf-8"))
    if static.get("decision") != "PASS" or static.get("test_opened") is not False:
        raise RuntimeError("Static audit belum PASS atau test contract invalid")
    if static.get("d0_checkpoint_sha256") != sha256(d0_checkpoint):
        raise RuntimeError("D0 berubah setelah static audit")
    if static.get("teacher_checkpoint_sha256") != sha256(teacher_checkpoint):
        raise RuntimeError("STB1 teacher berubah setelah static audit")

    dataset_audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    factorization = WAV1FactorizationConfig.from_mapping(payload["factorization"])
    guided_payload = dict(payload["stb_guided"])
    guided_payload["teacher_checkpoint"] = str(teacher_checkpoint)
    guided = STBGuidedConfig.from_mapping(guided_payload)
    if guided.mode != "crosskd":
        raise RuntimeError("Runner Stage A hanya mengizinkan crosskd; S3 tetap diblok")

    epochs = int(payload["train"]["epochs"])
    run_name = f"S2_STB_CROSSKD_seed{seed}"
    run_dir = output_root / run_name
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False

    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_stb_guided_trainer(
            factorization,
            guided,
            d0_checkpoint=d0_checkpoint,
        )
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root,
            lock_name=f"{run_name}.training.lock",
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME S2 dari epoch {epoch + 1}/{epochs}", flush=True)
                model = YOLO(str(last))
                args = {"resume": True}
            else:
                print("START S2 dari native seed-42 D0; STB1 teacher frozen", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root),
                    name=run_name,
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
        raise RuntimeError(f"Run S2 belum lengkap: {run_dir}")

    expected_parameters = int(static["parameters"]["wav_l1_reference"])
    checkpoint_audit = audit_stb_guided_checkpoint(
        best,
        expected_parameters=expected_parameters,
        output_path=reports / f"{run_name}_checkpoint_audit.json",
    )
    if checkpoint_audit["decision"] != "PASS":
        raise RuntimeError(
            "Checkpoint S2 melanggar deployment contract; hentikan sebelum evaluasi"
        )

    report = evaluate(
        best,
        data_root,
        reports / f"{run_name}_val.json",
        split="val",
        device=device,
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation S2 kehilangan kelas")

    candidate = _metrics(report)
    wav_l1, stb1 = _references()
    gate = _decision(candidate, wav_l1)
    result = {
        "format": "coffee_detector.stb_guided.seed42_decision.v1",
        "protocol": "faruq-v3-stb-guided-wavl1-seed42-v1",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": sha256(d0_checkpoint),
        "teacher_checkpoint_sha256": sha256(teacher_checkpoint),
        "student_deployment": "WAV_L1 + native YOLO26n; no STB1/AF2 inference dependency",
        "models": {
            "WAV_L1_frozen_reference": wav_l1,
            "STB1_frozen_teacher_reference": stb1,
            "S2_STB_CROSSKD": candidate,
        },
        "comparison": {
            **gate,
            "descriptive_delta_vs_STB1_teacher": _delta(candidate, stb1),
        },
        "static_parameters": static.get("parameters", {}),
        "checkpoint_audit": checkpoint_audit,
        "training_executed_this_call": training_executed,
        "checkpoint": str(best),
        "decision": gate["decision"],
        "next_action": (
            "FREEZE_S2_PAIRED_MULTI_SEED_CONFIRMATION_PROTOCOL"
            if gate["decision"] == "PASS"
            else "STOP_S2_CROSSKD_WITHOUT_TUNING_EXTRA_SEEDS_OR_TEST"
        ),
        "s3_training_authorized": False,
    }
    summary = reports / "stb_guided_s2_seed42_decision.json"
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 STB-guided WAV-L1 S2 seed42")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("static", "train"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_stb_guided(
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.teacher_checkpoint,
        args.output_root,
        stage=args.stage,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
