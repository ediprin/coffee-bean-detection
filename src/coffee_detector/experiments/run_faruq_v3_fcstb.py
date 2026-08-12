"""Fail-fast FC-STB teacher-headroom and distillation screening."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.fcstb import (
    FCSTBConfig,
    audit_fcstb_checkpoint_invariance,
    make_fcstb_trainer,
    run_frequency_teacher_headroom,
    static_fcstb_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "FCT0": REPO_ROOT / "configs/fcstb/FCT0_stb_joint_control.yaml",
    "FCD1": REPO_ROOT / "configs/fcstb/FCD1_stb_af2_gt_bounded_distillation.yaml",
}
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


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


def _checkpoint_state(path: Path) -> tuple[int | None, bool]:
    if not path.is_file():
        return None, False
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None, False
    epoch = payload.get("epoch")
    return int(epoch) if epoch is not None else None, payload.get("optimizer") is not None


def _result_epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return 0
    try:
        return int(float(rows[-1].get("epoch", len(rows))))
    except (TypeError, ValueError):
        return len(rows)


def _run_complete(run_dir: Path, epochs: int) -> bool:
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    if not best.is_file() or not last.is_file():
        return False
    if _result_epochs(run_dir / "results.csv") >= int(epochs):
        return True
    epoch, resumable = _checkpoint_state(last)
    return epoch == -1 and not resumable


def _train_arm(
    arm: str,
    data_root: Path,
    output_root: Path,
    stb_checkpoint: Path,
    af2_checkpoint: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[Path, bool, dict]:
    from ultralytics import YOLO

    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    config_payload = dict(payload["fcstb"])
    if arm == "FCD1":
        config_payload["teacher_checkpoint"] = str(af2_checkpoint)
    config = FCSTBConfig.from_mapping(config_payload)
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    trainer = make_fcstb_trainer(
        config, stb=payload["stb"], source_checkpoint=stb_checkpoint
    )
    executed = False
    if not _run_complete(run_dir, epochs):
        epoch, resumable = _checkpoint_state(last)
        if last.is_file() and resumable and epoch is not None and epoch >= 0:
            print(f"RESUME {arm} dari epoch {epoch + 1}/{epochs}", flush=True)
            model = YOLO(str(last))
            args: dict = {"resume": True}
        else:
            print(f"START {arm} dari STB1 yang sama", flush=True)
            model = YOLO(str(REPO_ROOT / payload["model"]))
            args = dict(payload["train"])
            args.update(
                data=str(data_root / "data.yaml"),
                project=str(output_root),
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
        executed = True
    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run {arm} belum lengkap: {run_dir}")
    return best, executed, config.to_dict()


def _comparison(candidate: dict, reference: dict) -> dict:
    delta = {name: candidate[name] - reference[name] for name in METRICS}
    criteria = {
        "macro_gain_at_least_0_5_point": delta["macro_map50_95"] >= 0.005,
        "bottom3_not_lower": delta["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_1_point": delta["worst_class_map50_95"] >= -0.01,
    }
    return {"deltas": delta, "criteria": criteria, "decision": "PASS" if all(criteria.values()) else "FAIL"}


def run_fcstb(
    data_root: str | Path,
    grouped_summary: str | Path,
    stb_summary: str | Path,
    stb_checkpoint: str | Path,
    af2_checkpoint: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42 or stage not in {"static", "diagnostic", "train"}:
        raise ValueError("FC-STB dikunci seed 42 dan stage static/diagnostic/train")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    stb_checkpoint = Path(stb_checkpoint).expanduser().resolve()
    af2_checkpoint = Path(af2_checkpoint).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if stage == "static":
        return static_fcstb_audit(
            REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml",
            stb_checkpoint,
            af2_checkpoint,
            output_root / "static_audit.json",
        )

    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    static = _load_json(output_root / "static_audit.json", "Static audit")
    if (
        static.get("decision") != "PASS"
        or static.get("stb_sha256") != _sha256(stb_checkpoint)
        or static.get("af2_sha256") != _sha256(af2_checkpoint)
    ):
        raise RuntimeError("Static audit FC-STB belum valid")
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    if stage == "diagnostic":
        return run_frequency_teacher_headroom(
            stb_checkpoint,
            af2_checkpoint,
            data_root,
            reports / "frequency_teacher_headroom_seed42.json",
            split="val",
            device=device or "cpu",
        )

    if not authorize_training:
        raise RuntimeError("Training FC-STB belum diotorisasi")
    diagnostic = _load_json(
        reports / "frequency_teacher_headroom_seed42.json", "Teacher headroom"
    )
    if diagnostic.get("decision") != "PASS" or diagnostic.get("test_images_accessed") is not False:
        raise RuntimeError("Teacher AF2 tidak lolos headroom gate")
    reference_payload = _load_json(stb_summary, "STB1 summary")
    if reference_payload.get("decision") != "RETAIN" or reference_payload.get("test_opened") is not False:
        raise RuntimeError("STB1 summary tidak sesuai protokol")
    reference = _metrics(reference_payload["candidate"]["STB1"])
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    candidates, executed, configs, invariance = {}, {}, {}, {}
    for arm in ("FCT0", "FCD1"):
        checkpoint, ran, config = _train_arm(
            arm,
            data_root,
            output_root,
            stb_checkpoint,
            af2_checkpoint,
            seed=seed,
            device=device,
        )
        report = evaluate(
            checkpoint,
            data_root,
            reports / f"{arm}_seed{seed}_val.json",
            split="val",
            device=device,
        )
        if report["metrics"].get("classes_without_ground_truth", []):
            raise RuntimeError(f"Validation {arm} kehilangan kelas")
        candidates[arm] = _metrics(report)
        executed[arm], configs[arm] = ran, config
        invariance[arm] = audit_fcstb_checkpoint_invariance(stb_checkpoint, checkpoint)
        if invariance[arm]["decision"] != "PASS":
            raise RuntimeError(f"Checkpoint invariance gagal: {arm}")

    comparisons = {
        "STB1_vs_FCD1": _comparison(candidates["FCD1"], reference),
        "FCT0_vs_FCD1": _comparison(candidates["FCD1"], candidates["FCT0"]),
    }
    decision = "PASS" if all(row["decision"] == "PASS" for row in comparisons.values()) else "FAIL"
    payload = {
        "protocol": "faruq-v3-fcstb-frequency-distillation-v1",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "reference": {"STB1": reference},
        "candidates": candidates,
        "comparisons": comparisons,
        "teacher_headroom": diagnostic,
        "checkpoint_invariance": invariance,
        "configs": configs,
        "training_executed_this_call": executed,
        "decision": decision,
        "next_action": "AUTHORIZE_FCD1_MULTI_SEED_CONFIRMATION"
        if decision == "PASS"
        else "STOP_FCSTB_WITHOUT_TEST_OR_EXTRA_SEEDS",
    }
    summary = reports / "fcstb_seed42_decision.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 FC-STB screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--stb-summary", required=True)
    parser.add_argument("--stb-checkpoint", required=True)
    parser.add_argument("--af2-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("static", "diagnostic", "train"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_fcstb(
        args.data_root,
        args.grouped_summary,
        args.stb_summary,
        args.stb_checkpoint,
        args.af2_checkpoint,
        args.output_root,
        stage=args.stage,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
