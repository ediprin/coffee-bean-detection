"""Fail-fast FMH1 focal-modulation classification-head screening."""

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
from coffee_detector.focal_modulation import (
    FocalModulationConfig,
    make_focal_modulation_trainer,
    static_focal_modulation_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/focal_modulation/FMH1_yolo26n_classification.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
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


def _epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return 0
    try:
        return int(float(rows[-1].get("epoch", len(rows)))) + 1
    except (TypeError, ValueError):
        return len(rows)


def _checkpoint_state(path: Path) -> tuple[int | None, bool]:
    if not path.is_file():
        return None, False
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None, False
    epoch = payload.get("epoch")
    return int(epoch) if epoch is not None else None, payload.get("optimizer") is not None


def _run_complete(run_dir: Path, epochs: int) -> bool:
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    if not best.is_file() or not last.is_file():
        return False
    if _epochs(run_dir / "results.csv") >= epochs:
        return True
    epoch, resumable = _checkpoint_state(last)
    return epoch == -1 and not resumable


def _reference(payload: dict, names: tuple[str, ...]) -> dict[str, float]:
    for container in ("candidate", "candidates", "results", "reference"):
        rows = payload.get(container, {})
        for name in names:
            if name in rows:
                return _metrics(rows[name])
    for name in names:
        if name in payload:
            return _metrics(payload[name])
    raise KeyError(f"Tidak menemukan reference {names}")


def _stb_comparison(candidate: dict, reference: dict) -> dict:
    delta = {name: candidate[name] - reference[name] for name in METRICS}
    criteria = {
        "macro_gain_at_least_0_5_point": delta["macro_map50_95"] >= 0.005,
        "bottom3_not_lower": delta["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_1_point": delta["worst_class_map50_95"] >= -0.01,
    }
    return {"deltas": delta, "criteria": criteria, "decision": "PASS" if all(criteria.values()) else "FAIL"}


def _fct0_comparison(candidate: dict, reference: dict) -> dict:
    delta = {name: candidate[name] - reference[name] for name in METRICS}
    criteria = {
        "macro_not_lower_than_optimization_control": delta["macro_map50_95"] >= 0.0,
        "bottom3_not_lower_than_optimization_control": delta["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_1_point": delta["worst_class_map50_95"] >= -0.01,
    }
    return {"deltas": delta, "criteria": criteria, "decision": "PASS" if all(criteria.values()) else "FAIL"}


def run_focal_modulation_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    stb_summary: str | Path,
    fct0_summary: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42 or stage not in {"static", "train"}:
        raise ValueError("FMH1 dikunci seed 42 dan stage static/train")
    data_root = Path(data_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if stage == "static":
        return static_focal_modulation_audit(
            MODEL_YAML, d0_checkpoint, output_root / "static_audit.json"
        )

    if not authorize_training:
        raise RuntimeError("Training FMH1 belum diotorisasi")
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    static = _load_json(output_root / "static_audit.json", "Static audit")
    if static.get("decision") != "PASS" or static.get("d0_checkpoint_sha256") != _sha256(d0_checkpoint):
        raise RuntimeError("Static audit FMH1 belum valid untuk checkpoint D0 ini")
    stb_payload = _load_json(stb_summary, "STB1 summary")
    fct0_payload = _load_json(fct0_summary, "FCT0 summary")
    for label, payload in (("STB1", stb_payload), ("FCT0", fct0_payload)):
        if payload.get("test_opened") is not False and payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"{label} tidak membuktikan test lock")
    references = {
        "STB1": _reference(stb_payload, ("STB1",)),
        "FCT0": _reference(fct0_payload, ("FCT0",)),
    }
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "FMH1":
        raise ValueError("Config FMH1 tidak valid")
    config = FocalModulationConfig.from_mapping(payload["focal_modulation"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / f"FMH1_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_focal_modulation_trainer(config, d0_checkpoint=d0_checkpoint)
        epoch, resumable = _checkpoint_state(last)
        if last.is_file() and resumable and epoch is not None and epoch >= 0:
            print(f"RESUME FMH1 dari artefak Drive ({_epochs(run_dir / 'results.csv')}/{epochs})", flush=True)
            model, args = YOLO(str(last)), {"resume": True}
        else:
            print("START FMH1 dari checkpoint D0", flush=True)
            model = YOLO(str(REPO_ROOT / payload["model"]))
            args = dict(payload["train"])
            args.update(
                data=str(data_root / "data.yaml"), project=str(output_root),
                name=f"FMH1_seed{seed}", exist_ok=True, seed=seed,
                deterministic=True, plots=True, verbose=True,
            )
        if device is not None:
            args["device"] = device
        model.train(trainer=trainer, **args)
        training_executed = True
    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run FMH1 belum lengkap: {run_dir}")
    report = evaluate(
        best, data_root, reports / f"FMH1_seed{seed}_val.json", split="val", device=device
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation FMH1 kehilangan kelas")
    candidate = _metrics(report)
    comparisons = {
        "STB1_vs_FMH1": _stb_comparison(candidate, references["STB1"]),
        "FCT0_vs_FMH1": _fct0_comparison(candidate, references["FCT0"]),
    }
    decision = "PASS" if all(row["decision"] == "PASS" for row in comparisons.values()) else "FAIL"
    result = {
        "protocol": "faruq-v3-fmh1-focal-modulation-screening-v1",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "references": references,
        "candidate": {"FMH1": candidate},
        "comparisons": comparisons,
        "parameters": static["parameters"],
        "config": config.to_dict(),
        "training_executed_this_call": training_executed,
        "decision": decision,
        "next_action": "AUTHORIZE_FMH1_CAPACITY_CONTROL" if decision == "PASS" else "STOP_FMH1_WITHOUT_TEST_OR_EXTRA_SEEDS",
        "checkpoint": str(best),
    }
    summary = reports / "fmh1_seed42_decision.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 FMH1 focal modulation screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--stb-summary", required=True)
    parser.add_argument("--fct0-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("static", "train"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_focal_modulation_screening(
        args.data_root, args.grouped_summary, args.d0_checkpoint, args.stb_summary,
        args.fct0_summary, args.output_root, stage=args.stage, seed=args.seed,
        device=args.device, authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
