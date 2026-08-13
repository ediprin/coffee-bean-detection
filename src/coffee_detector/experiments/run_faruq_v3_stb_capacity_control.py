"""Capacity- and schedule-matched causal control for STB1."""

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
from coffee_detector.stb import STBConfig
from coffee_detector.stb_control import (
    make_stb_control_trainer,
    static_stb_capacity_control_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/stb_control/CMC0_yolo26n_channel_capacity.yaml"
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


def _comparison(stb: dict, control: dict) -> dict:
    delta = {name: stb[name] - control[name] for name in METRICS}
    criteria = {
        "stb_macro_gain_at_least_0_5_point": delta["macro_map50_95"] >= 0.005,
        "stb_bottom3_not_lower": delta["bottom3_class_map50_95"] >= 0.0,
        "stb_worst_drop_no_more_than_1_point": delta["worst_class_map50_95"] >= -0.01,
    }
    return {"deltas": delta, "criteria": criteria, "decision": "PASS" if all(criteria.values()) else "FAIL"}


def _control_validity(control: dict, d0ft: dict) -> dict:
    delta = {name: control[name] - d0ft[name] for name in METRICS}
    criteria = {
        "macro_drop_no_more_than_1_point": delta["macro_map50_95"] >= -0.01,
        "bottom3_drop_no_more_than_2_points": delta["bottom3_class_map50_95"] >= -0.02,
        "worst_drop_no_more_than_3_points": delta["worst_class_map50_95"] >= -0.03,
    }
    return {"deltas": delta, "criteria": criteria, "decision": "PASS" if all(criteria.values()) else "FAIL"}


def run_stb_capacity_control(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    stb_summary: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42 or stage not in {"static", "train"}:
        raise ValueError("STB causal control dikunci seed 42 dan stage static/train")
    data_root = Path(data_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if stage == "static":
        return static_stb_capacity_control_audit(
            MODEL_YAML, d0_checkpoint, output_root / "static_audit.json"
        )
    if not authorize_training:
        raise RuntimeError("Training CMC0 belum diotorisasi")
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    static = _load_json(output_root / "static_audit.json", "Static audit")
    if static.get("decision") != "PASS":
        raise RuntimeError("Static audit belum PASS")
    reference = _load_json(stb_summary, "STB1 summary")
    if reference.get("d0_checkpoint_sha256") != _sha256(d0_checkpoint):
        raise RuntimeError("STB1 dan CMC0 tidak berasal dari checkpoint D0 yang sama")
    if reference.get("seed") != 42 or reference.get("test_opened") is not False:
        raise RuntimeError("Reference STB1 tidak sesuai protokol")
    stb = _metrics(reference["candidate"]["STB1"])
    d0ft = _metrics(reference["controls"]["D0FT"])
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    config = STBConfig.from_mapping(payload["stb"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / f"CMC0_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_stb_control_trainer(config, d0_checkpoint=d0_checkpoint)
        epoch, resumable = _checkpoint_state(last)
        if last.is_file() and resumable and epoch is not None and epoch >= 0:
            print(f"RESUME CMC0 dari epoch {epoch + 1}/{epochs}", flush=True)
            model, args = YOLO(str(last)), {"resume": True}
        else:
            print("START CMC0 dari checkpoint D0 yang sama dengan STB1", flush=True)
            model = YOLO(str(REPO_ROOT / payload["model"]))
            args = dict(payload["train"])
            args.update(
                data=str(data_root / "data.yaml"), project=str(output_root),
                name=f"CMC0_seed{seed}", exist_ok=True, seed=seed,
                deterministic=True, plots=True, verbose=True,
            )
        if device is not None:
            args["device"] = device
        model.train(trainer=trainer, **args)
        training_executed = True
    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run CMC0 belum lengkap: {run_dir}")
    report = evaluate(best, data_root, reports / "CMC0_seed42_val.json", split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation CMC0 kehilangan kelas")
    control = _metrics(report)
    comparisons = {
        "CMC0_validity_vs_D0FT": _control_validity(control, d0ft),
        "CMC0_vs_STB1_causal": _comparison(stb, control),
    }
    decision = "PASS" if all(row["decision"] == "PASS" for row in comparisons.values()) else "FAIL"
    result = {
        "protocol": "faruq-v3-stb-capacity-causal-control-v1",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "models": {"D0FT": d0ft, "CMC0": control, "STB1": stb},
        "comparisons": comparisons,
        "capacity": static["models"],
        "parameter_relative_gap": static["parameter_relative_gap"],
        "training_executed_this_call": training_executed,
        "decision": decision,
        "next_action": "AUTHORIZE_PAIRED_STB_CONTROL_MULTI_SEED" if decision == "PASS" else "STOP_STB_CAUSAL_CLAIM_WITHOUT_TEST",
        "checkpoint": str(best),
    }
    summary = reports / "stb_capacity_control_seed42_decision.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 STB capacity causal control")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--stb-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("static", "train"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_stb_capacity_control(
        args.data_root, args.grouped_summary, args.d0_checkpoint, args.stb_summary,
        args.output_root, stage=args.stage, seed=args.seed, device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
