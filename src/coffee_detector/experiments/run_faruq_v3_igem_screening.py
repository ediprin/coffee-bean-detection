from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.igem import IGEMConfig, make_igem_trainer


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/igem/IGEM1_yolo26n_classification_guidance.yaml"
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def _load_config() -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "IGEM1":
        raise ValueError("Config IGEM1 tidak valid")
    if not {"model", "igem", "train"} <= set(payload):
        raise ValueError("Config IGEM1 tidak lengkap")
    IGEMConfig.from_mapping(payload["igem"])
    return payload


def run_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    control_summary: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("IGEM breadth discovery dikunci seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos split test")
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)

    control = _load_json(control_summary, "D0FT/ACMC1 control summary")
    if control.get("test_images_accessed") is not False or control.get("test_opened") is not False:
        raise RuntimeError("Control summary tidak membuktikan test lock")
    checkpoint_hash = _sha256(d0_checkpoint)
    if control.get("d0_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("D0 checkpoint berbeda dari control summary")

    config_payload = _load_config()
    config = IGEMConfig.from_mapping(config_payload["igem"])
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    from ultralytics import YOLO

    run_dir = output_root / f"IGEM1_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False
    trainer = make_igem_trainer(config, d0_checkpoint=d0_checkpoint)
    if not best.is_file():
        if last.is_file():
            model = YOLO(str(last))
            args = {"resume": True}
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        else:
            model = YOLO(str(REPO_ROOT / config_payload["model"]))
            args = dict(config_payload["train"])
            args.update(
                {
                    "data": str(data_root / "data.yaml"),
                    "project": str(output_root),
                    "name": f"IGEM1_seed{seed}",
                    "exist_ok": True,
                    "seed": seed,
                    "deterministic": True,
                    "plots": True,
                    "verbose": True,
                }
            )
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        training_executed = True

    if not best.is_file():
        raise FileNotFoundError(best)
    report_path = reports / f"IGEM1_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation IGEM1 kehilangan kelas")

    candidate = _metrics(report)
    controls = {
        name: _metrics(control["results"][name])
        for name in ("D0", "D0FT", "ACMC1")
    }
    delta = {name: candidate[name] - controls["D0FT"][name] for name in METRICS}
    criteria = {
        "macro_retention": delta["macro_map50_95"] >= -0.010,
        "bottom3_retention": delta["bottom3_class_map50_95"] >= -0.020,
        "worst_retention": delta["worst_class_map50_95"] >= -0.020,
        "discovery_signal": (
            delta["macro_map50_95"] >= 0.002
            or delta["bottom3_class_map50_95"] >= 0.005
            or delta["worst_class_map50_95"] >= 0.005
        ),
    }
    result = {
        "protocol": "faruq-v3-igem-classification-guidance-screening-v1",
        "stage": "breadth_discovery",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": checkpoint_hash,
        "paper_settings_retained": {
            "reference_depth": 3,
            "mask_loss_weight": 0.05,
            "mask_classes": "N+1 including background",
            "static_dynamic_soft_fusion": True,
        },
        "transfer_choices": {
            "mask_target": "axis-aligned training bbox rectangles rather than aircraft-specific fine cross masks",
            "kernel_size": config.kernel_size,
            "attention_heads": config.attention_heads,
            "channel_reduction": config.channel_reduction,
            "identity_start": "zero-initialized fine-class residual correction",
        },
        "controls": controls,
        "candidate": {"IGEM1": candidate},
        "delta_vs_D0FT": delta,
        "criteria": criteria,
        "decision": "RETAIN" if all(criteria.values()) else "REJECT",
        "training_executed_this_call": training_executed,
        "checkpoint": str(best),
        "report": str(report_path),
    }
    summary = reports / "igem_seed42_screening.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 IGEM breadth screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_screening(
        args.data_root,
        args.grouped_summary,
        args.control_summary,
        args.d0_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
