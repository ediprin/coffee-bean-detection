from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.cgfi import CGFIConfig, make_cgfi_trainer
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/cgfi/CG1_yolo26n_frequency_classification.yaml"
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
        raise ValueError("CG1 discovery dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("CG1 training belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh memiliki split test")
    control = _load_json(control_summary, "D0FT/ACMC1 control summary")
    if control.get("test_images_accessed") is not False or control.get("test_opened") is not False:
        raise RuntimeError("Control summary tidak membuktikan test lock")
    if control.get("d0_checkpoint_sha256") != _sha256(d0_checkpoint):
        raise RuntimeError("Checkpoint D0 berbeda dari control summary")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    config = CGFIConfig.from_mapping(payload["cgfi"])
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    from ultralytics import YOLO
    run_dir = output_root / f"CG1_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    trainer = make_cgfi_trainer(config, d0_checkpoint=d0_checkpoint)
    if not best.is_file():
        if last.is_file():
            model = YOLO(str(last))
            args = {"resume": True}
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        else:
            model = YOLO(str(REPO_ROOT / payload["model"]))
            args = dict(payload["train"])
            args.update(
                {
                    "data": str(data_root / "data.yaml"),
                    "project": str(output_root),
                    "name": f"CG1_seed{seed}",
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

    report_path = reports / f"CG1_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    candidate = _metrics(report)
    controls = {name: _metrics(control["results"][name]) for name in ("D0", "D0FT", "ACMC1")}
    delta = {name: candidate[name] - controls["D0FT"][name] for name in METRICS}
    criteria = {
        "macro_not_below_d0ft_by_more_than_0_5_point": delta["macro_map50_95"] >= -0.005,
        "bottom3_not_below_d0ft_by_more_than_2_points": delta["bottom3_class_map50_95"] >= -0.02,
        "worst_not_below_d0ft_by_more_than_3_points": delta["worst_class_map50_95"] >= -0.03,
        "has_discovery_signal": (
            delta["macro_map50_95"] >= 0.002
            or delta["bottom3_class_map50_95"] >= 0.005
            or delta["worst_class_map50_95"] >= 0.005
        ),
    }
    result = {
        "protocol": "faruq-v3-lfdet-cgfi-discovery-v1",
        "stage": "breadth_discovery",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "design_boundary": (
            "CG1 follows LFDet CGFI Eq.14 with dynamic FFT-domain filtering and the paper's "
            "structure-1 principle of independent original/recovered projections plus summation. "
            "Because YOLO26 does not expose LFDet's exact pre-neck channel-reduction tensor in this "
            "experiment, CG1 applies CGFI to P3/P4/P5 entering Detect and routes the enhanced feature "
            "only to classification; native localization uses untouched features. The linear-stack hidden "
            "width is an explicit transfer choice because it is not specified in the available paper text."
        ),
        "controls": controls,
        "candidate": {"CG1": candidate},
        "delta_vs_D0FT": delta,
        "delta_vs_ACMC1": {name: candidate[name] - controls["ACMC1"][name] for name in METRICS},
        "criteria": criteria,
        "decision": "RETAIN" if all(criteria.values()) else "REJECT",
        "training_executed_this_call": training_executed,
    }
    summary = reports / "cgfi_seed42_screening.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Faruq-v3 LFDet-CGFI discovery")
    p.add_argument("--data-root", required=True)
    p.add_argument("--grouped-summary", required=True)
    p.add_argument("--control-summary", required=True)
    p.add_argument("--d0-checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--authorize-training", action="store_true")
    a = p.parse_args()
    print(json.dumps(run_screening(a.data_root, a.grouped_summary, a.control_summary, a.d0_checkpoint, a.output_root, seed=a.seed, device=a.device, authorize_training=a.authorize_training), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
