from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.gds_cls import GDSClsConfig, make_gds_cls_trainer

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/gds_cls/GDSC1_yolo26n_grid_distance_classification.yaml"
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


def run_screening(data_root, grouped_summary, control_summary, d0_checkpoint, output_root,
                  *, seed=42, device=None, authorize_training=False) -> dict:
    if int(seed) != 42:
        raise ValueError("GDSC1 discovery dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("GDSC1 training belum diotorisasi")
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
    config = GDSClsConfig.from_mapping(payload["gds_cls"])
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    from ultralytics import YOLO
    run_dir = output_root / f"GDSC1_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    trainer = make_gds_cls_trainer(config, d0_checkpoint=d0_checkpoint)
    training_executed = False
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
            args.update({
                "data": str(data_root / "data.yaml"),
                "project": str(output_root),
                "name": f"GDSC1_seed{seed}",
                "exist_ok": True,
                "seed": int(seed),
                "deterministic": True,
                "plots": True,
                "verbose": True,
            })
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        training_executed = True
    if not best.is_file():
        raise FileNotFoundError(best)

    report_path = reports / f"GDSC1_seed{seed}_val.json"
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
        "protocol": "faruq-v3-zhao-gds-classification-aux-discovery-v1",
        "stage": "breadth_discovery",
        "seed": int(seed),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "design_boundary": (
            "GDSC1 preserves Zhao et al.'s k=7 and T=0.05 grid-distance criterion but does not "
            "claim literal S2A-Net label assignment. YOLO26 has anchor points plus Task-Aligned "
            "Assignment rather than predefined oriented anchor boxes. Therefore GDSC1 computes the "
            "horizontal Eq.1-3 specialization on decoded native-TAL positive boxes versus their assigned "
            "GT and applies extra classification BCE only when D_grid<T. Native TAL, box loss, DFL, and "
            "one-to-one loss stay unchanged. Ordered horizontal grid correspondence replaces the source "
            "Hungarian rotated-point matching; auxiliary_weight=0.25 is a transfer hyperparameter, not a "
            "paper value."
        ),
        "controls": controls,
        "candidate": {"GDSC1": candidate},
        "delta_vs_D0FT": delta,
        "delta_vs_ACMC1": {name: candidate[name] - controls["ACMC1"][name] for name in METRICS},
        "criteria": criteria,
        "decision": "RETAIN" if all(criteria.values()) else "REJECT",
        "training_executed_this_call": training_executed,
    }
    summary = reports / "gds_cls_seed42_screening.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Faruq-v3 Zhao GDS classification-aux discovery")
    p.add_argument("--data-root", required=True)
    p.add_argument("--grouped-summary", required=True)
    p.add_argument("--control-summary", required=True)
    p.add_argument("--d0-checkpoint", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device")
    p.add_argument("--authorize-training", action="store_true")
    a = p.parse_args()
    print(json.dumps(run_screening(a.data_root, a.grouped_summary, a.control_summary,
                                   a.d0_checkpoint, a.output_root, seed=a.seed,
                                   device=a.device, authorize_training=a.authorize_training),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
