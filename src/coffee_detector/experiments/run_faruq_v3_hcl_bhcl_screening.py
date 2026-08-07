"""Validation-only paired breadth screen: HCL1 vs BH1 on identical YOLO26 scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.bhcl import BHCLConfig, make_bhcl_trainer
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "HCL1": REPO_ROOT / "configs/bhcl/HCL1_yolo26n_entity_family.yaml",
    "BH1": REPO_ROOT / "configs/bhcl/BH1_yolo26n_entity_family.yaml",
}
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


def _metrics(payload: dict, preferred: str | None = None) -> dict[str, float]:
    source = payload.get("metrics", payload)
    if "results" in source and isinstance(source["results"], dict):
        results = source["results"]
        if preferred and preferred in results:
            source = results[preferred]
        else:
            source = next(iter(results.values()))
    return {name: float(source[name]) for name in METRICS}


def _load_arm(arm: str) -> dict:
    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    config = BHCLConfig.from_mapping(payload.get("bhcl"))
    expected = "hcl" if arm == "HCL1" else "bhcl"
    if config.variant != expected:
        raise RuntimeError(f"{arm} variant berubah: {config.variant}")
    hierarchy = payload.get("hierarchy", {})
    if hierarchy.get("coarse_field") != "entity_family" or hierarchy.get("levels_excluding_root") != 2:
        raise RuntimeError(f"{arm} hierarchy berubah")
    return {**payload, "bhcl": config.to_dict()}


def _decision(metrics: dict, d0ft: dict) -> tuple[dict, dict, str]:
    delta = {name: metrics[name] - d0ft[name] for name in METRICS}
    criteria = {
        "macro_not_below_d0ft_by_more_than_0_2_point": delta["macro_map50_95"] >= -0.002,
        "bottom3_not_below_d0ft_by_more_than_2_points": delta["bottom3_class_map50_95"] >= -0.020,
        "worst_not_below_d0ft_by_more_than_3_points": delta["worst_class_map50_95"] >= -0.030,
        "has_discovery_signal": (
            delta["macro_map50_95"] >= 0.002
            or delta["bottom3_class_map50_95"] >= 0.005
            or delta["worst_class_map50_95"] >= 0.005
        ),
    }
    return delta, criteria, "RETAIN" if all(criteria.values()) else "REJECT"


def _train_arm(arm, data_root, d0_checkpoint, output_root, seed, device):
    from ultralytics import YOLO
    payload = _load_arm(arm)
    run_name = f"{arm}_seed{seed}"
    run_dir = output_root / run_name
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    executed = False
    if not best.is_file():
        trainer = make_bhcl_trainer(payload["bhcl"], d0_checkpoint=d0_checkpoint)
        if last.is_file():
            model = YOLO(str(last))
            args = {"resume": True}
            if device is not None:
                args["device"] = device
        else:
            model = YOLO(str(REPO_ROOT / payload["model"]))
            model.load(str(d0_checkpoint))
            args = dict(payload["train"])
            args.update({
                "data": str(data_root / "data.yaml"),
                "project": str(output_root),
                "name": run_name,
                "exist_ok": True,
                "seed": seed,
                "deterministic": True,
                "plots": True,
                "verbose": True,
            })
            if device is not None:
                args["device"] = device
        print(f"{'RESUME' if last.is_file() else 'START'} {arm} | seed={seed}", flush=True)
        model.train(trainer=trainer, **args)
        executed = True
    if not best.is_file():
        raise FileNotFoundError(best)
    return best, executed, payload["bhcl"]


def run_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    d0ft_report: str | Path,
    acmc1_report: str | Path,
    output_root: str | Path,
    *, seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("HCL/BHCL breadth discovery dikunci seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")
    data_root = Path(data_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh memiliki split test")
    d0ft = _metrics(_load_json(d0ft_report, "D0FT"), "D0FT")
    acmc1 = _metrics(_load_json(acmc1_report, "ACMC1"), "ACMC1")
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    results = {"D0FT": d0ft, "ACMC1": acmc1}
    decisions, configs, trained = {}, {}, {}
    for arm in ("HCL1", "BH1"):
        best, executed, config = _train_arm(
            arm, data_root, d0_checkpoint, output_root, seed, device
        )
        report = evaluate(best, data_root, reports / f"{arm}_seed{seed}_val.json", split="val", device=device)
        if report["metrics"].get("classes_without_ground_truth", []):
            raise RuntimeError(f"{arm} validation kehilangan kelas")
        metrics = _metrics(report)
        delta, criteria, decision = _decision(metrics, d0ft)
        results[arm] = metrics
        decisions[arm] = {
            "delta_vs_D0FT": delta,
            "delta_vs_ACMC1": {name: metrics[name] - acmc1[name] for name in METRICS},
            "criteria": criteria,
            "decision": decision,
        }
        configs[arm] = config
        trained[arm] = executed

    bhcl_minus_hcl = {name: results["BH1"][name] - results["HCL1"][name] for name in METRICS}
    payload = {
        "protocol": "faruq-v3-hcl-bhcl-entity-family-breadth-v2",
        "stage": "breadth_discovery_ablation",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "paper_parameters": {"loss_weight": 0.6, "temperature": 0.1, "epsilon_bhcl": 0.1},
        "hierarchy": "SNI21 root -> entity_family -> 21 leaves; root excluded",
        "ablation_contract": (
            "HCL1 and BH1 share the same YOLO26 checkpoint, 128D P3/P4/P5 projection, TAL positives, "
            "two-level SNI21 hierarchy, training budget, lambda=0.6 and tau=0.1. BH1 alone adds Eq.8 class-balanced prototype denominator and Eq.10 EMA prototypes."
        ),
        "results": results,
        "decisions": decisions,
        "BH1_minus_HCL1": bhcl_minus_hcl,
        "configs": configs,
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "training_executed_this_call": trained,
    }
    summary = reports / "hcl_bhcl_seed42_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Faruq-v3 HCL1 vs BH1 breadth screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--d0ft-report", required=True)
    parser.add_argument("--acmc1-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_screening(
        args.data_root, args.grouped_summary, args.d0_checkpoint, args.d0ft_report,
        args.acmc1_report, args.output_root, seed=args.seed, device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
