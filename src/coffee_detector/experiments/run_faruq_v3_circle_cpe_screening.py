from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.circle_cpe import make_circle_cpe_trainer
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIGS = {
    "CIR0": REPO_ROOT / "configs/circle_cpe/CIR0_all_positive.yaml",
    "CIR7": REPO_ROOT / "configs/circle_cpe/CIR7_iou07.yaml",
}
MATCHED_CPE = {"CIR0": "CPE0", "CIR7": "CPE7"}
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(payload: dict, preferred_arm: str | None = None) -> dict[str, float]:
    source = payload.get("metrics", payload)
    if "results" in source and isinstance(source["results"], dict):
        results = source["results"]
        if preferred_arm and preferred_arm in results:
            source = results[preferred_arm]
        else:
            for name in ("CPE0", "CPE7", "D0FT", "D0"):
                if name in results:
                    source = results[name]
                    break
    return {name: float(source[name]) for name in METRICS}


def _decision(candidate: dict[str, float], cpe: dict[str, float], d0ft: dict[str, float]):
    delta_cpe = {name: candidate[name] - cpe[name] for name in METRICS}
    delta_d0ft = {name: candidate[name] - d0ft[name] for name in METRICS}
    criteria = {
        "macro_gain_vs_matched_cpe_at_least_0_2_point": delta_cpe["macro_map50_95"] >= 0.002,
        "bottom3_drop_vs_matched_cpe_no_more_than_1_point": delta_cpe["bottom3_class_map50_95"] >= -0.010,
        "worst_drop_vs_matched_cpe_no_more_than_1_point": delta_cpe["worst_class_map50_95"] >= -0.010,
        "macro_vs_d0ft_drop_no_more_than_0_2_point": delta_d0ft["macro_map50_95"] >= -0.002,
        "tail_signal_vs_d0ft": (
            delta_d0ft["bottom3_class_map50_95"] >= 0.005
            or delta_d0ft["worst_class_map50_95"] >= 0.005
        ),
    }
    return delta_cpe, delta_d0ft, criteria, "RETAIN" if all(criteria.values()) else "REJECT"


def _run_arm(arm, data_root, d0_checkpoint, output_root, *, seed, device):
    config_payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    if config_payload.get("variant") != "circle_cpe":
        raise RuntimeError(f"Config {arm} bukan circle_cpe")
    run_name = f"{arm}_seed{seed}"
    run_dir = output_root / run_name
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    executed = False
    if not best.is_file():
        from ultralytics import YOLO
        trainer = make_circle_cpe_trainer(config_payload["circle_cpe"], d0_checkpoint=d0_checkpoint)
        if last.is_file():
            model = YOLO(str(last))
            args = {"resume": True}
            if device is not None:
                args["device"] = device
        else:
            model = YOLO(str(MODEL_YAML))
            model.load(str(d0_checkpoint))
            args = dict(config_payload["train"])
            args.update({
                "data": str(data_root / "data.yaml"),
                "project": str(output_root),
                "name": run_name,
                "exist_ok": True,
                "seed": seed,
                "deterministic": True,
                "plots": True,
                "verbose": True,
                "pretrained": True,
            })
            if device is not None:
                args["device"] = device
        model.train(trainer=trainer, **args)
        executed = True
    if not best.is_file():
        raise FileNotFoundError(best)
    return best, executed, config_payload["circle_cpe"]


def run_screening(
    data_root,
    grouped_summary,
    d0_checkpoint,
    d0ft_report,
    cpe0_report,
    cpe7_report,
    output_root,
    *,
    seed=42,
    device=None,
    authorize_training=False,
):
    if seed != 42:
        raise ValueError("Circle-CPE screening dikunci seed42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training hanya setelah protocol dibekukan")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos split test")

    controls = {
        "D0FT": _metrics(_load_json(d0ft_report, "D0FT report"), "D0FT"),
        "CPE0": _metrics(_load_json(cpe0_report, "CPE0 report"), "CPE0"),
        "CPE7": _metrics(_load_json(cpe7_report, "CPE7 report"), "CPE7"),
    }
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    results, decisions, configs, executed = {}, {}, {}, {}
    for arm in ("CIR0", "CIR7"):
        best, trained, config = _run_arm(
            arm, data_root, d0_checkpoint, output_root, seed=seed, device=device
        )
        report = evaluate(
            best, data_root, reports / f"{arm}_seed{seed}_val.json", split="val", device=device
        )
        if report["metrics"].get("classes_without_ground_truth", []):
            raise RuntimeError("Validation kehilangan kelas")
        metrics = _metrics(report)
        matched_name = MATCHED_CPE[arm]
        dcpe, dd0ft, criteria, decision = _decision(
            metrics, controls[matched_name], controls["D0FT"]
        )
        results[arm] = metrics
        decisions[arm] = {
            "matched_supcon_control": matched_name,
            "delta_vs_matched_cpe": dcpe,
            "delta_vs_D0FT": dd0ft,
            "criteria": criteria,
            "decision": decision,
        }
        configs[arm] = config
        executed[arm] = trained

    retained = [arm for arm in ("CIR0", "CIR7") if decisions[arm]["decision"] == "RETAIN"]
    payload = {
        "protocol": "faruq-v3-circle-cpe-matched-objective-screening-v1",
        "stage": "seed42_matched_objective_screening",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "hypothesis": "shared_fine_grained_class_boundaries_benefit_from_adaptive_pair_weighting",
        "causal_contrast": "same_CPE_projection_assignment_schedule_and_inference; SupCon objective replaced by Circle objective",
        "circle_defaults": {"margin": 0.25, "gamma": 256.0},
        "engineering_calibration": {
            "loss_weight": 0.005,
            "status": "frozen_before_validation_results",
            "reason": "match approximate initial auxiliary contribution scale; not a Circle-paper default",
        },
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "controls": controls,
        "candidate": results,
        "decisions": decisions,
        "retained_for_multiseed_confirmation": retained,
        "next_action": "RUN_PAIRED_MULTI_SEED_CONFIRMATION" if retained else "STOP_CIRCLE_CPE",
        "configs": configs,
        "training_executed_this_call": executed,
    }
    summary = reports / "circle_cpe_seed42_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Faruq-v3 matched Circle-CPE objective screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--d0ft-report", required=True)
    parser.add_argument("--cpe0-report", required=True)
    parser.add_argument("--cpe7-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_screening(
        args.data_root, args.grouped_summary, args.d0_checkpoint, args.d0ft_report,
        args.cpe0_report, args.cpe7_report, args.output_root,
        seed=args.seed, device=args.device, authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
