from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import (
    load_faruq_grouped_summary,
)
from coffee_detector.run_baseline import is_training_complete
from coffee_detector.train import (
    recover_completed_training_manifest,
    train_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "MHC0": REPO_ROOT / "configs/multilevel_head/MHC0_yolo26n_p5_control.yaml",
    "MHF1": REPO_ROOT / "configs/multilevel_head/MHF1_yolo26n_pyramid_fusion.yaml",
}
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    missing = [name for name in METRICS if name not in source]
    if missing:
        raise KeyError("Metrik belum lengkap: " + ", ".join(missing))
    return {name: float(source[name]) for name in METRICS}


def _comparison(left_name: str, left: dict, right_name: str, right: dict) -> dict:
    deltas = {name: right[name] - left[name] for name in METRICS}
    return {
        "baseline": left_name,
        "candidate": right_name,
        "deltas": deltas,
    }


def run_faruq_v3_multilevel_head(
    data_root: str | Path,
    grouped_summary: str | Path,
    baseline_summary: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("Screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi; gunakan --authorize-training")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki test")
    static = _load_json(static_audit, "Static audit")
    if static.get("decision") != "PASS" or static.get("test_images_accessed") is not False:
        raise RuntimeError("Static multilevel-head audit belum PASS dengan test terkunci")
    baseline = _load_json(baseline_summary, "D0 baseline summary")
    if int(baseline.get("seed", -1)) != seed:
        raise RuntimeError("Seed baseline D0 tidak cocok")
    if baseline.get("test_images_accessed") is not False:
        raise RuntimeError("Provenance baseline test lock tidak valid")
    baseline_metrics = _metrics(baseline)

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    audit_path = reports_root / "dataset_audit.json"
    audit = audit_dataset(data_root, audit_path, near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError(f"Audit dataset gagal: {audit_path}")

    results = {"D0": baseline_metrics}
    training_executed = {}
    for code, config in CONFIGS.items():
        run_dir = output_root / f"{code}_seed{seed}"
        recover_completed_training_manifest(config, data_root, run_dir, seed)
        training_executed[code] = not is_training_complete(run_dir)
        if training_executed[code]:
            action = "RESUME" if (run_dir / "weights/last.pt").is_file() else "START"
            print(f"{action} {code} | seed={seed}", flush=True)
            train_experiment(
                config,
                data_root,
                output_root,
                seed,
                device=device,
                resume=True,
            )
        else:
            print(f"SKIP TRAINING: {code} seed {seed} lengkap", flush=True)
        checkpoint = run_dir / "weights/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint belum tersedia: {checkpoint}")
        report_path = reports_root / f"{code}_seed{seed}_val.json"
        report = evaluate(checkpoint, data_root, report_path, split="val", device=device)
        missing = report["metrics"].get("classes_without_ground_truth", [])
        if missing:
            raise RuntimeError("Validation kehilangan kelas: " + ", ".join(missing))
        results[code] = _metrics(report)

    comparisons = {
        "D0_vs_MHC0": _comparison("D0", results["D0"], "MHC0", results["MHC0"]),
        "D0_vs_MHF1": _comparison("D0", results["D0"], "MHF1", results["MHF1"]),
        "MHC0_vs_MHF1": _comparison(
            "MHC0", results["MHC0"], "MHF1", results["MHF1"]
        ),
    }
    d0 = comparisons["D0_vs_MHF1"]["deltas"]
    control = comparisons["MHC0_vs_MHF1"]["deltas"]
    criteria = {
        "d0_macro_gain_at_least_0_5_point": d0["macro_map50_95"] >= 0.005,
        "d0_bottom3_not_lower": d0["bottom3_class_map50_95"] >= 0.0,
        "d0_worst_drop_no_more_than_1_point": d0["worst_class_map50_95"] >= -0.01,
        "control_macro_gain_at_least_0_5_point": control["macro_map50_95"] >= 0.005,
        "control_bottom3_not_lower": control["bottom3_class_map50_95"] >= 0.0,
        "control_worst_drop_no_more_than_1_point": control["worst_class_map50_95"] >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "protocol": "faruq-v3-multilevel-head-training-v1",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "results": results,
        "comparisons": comparisons,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "AUTHORIZE_THREE_SEED_MULTILEVEL_CONFIRMATION_PROTOCOL"
            if decision == "PASS"
            else "STOP_MULTILEVEL_HEAD_WITHOUT_TEST_OR_EXTRA_SEEDS"
        ),
        "training_executed_this_call": training_executed,
    }
    summary = reports_root / "multilevel_head_seed42_decision.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 MHC0/MHF1 one-seed screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_multilevel_head(
        args.data_root,
        args.grouped_summary,
        args.baseline_summary,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
