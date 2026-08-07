from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.fsce_cpe import make_fsce_cpe_trainer

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIGS = {
    "CPE0": REPO_ROOT / "configs/fsce_cpe/CPE0_all_positive.yaml",
    "CPE7": REPO_ROOT / "configs/fsce_cpe/CPE7_iou07.yaml",
}
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
            for name in ("ACMC1", "D0FT", "D0"):
                if name in results:
                    source = results[name]
                    break
    return {name: float(source[name]) for name in METRICS}


def _decision(candidate: dict[str, float], d0ft: dict[str, float]) -> tuple[dict, dict, str]:
    delta = {name: candidate[name] - d0ft[name] for name in METRICS}
    criteria = {
        "macro_drop_no_more_than_1_point": delta["macro_map50_95"] >= -0.010,
        "bottom3_drop_no_more_than_2_points": delta["bottom3_class_map50_95"] >= -0.020,
        "worst_drop_no_more_than_2_points": delta["worst_class_map50_95"] >= -0.020,
        "has_discovery_signal": (
            delta["macro_map50_95"] >= 0.002
            or delta["bottom3_class_map50_95"] >= 0.005
            or delta["worst_class_map50_95"] >= 0.005
        ),
    }
    return delta, criteria, "RETAIN" if all(criteria.values()) else "REJECT"


def _run_arm(
    arm: str,
    data_root: Path,
    d0_checkpoint: Path,
    output_root: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[Path, bool, dict]:
    config_payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    if config_payload.get("variant") != "fsce_cpe":
        raise RuntimeError(f"Config {arm} bukan fsce_cpe")
    run_name = f"{arm}_seed{seed}"
    run_dir = output_root / run_name
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    executed = False
    if not best.is_file():
        from ultralytics import YOLO
        trainer = make_fsce_cpe_trainer(config_payload["cpe"], d0_checkpoint=d0_checkpoint)
        if last.is_file():
            model = YOLO(str(last))
            args = {"resume": True}
            if device is not None:
                args["device"] = device
        else:
            model = YOLO(str(MODEL_YAML))
            model.load(str(d0_checkpoint))
            args = dict(config_payload["train"])
            args.update(
                {
                    "data": str(data_root / "data.yaml"),
                    "project": str(output_root),
                    "name": run_name,
                    "exist_ok": True,
                    "seed": seed,
                    "deterministic": True,
                    "plots": True,
                    "verbose": True,
                    "pretrained": True,
                }
            )
            if device is not None:
                args["device"] = device
        model.train(trainer=trainer, **args)
        executed = True
    if not best.is_file():
        raise FileNotFoundError(best)
    return best, executed, config_payload["cpe"]


def run_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    d0ft_report: str | Path,
    acmc1_report: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("FSCE-CPE breadth screening dikunci seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos split test")

    d0ft = _metrics(_load_json(d0ft_report, "D0FT report"), "D0FT")
    acmc1 = _metrics(_load_json(acmc1_report, "ACMC1 report"), "ACMC1")
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    results, decisions, configs, executed = {}, {}, {}, {}
    for arm in ("CPE0", "CPE7"):
        best, trained, config = _run_arm(
            arm, data_root, d0_checkpoint, output_root, seed=seed, device=device
        )
        report = evaluate(
            best,
            data_root,
            reports / f"{arm}_seed{seed}_val.json",
            split="val",
            device=device,
        )
        if report["metrics"].get("classes_without_ground_truth", []):
            raise RuntimeError("Validation kehilangan kelas")
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
        executed[arm] = trained

    payload = {
        "protocol": "faruq-v3-fsce-cpe-breadth-screening-v1",
        "stage": "broad_search_screening",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "mechanistic_hypothesis": "instance_level_embedding_overlap_limits_fine_grained_classification",
        "paper_operator": "FSCE_CPE_Eqs_2_to_5",
        "paper_defaults": {
            "embedding_dim": 128,
            "temperature": 0.2,
            "paper_iou_cutoff": 0.7,
            "loss_weight": 0.5,
        },
        "transfer_boundary": (
            "YOLO26 dense P3/P4/P5 one-to-many positive locations replace Faster-RCNN RoI proposals; "
            "CPE projection/loss is training-only; native box/classification branches and inference remain unchanged"
        ),
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "controls": {"D0FT": d0ft, "ACMC1": acmc1},
        "candidate": results,
        "decisions": decisions,
        "configs": configs,
        "training_executed_this_call": executed,
    }
    summary = reports / "fsce_cpe_seed42_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 FSCE-CPE breadth screening")
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
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.d0ft_report,
        args.acmc1_report,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
