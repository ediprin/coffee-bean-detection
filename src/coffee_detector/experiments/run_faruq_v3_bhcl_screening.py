"""Validation-only breadth screening for BHCL on native YOLO26 TAL positives."""

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
CONFIG = REPO_ROOT / "configs/bhcl/BH1_yolo26n_entity_family.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
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


def _load_config() -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    config = BHCLConfig.from_mapping(payload.get("bhcl"))
    hierarchy = payload.get("hierarchy", {})
    if hierarchy.get("coarse_field") != "entity_family" or hierarchy.get("levels_excluding_root") != 2:
        raise RuntimeError("BH1 hierarchy config berubah dari frozen SNI21 two-level tree")
    return {**payload, "bhcl": config.to_dict()}


def run_bhcl_screening(
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
        raise ValueError("BH1 breadth discovery dikunci seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki split test")
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)

    d0ft = _metrics(_load_json(d0ft_report, "D0FT report"), "D0FT")
    acmc1 = _metrics(_load_json(acmc1_report, "ACMC1 report"), "ACMC1")
    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    config = _load_config()
    run_name = f"BH1_seed{seed}"
    run_dir = output_root / run_name
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False

    if not best.is_file():
        from ultralytics import YOLO
        trainer = make_bhcl_trainer(config["bhcl"], d0_checkpoint=d0_checkpoint)
        if last.is_file():
            model = YOLO(str(last))
            train_args = {"resume": True}
            if device is not None:
                train_args["device"] = device
        else:
            model = YOLO(str(MODEL_YAML))
            model.load(str(d0_checkpoint))
            train_args = dict(config["train"])
            train_args.update(
                {
                    "data": str(data_root / "data.yaml"),
                    "project": str(output_root),
                    "name": run_name,
                    "exist_ok": True,
                    "seed": seed,
                    "deterministic": True,
                    "plots": True,
                    "verbose": True,
                }
            )
            if device is not None:
                train_args["device"] = device
        print(f"{'RESUME' if last.is_file() else 'START'} BH1 | seed={seed}", flush=True)
        model.train(trainer=trainer, **train_args)
        training_executed = True

    if not best.is_file():
        raise FileNotFoundError(best)
    report = evaluate(
        best, data_root, reports_root / f"BH1_seed{seed}_val.json", split="val", device=device
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation kehilangan kelas")
    candidate = _metrics(report)
    delta_d0ft = {name: candidate[name] - d0ft[name] for name in METRICS}
    delta_acmc1 = {name: candidate[name] - acmc1[name] for name in METRICS}
    criteria = {
        "macro_not_below_d0ft_by_more_than_0_2_point": delta_d0ft["macro_map50_95"] >= -0.002,
        "bottom3_not_below_d0ft_by_more_than_2_points": delta_d0ft["bottom3_class_map50_95"] >= -0.020,
        "worst_not_below_d0ft_by_more_than_3_points": delta_d0ft["worst_class_map50_95"] >= -0.030,
        "has_discovery_signal": (
            delta_d0ft["macro_map50_95"] >= 0.002
            or delta_d0ft["bottom3_class_map50_95"] >= 0.005
            or delta_d0ft["worst_class_map50_95"] >= 0.005
        ),
    }
    decision = "RETAIN" if all(criteria.values()) else "REJECT"
    payload = {
        "protocol": "faruq-v3-bhcl-entity-family-breadth-v1",
        "stage": "breadth_discovery",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "candidate": "BH1",
        "paper_parameters": {"lambda_bhcl": 0.6, "temperature": 0.1, "epsilon": 0.1},
        "hierarchy": {
            "root": "SNI21_object",
            "level_1": "entity_family",
            "level_2": "21_leaf_classes",
            "root_excluded_from_loss": True,
            "source": "configs/sni21/structured_ontology_v1.yaml",
        },
        "adaptation_boundary": (
            "BHCL is applied to APCL-capacity-matched 128D embeddings from native YOLO26 P3/P4/P5 one-to-many TAL positives. "
            "The DETR-specific decoupled-query module is not transferred. Native box heads and inference remain unchanged. "
            "Because TAL supplies multiple positive locations per GT, this breadth arm does not create the paper's explicit paired augmented views. "
            "Prototype initialization is deterministic zero because the paper does not specify initialization; Eq.10 EMA is then used literally. "
            "The PDF prints an extra outer minus in Eq.9 despite Eq.8 already defining -log pair loss; implementation aggregates the positive -log pair loss."
        ),
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "results": {"D0FT": d0ft, "ACMC1": acmc1, "BH1": candidate},
        "deltas": {"BH1_vs_D0FT": delta_d0ft, "BH1_vs_ACMC1": delta_acmc1},
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "KEEP_BH1_FOR_CANDIDATE_POOL" if decision == "RETAIN" else "ARCHIVE_BH1_AND_CONTINUE_BROAD_SEARCH"
        ),
        "training_executed_this_call": training_executed,
    }
    summary = reports_root / "bh1_seed42_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 BHCL breadth screening")
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
    result = run_bhcl_screening(
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
