"""Frozen seed-42 screening for ACMC1 + hard-competitor ranking (ACMC1-HCR)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.ambiguity_multilevel.audit import static_ambiguity_multilevel_audit
from coffee_detector.ambiguity_multilevel.ranking import make_ambiguity_multilevel_ranking_trainer
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.run_baseline import is_training_complete


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIG = REPO_ROOT / "configs/ambiguity_multilevel/ACMC1H_yolo26n_hard_competitor.yaml"
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(source: dict) -> dict[str, float]:
    return {name: float(source[name]) for name in METRICS}


def screening_decision(d0ft: dict, acmc1: dict, candidate: dict) -> tuple[dict, dict, dict, str]:
    """Frozen before ACMC1-HCR training; thresholds are intentionally non-trivial."""
    vs_d0ft = {name: candidate[name] - d0ft[name] for name in METRICS}
    vs_acmc1 = {name: candidate[name] - acmc1[name] for name in METRICS}
    tail_deltas = (
        vs_acmc1["bottom3_class_map50_95"],
        vs_acmc1["worst_class_map50_95"],
    )
    criteria = {
        "macro_gain_over_acmc1_at_least_0_20_point": vs_acmc1["macro_map50_95"] >= 0.002,
        "at_least_one_tail_gain_over_acmc1_at_least_0_50_point": max(tail_deltas) >= 0.005,
        "neither_tail_drop_vs_acmc1_worse_than_0_50_point": min(tail_deltas) >= -0.005,
        "macro_gain_over_d0ft_remains_at_least_0_50_point": vs_d0ft["macro_map50_95"] >= 0.005,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    return vs_d0ft, vs_acmc1, criteria, decision


def _train_or_resume(
    config: dict,
    data_yaml: Path,
    d0_checkpoint: Path,
    output_root: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[Path, bool]:
    from ultralytics import YOLO

    run_dir = output_root / f"ACMC1H_seed{seed}"
    last = run_dir / "weights/last.pt"
    training_was_run = not is_training_complete(run_dir)
    if not training_was_run:
        return run_dir, False

    train_args = dict(config["train"])
    train_args.update(
        {
            "data": str(data_yaml),
            "project": str(output_root),
            "name": f"ACMC1H_seed{seed}",
            "exist_ok": True,
            "seed": seed,
            "deterministic": True,
            "plots": True,
            "verbose": True,
        }
    )
    if device is not None:
        train_args["device"] = device

    trainer = make_ambiguity_multilevel_ranking_trainer(
        config["ambiguity_multilevel"],
        config["hard_competitor_ranking"],
        d0_checkpoint=d0_checkpoint,
    )
    if last.is_file():
        print(f"RESUME ACMC1-HCR | seed={seed}", flush=True)
        model = YOLO(str(last))
        resume_args = {"resume": True}
        if device is not None:
            resume_args["device"] = device
        model.train(trainer=trainer, **resume_args)
    else:
        print(f"START ACMC1-HCR | seed={seed}", flush=True)
        model = YOLO(str(MODEL_YAML))
        model.load(str(d0_checkpoint))
        train_args["pretrained"] = True
        model.train(trainer=trainer, **train_args)

    manifest = {
        "protocol": "faruq-v3-acmc1-hard-competitor-ranking-screening-v1",
        "code": "ACMC1H",
        "seed": seed,
        "variant": "ambiguity_multilevel_ranking",
        "model": str(MODEL_YAML),
        "ambiguity_multilevel": config["ambiguity_multilevel"],
        "hard_competitor_ranking": config["hard_competitor_ranking"],
        "train": config["train"],
        "d0_checkpoint": str(d0_checkpoint),
        "d0_checkpoint_sha256": _sha256_file(d0_checkpoint),
        "training_only_change": True,
        "inference_architecture": "ACMC1",
        "test_images_accessed": False,
        "test_opened": False,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir, True


def run_faruq_v3_acmc1h_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    acmc1_control_summary: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("ACMC1-HCR screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("ACMC1-HCR training belum diotorisasi")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    grouped = load_faruq_grouped_summary(grouped_summary, data_root)
    data_yaml = data_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml tidak ditemukan: {data_yaml}")
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki split test")

    control = _load_json(acmc1_control_summary, "ACMC1 optimization control")
    checkpoint_hash = _sha256_file(d0_checkpoint)
    if (
        control.get("protocol") != "faruq-v3-acmc-optimization-control-v1"
        or int(control.get("seed", -1)) != seed
        or control.get("decision") != "PASS"
        or control.get("test_images_accessed") is not False
        or control.get("test_opened") is not False
        or control.get("d0_checkpoint_sha256") != checkpoint_hash
    ):
        raise RuntimeError("ACMC1 optimization control tidak valid untuk ACMC1-HCR")
    results = control.get("results", {})
    if not {"D0", "D0FT", "ACMC1"} <= set(results):
        raise RuntimeError("ACMC1 control harus memuat D0, D0FT, dan ACMC1")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config.get("variant") != "ambiguity_multilevel_ranking":
        raise RuntimeError("Config ACMC1-HCR salah variant")
    ranking = config.get("hard_competitor_ranking", {})
    if (
        float(ranking.get("weight", -1)) != 0.25
        or ranking.get("branch") != "one2one_only"
        or ranking.get("competitor") != "strongest_wrong_class"
        or ranking.get("loss") != "softplus_pairwise"
    ):
        raise RuntimeError("Config HCR menyimpang dari protokol frozen")

    reports_root = output_root / "val_reports"
    static_root = output_root / "static_audits"
    reports_root.mkdir(parents=True, exist_ok=True)
    static_root.mkdir(parents=True, exist_ok=True)

    dataset_audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    static_path = static_root / "D0_seed42_acmc1h_static.json"
    static = static_ambiguity_multilevel_audit(
        MODEL_YAML,
        d0_checkpoint,
        static_path,
        nc=21,
        image_size=128,
        config=config["ambiguity_multilevel"],
    )
    if static["decision"] != "PASS":
        raise RuntimeError(f"Static audit ACMC1 architecture gagal: {static_path}")

    run_dir, training_was_run = _train_or_resume(
        config,
        data_yaml,
        d0_checkpoint,
        output_root,
        seed=seed,
        device=device,
    )
    checkpoint = run_dir / "weights/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ACMC1-HCR best.pt tidak ditemukan: {checkpoint}")

    report = evaluate(
        checkpoint,
        data_root,
        reports_root / f"ACMC1H_seed{seed}_val.json",
        split="val",
        device=device,
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation kehilangan kelas")

    d0 = _metrics(results["D0"])
    d0ft = _metrics(results["D0FT"])
    acmc1 = _metrics(results["ACMC1"])
    candidate = _metrics(report["metrics"])
    vs_d0ft, vs_acmc1, criteria, decision = screening_decision(d0ft, acmc1, candidate)

    payload = {
        "protocol": "faruq-v3-acmc1-hard-competitor-ranking-screening-v1",
        "stage": "seed42_screening",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": checkpoint_hash,
        "grouped_dataset": {
            "images_by_split": grouped["images_by_split"],
            "annotations_by_split": grouped["annotations_by_split"],
        },
        "method": {
            "base": "ACMC1",
            "training_only_change": "hard_competitor_pairwise_softplus",
            "branch": "one2one_only",
            "weight": 0.25,
            "validation_confusion_pairs_hardcoded": False,
            "inference_parameter_delta": 0,
            "inference_flops_delta": 0,
        },
        "results": {"D0": d0, "D0FT": d0ft, "ACMC1": acmc1, "ACMC1H": candidate},
        "deltas_acmc1h_vs_d0ft": vs_d0ft,
        "deltas_acmc1h_vs_acmc1": vs_acmc1,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "AUTHORIZE_PAIRED_THREE_SEED_ACMC1H_CONFIRMATION"
            if decision == "PASS"
            else "STOP_OPTIMIZATION_KEEP_ACMC1"
        ),
        "training_executed_this_call": training_was_run,
    }
    summary = reports_root / "acmc1h_seed42_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 ACMC1-HCR seed42 screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--acmc1-control-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_acmc1h_screening(
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.acmc1_control_summary,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
