"""Seed-42 validation screening for ACMC2 entropy+margin gating."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coffee_detector.ambiguity_multilevel.audit import static_ambiguity_multilevel_audit
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.run_baseline import is_training_complete
from coffee_detector.train import load_experiment, recover_completed_training_manifest, train_experiment


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIG = REPO_ROOT / "configs/ambiguity_multilevel/ACMC2_yolo26n_entropy_margin.yaml"
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


def screening_decision(d0ft: dict, acmc1: dict, acmc2: dict) -> tuple[dict, dict, dict, str]:
    """Frozen seed-42 progression gate, defined before ACMC2 training."""
    vs_d0ft = {name: acmc2[name] - d0ft[name] for name in METRICS}
    vs_acmc1 = {name: acmc2[name] - acmc1[name] for name in METRICS}
    criteria = {
        "macro_gain_over_d0ft_at_least_0_5_point": vs_d0ft["macro_map50_95"] >= 0.005,
        "bottom3_not_lower_than_d0ft": vs_d0ft["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_1_point_vs_d0ft": vs_d0ft["worst_class_map50_95"] >= -0.01,
        "macro_not_lower_than_acmc1": vs_acmc1["macro_map50_95"] >= 0.0,
        "at_least_one_tail_metric_improves_over_acmc1": max(
            vs_acmc1["bottom3_class_map50_95"], vs_acmc1["worst_class_map50_95"]
        ) > 0.0,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    return vs_d0ft, vs_acmc1, criteria, decision


def run_faruq_v3_acmc2_screening(
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
        raise ValueError("ACMC2 screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("ACMC2 training belum diotorisasi")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    grouped = load_faruq_grouped_summary(grouped_summary, data_root)
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
        raise RuntimeError("ACMC1 optimization control tidak valid untuk ACMC2")
    results = control.get("results", {})
    if not {"D0", "D0FT", "ACMC1"} <= set(results):
        raise RuntimeError("ACMC1 control harus memuat D0, D0FT, dan ACMC1")

    reports_root = output_root / "val_reports"
    static_root = output_root / "static_audits"
    reports_root.mkdir(parents=True, exist_ok=True)
    static_root.mkdir(parents=True, exist_ok=True)

    dataset_audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    config_payload = load_experiment(CONFIG)
    static_path = static_root / "D0_seed42_acmc2_static.json"
    static = static_ambiguity_multilevel_audit(
        MODEL_YAML,
        d0_checkpoint,
        static_path,
        nc=21,
        image_size=128,
        config=config_payload["ambiguity_multilevel"],
    )
    if static["decision"] != "PASS":
        raise RuntimeError(f"Static audit ACMC2 gagal: {static_path}")

    run_dir = output_root / f"ACMC2_seed{seed}"
    recover_completed_training_manifest(CONFIG, data_root, run_dir, seed, weights_override=d0_checkpoint)
    manifest = run_dir / "experiment_manifest.json"
    if manifest.is_file():
        provenance = _load_json(manifest, "ACMC2 manifest")
        if provenance.get("weights_override_sha256") != checkpoint_hash:
            raise RuntimeError("Run ACMC2 memakai checkpoint D0 yang berbeda")
        ambiguity = provenance.get("ambiguity_multilevel", {})
        if ambiguity.get("ambiguity_mode") != "entropy_margin":
            raise RuntimeError("Manifest ACMC2 bukan entropy_margin")

    training_was_run = not is_training_complete(run_dir)
    if training_was_run:
        action = "RESUME" if (run_dir / "weights/last.pt").is_file() else "START"
        print(f"{action} ACMC2 | entropy+margin gate | seed={seed}", flush=True)
        train_experiment(
            CONFIG,
            data_root,
            output_root,
            seed,
            device=device,
            resume=True,
            weights_override=d0_checkpoint,
        )

    checkpoint = run_dir / "weights/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ACMC2 best.pt tidak ditemukan: {checkpoint}")
    report = evaluate(
        checkpoint,
        data_root,
        reports_root / f"ACMC2_seed{seed}_val.json",
        split="val",
        device=device,
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation kehilangan kelas")

    d0 = _metrics(results["D0"])
    d0ft = _metrics(results["D0FT"])
    acmc1 = _metrics(results["ACMC1"])
    acmc2 = _metrics(report["metrics"])
    vs_d0ft, vs_acmc1, criteria, decision = screening_decision(d0ft, acmc1, acmc2)

    payload = {
        "protocol": "faruq-v3-acmc2-entropy-margin-v1",
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
        "acmc1_control_summary": str(Path(acmc1_control_summary).expanduser().resolve()),
        "static_audit": str(static_path),
        "results": {"D0": d0, "D0FT": d0ft, "ACMC1": acmc1, "ACMC2": acmc2},
        "deltas_acmc2_vs_d0ft": vs_d0ft,
        "deltas_acmc2_vs_acmc1": vs_acmc1,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "AUTHORIZE_PAIRED_THREE_SEED_ACMC2_CONFIRMATION"
            if decision == "PASS"
            else "STOP_ACMC2_KEEP_ACMC1"
        ),
        "training_executed_this_call": training_was_run,
    }
    summary = reports_root / "acmc2_seed42_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 ACMC2 seed-42 validation screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--acmc1-control-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_acmc2_screening(
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
