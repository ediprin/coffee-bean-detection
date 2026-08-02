from __future__ import annotations

import argparse
import hashlib
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
CONFIG = REPO_ROOT / "configs/frozen_residual/FRM1_yolo26n_frozen_multilevel.yaml"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def run_faruq_v3_frozen_residual(
    data_root: str | Path,
    grouped_summary: str | Path,
    baseline_summary: str | Path,
    d0_checkpoint: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("FRM1 screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("FRM1 training belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki test")

    baseline = _load_json(baseline_summary, "D0 baseline summary")
    if int(baseline.get("seed", -1)) != seed:
        raise RuntimeError("Seed D0 baseline tidak cocok")
    if baseline.get("test_images_accessed") is not False:
        raise RuntimeError("Provenance D0 test lock tidak valid")
    static = _load_json(static_audit, "FRM1 static audit")
    if static.get("decision") != "PASS" or static.get("test_images_accessed") is not False:
        raise RuntimeError("FRM1 static preservation audit belum PASS")
    checkpoint_hash = _sha256_file(d0_checkpoint)
    if static.get("d0_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("Checkpoint D0 berbeda dari static audit")

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    audit_path = reports_root / "dataset_audit.json"
    audit = audit_dataset(data_root, audit_path, near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError(f"Audit dataset gagal: {audit_path}")

    run_dir = output_root / f"FRM1_seed{seed}"
    recover_completed_training_manifest(
        CONFIG,
        data_root,
        run_dir,
        seed,
        weights_override=d0_checkpoint,
    )
    existing_manifest = run_dir / "experiment_manifest.json"
    if existing_manifest.is_file():
        provenance = _load_json(existing_manifest, "FRM1 experiment manifest")
        if provenance.get("weights_override_sha256") != checkpoint_hash:
            raise RuntimeError(
                "Artefak FRM1 berasal dari checkpoint dasar berbeda; gunakan output baru"
            )
    training_was_run = not is_training_complete(run_dir)
    if training_was_run:
        action = "RESUME" if (run_dir / "weights/last.pt").is_file() else "START"
        print(f"{action} FRM1 | frozen D0 | seed={seed}", flush=True)
        train_experiment(
            CONFIG,
            data_root,
            output_root,
            seed,
            device=device,
            resume=True,
            weights_override=d0_checkpoint,
        )
    else:
        print("SKIP TRAINING: FRM1 seed 42 lengkap", flush=True)

    checkpoint = run_dir / "weights/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FRM1 best.pt tidak ditemukan: {checkpoint}")
    report_path = reports_root / "FRM1_seed42_val.json"
    report = evaluate(checkpoint, data_root, report_path, split="val", device=device)
    missing = report["metrics"].get("classes_without_ground_truth", [])
    if missing:
        raise RuntimeError("Validation kehilangan kelas: " + ", ".join(missing))

    d0 = _metrics(baseline)
    frm1 = _metrics(report)
    deltas = {name: frm1[name] - d0[name] for name in METRICS}
    criteria = {
        "macro_gain_at_least_0_5_point": deltas["macro_map50_95"] >= 0.005,
        "bottom3_not_lower": deltas["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_1_point": deltas["worst_class_map50_95"] >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "protocol": "faruq-v3-frozen-residual-v1",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": checkpoint_hash,
        "results": {"D0": d0, "FRM1": frm1},
        "deltas": deltas,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "AUTHORIZE_FRM1_THREE_SEED_CONFIRMATION_PROTOCOL"
            if decision == "PASS"
            else "STOP_FRM1_WITHOUT_TEST_OR_EXTRA_SEEDS"
        ),
        "training_executed_this_call": training_was_run,
    }
    summary = reports_root / "frozen_residual_seed42_decision.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 frozen-D0 residual screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_frozen_residual(
        args.data_root,
        args.grouped_summary,
        args.baseline_summary,
        args.d0_checkpoint,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
