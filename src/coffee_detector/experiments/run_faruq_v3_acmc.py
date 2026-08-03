"""Validation-only screening for the native one-stage ACMC classification head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.run_baseline import is_training_complete
from coffee_detector.train import recover_completed_training_manifest, train_experiment


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/ambiguity_multilevel/ACMC1_yolo26n_field_level.yaml"
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def run_faruq_v3_acmc(
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
    confirmation_seed: bool = False,
) -> dict:
    """Run one pre-registered seed only after the D0-preservation audit passes."""
    if seed != 42 and not confirmation_seed:
        raise ValueError("ACMC1 screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("ACMC1 training belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki split test")

    baseline = _load_json(baseline_summary, "D0 baseline summary")
    if int(baseline.get("seed", -1)) != seed or baseline.get("test_images_accessed") is not False:
        raise RuntimeError("Provenance D0 baseline tidak sesuai")
    static = _load_json(static_audit, "ACMC static audit")
    checkpoint_hash = _sha256_file(d0_checkpoint)
    required_static_gates = {
        "native_d0_head_bitwise_preserved",
        "zero_output_is_d0",
        "zero_boxes_bitwise_equal",
        "zero_scores_bitwise_equal",
        "no_roi_align",
        "no_topk_candidate_selection",
        "no_box_decode_before_classification",
        "finite_correction_gradients",
        "active_correction_changes_scores",
        "active_correction_preserves_boxes",
    }
    if (
        static.get("decision") != "PASS"
        or static.get("test_images_accessed") is not False
        or static.get("d0_checkpoint_sha256") != checkpoint_hash
        or not required_static_gates <= set(name for name, passed in static.get("gates", {}).items() if passed)
    ):
        raise RuntimeError("ACMC static audit belum PASS untuk checkpoint D0 ini")

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    run_dir = output_root / f"ACMC1_seed{seed}"
    recover_completed_training_manifest(CONFIG, data_root, run_dir, seed, weights_override=d0_checkpoint)
    manifest = run_dir / "experiment_manifest.json"
    if manifest.is_file():
        provenance = _load_json(manifest, "ACMC1 manifest")
        if provenance.get("weights_override_sha256") != checkpoint_hash:
            raise RuntimeError("Run ACMC1 memakai checkpoint D0 yang berbeda")
    training_was_run = not is_training_complete(run_dir)
    if training_was_run:
        action = "RESUME" if (run_dir / "weights/last.pt").is_file() else "START"
        print(f"{action} ACMC1 | native one-stage | seed={seed}", flush=True)
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
        raise FileNotFoundError(f"ACMC1 best.pt tidak ditemukan: {checkpoint}")
    report = evaluate(checkpoint, data_root, reports_root / "ACMC1_seed42_val.json", split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation kehilangan kelas")

    d0, acmc1 = _metrics(baseline), _metrics(report)
    deltas = {name: acmc1[name] - d0[name] for name in METRICS}
    criteria = {
        "macro_gain_at_least_0_5_point": deltas["macro_map50_95"] >= 0.005,
        "bottom3_not_lower": deltas["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_1_point": deltas["worst_class_map50_95"] >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "protocol": "faruq-v3-acmc-one-stage-v1",
        "stage": "confirmation" if confirmation_seed else "screening",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": checkpoint_hash,
        "static_audit": str(Path(static_audit).expanduser().resolve()),
        "results": {"D0": d0, "ACMC1": acmc1},
        "deltas": deltas,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "AUTHORIZE_ACMC1_THREE_SEED_CONFIRMATION"
            if decision == "PASS"
            else "STOP_ACMC1_WITHOUT_TEST_OR_EXTRA_SEEDS"
        ),
        "training_executed_this_call": training_was_run,
    }
    summary = reports_root / "acmc1_seed42_decision.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 ACMC one-seed validation screening")
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
    result = run_faruq_v3_acmc(
        args.data_root, args.grouped_summary, args.baseline_summary, args.d0_checkpoint,
        args.static_audit, args.output_root, seed=args.seed, device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
