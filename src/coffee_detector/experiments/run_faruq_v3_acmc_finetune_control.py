"""Optimization-matched D0 control for the ACMC1 seed-42 screening result."""

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
CONFIG = REPO_ROOT / "configs/fine_tune_control/D0FT_yolo26n_p3.yaml"
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


def _metrics(payload: dict, result_key: str | None = None) -> dict[str, float]:
    """Read evaluation metrics from either a raw report or a screening summary.

    ``evaluate`` writes its values under ``metrics``.  The ACMC screening
    runner instead stores the two arms under ``results``.  The optimization
    control deliberately accepts both formats, while requiring the intended
    arm explicitly for a multi-arm summary.
    """
    candidates: list[dict] = []
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        candidates.append(metrics)
    if all(name in payload for name in METRICS):
        candidates.append(payload)
    results = payload.get("results")
    if isinstance(results, dict):
        if result_key is None:
            raise KeyError("Summary multi-arm memerlukan result_key")
        result = results.get(result_key)
        if isinstance(result, dict):
            candidates.append(result)

    for source in candidates:
        if all(name in source for name in METRICS):
            return {name: float(source[name]) for name in METRICS}
    label = result_key or "metrics"
    raise KeyError(f"Metrik {METRICS} tidak ditemukan untuk arm {label}")


def run_d0ft_continuation(
    data_root: str | Path,
    output_root: str | Path,
    d0_checkpoint: str | Path,
    *,
    seed: int,
    device: str | None = None,
) -> tuple[dict, bool]:
    """Continue a pinned D0 checkpoint with the native head and evaluate it.

    This is intentionally reusable by the paired confirmation protocol.  It
    never trains from an arbitrary checkpoint: the local manifest must retain
    the hash of the D0 checkpoint used as the continuation source.
    """
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256_file(d0_checkpoint)
    run_dir = output_root / f"D0FT_seed{seed}"
    recover_completed_training_manifest(CONFIG, data_root, run_dir, seed, weights_override=d0_checkpoint)
    manifest = run_dir / "experiment_manifest.json"
    if manifest.is_file():
        provenance = _load_json(manifest, "D0FT manifest")
        if provenance.get("weights_override_sha256") != checkpoint_hash:
            raise RuntimeError("D0FT tidak berasal dari checkpoint D0 yang dipin")
    training_was_run = not is_training_complete(run_dir)
    if training_was_run:
        action = "RESUME" if (run_dir / "weights/last.pt").is_file() else "START"
        print(f"{action} D0FT | native D0 continuation | seed={seed}", flush=True)
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
        raise FileNotFoundError(f"D0FT best.pt tidak ditemukan: {checkpoint}")
    report = evaluate(
        checkpoint,
        data_root,
        reports_root / f"D0FT_seed{seed}_val.json",
        split="val",
        device=device,
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation kehilangan kelas")
    return report, training_was_run


def run_faruq_v3_acmc_finetune_control(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_summary: str | Path,
    d0_checkpoint: str | Path,
    acmc_summary: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    """Compare ACMC1 to an equally long D0-only continuation from the same D0."""
    if seed != 42:
        raise ValueError("D0FT control dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("D0FT control belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    grouped = load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki split test")
    d0_payload = _load_json(d0_summary, "D0 summary")
    acmc_payload = _load_json(acmc_summary, "ACMC1 summary")
    if int(d0_payload.get("seed", -1)) != seed or int(acmc_payload.get("seed", -1)) != seed:
        raise RuntimeError("D0 dan ACMC1 harus memakai seed 42")
    if d0_payload.get("test_images_accessed") is not False or acmc_payload.get("test_images_accessed") is not False:
        raise RuntimeError("Provenance test lock tidak valid")
    checkpoint_hash = _sha256_file(d0_checkpoint)
    if acmc_payload.get("d0_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("ACMC1 bukan cabang dari checkpoint D0 yang sama")

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    report, training_was_run = run_d0ft_continuation(
        data_root, output_root, d0_checkpoint, seed=seed, device=device
    )

    d0 = _metrics(d0_payload, "D0")
    d0ft = _metrics(report)
    acmc1 = _metrics(acmc_payload, "ACMC1")
    control_deltas = {name: d0ft[name] - d0[name] for name in METRICS}
    head_deltas = {name: acmc1[name] - d0ft[name] for name in METRICS}
    criteria = {
        "acmc_macro_gain_over_d0ft_at_least_0_5_point": head_deltas["macro_map50_95"] >= 0.005,
        "acmc_bottom3_not_lower_than_d0ft": head_deltas["bottom3_class_map50_95"] >= 0.0,
        "acmc_worst_drop_no_more_than_1_point_vs_d0ft": head_deltas["worst_class_map50_95"] >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    payload = {
        "protocol": "faruq-v3-acmc-optimization-control-v1",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": checkpoint_hash,
        "grouped_dataset": {
            "images_by_split": grouped["images_by_split"],
            "annotations_by_split": grouped["annotations_by_split"],
        },
        "results": {"D0": d0, "D0FT": d0ft, "ACMC1": acmc1},
        "control_deltas_d0ft_vs_d0": control_deltas,
        "head_deltas_acmc1_vs_d0ft": head_deltas,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "AUTHORIZE_PAIRED_THREE_SEED_ACMC_CONTROL" if decision == "PASS"
            else "STOP_ACMC1_EFFECT_NOT_ISOLATED"
        ),
        "training_executed_this_call": training_was_run,
    }
    summary = reports_root / "acmc1_optimization_control_seed42.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 ACMC1 D0 fine-tuning control")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--acmc-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_acmc_finetune_control(
        args.data_root, args.grouped_summary, args.d0_summary, args.d0_checkpoint,
        args.acmc_summary, args.output_root, seed=args.seed, device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
