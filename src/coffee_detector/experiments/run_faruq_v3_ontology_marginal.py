from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from coffee_detector.analysis.faruq_v3_diagnostics import run_faruq_v3_diagnostics
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.run_baseline import is_training_complete
from coffee_detector.train import recover_completed_training_manifest, train_experiment


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "C0": REPO_ROOT / "configs/structured_ontology/C0_yolo26n_identity_control.yaml",
    "S0": REPO_ROOT / "configs/structured_ontology/S0_yolo26n_semantic_marginal.yaml",
}


def _read_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _validate_prerequisites(
    data_root: Path,
    grouped_summary: str | Path,
    support_report: str | Path,
    static_audit: str | Path,
) -> None:
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 screening tidak boleh menyediakan test")
    load_faruq_grouped_summary(grouped_summary, data_root)
    support = _read_json(support_report, "Structured-target support report")
    if support.get("protocol") != "sni21-structured-target-support-audit-v1":
        raise ValueError("Protocol support report tidak cocok")
    if not support.get("statistically_ready"):
        raise RuntimeError("Target terstruktur belum didukung statistik")
    if support.get("test_images_accessed") is not False:
        raise RuntimeError("Support report tidak membuktikan test lock")
    if support.get("blocked_tasks") != ["physical_size_mm"]:
        raise RuntimeError("Blocked-task support report berubah")
    static = _read_json(static_audit, "Ontology static audit")
    if static.get("protocol") != "sni21-ontology-marginal-static-audit-v1":
        raise ValueError("Protocol static audit tidak cocok")
    if static.get("decision") != "PASS" or not all(static.get("gates", {}).values()):
        raise RuntimeError("Static gate ontology-marginal belum PASS")
    if static.get("training_executed") is not False or static.get("test_images_accessed") is not False:
        raise RuntimeError("Provenance static audit tidak valid")


def _metrics(evaluation: dict, diagnostic: dict) -> dict:
    metrics = evaluation["metrics"]
    global_diagnostic = diagnostic["global"]
    return {
        "macro_map50_95": float(metrics["macro_map50_95"]),
        "bottom3_class_map50_95": float(metrics["bottom3_class_map50_95"]),
        "worst_class_map50_95": float(metrics["worst_class_map50_95"]),
        "proposal_accessibility": float(global_diagnostic["proposal_accessibility"]),
        "conditional_top1_accuracy": float(
            global_diagnostic["localization_conditioned_class_accuracy"]
        ),
    }


def _checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_report(path: Path, fingerprint: str, kind: str) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("ontology_screening_provenance", {})
    if (
        provenance.get("checkpoint_sha256") != fingerprint
        or provenance.get("kind") != kind
        or payload.get("test_images_accessed") is not False
    ):
        return None
    return payload


def _persist_provenance(path: Path, payload: dict, fingerprint: str, kind: str) -> None:
    payload["ontology_screening_provenance"] = {
        "checkpoint_sha256": fingerprint,
        "kind": kind,
        "evaluation_split": "val",
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def compare_screening(candidate: dict, reference: dict, *, semantic_control: bool) -> dict:
    deltas = {key: candidate[key] - reference[key] for key in candidate}
    criteria = {
        "macro_gain_at_least_0_5_point": deltas["macro_map50_95"] >= 0.005,
        "bottom3_drop_no_more_than_1_point": deltas["bottom3_class_map50_95"] >= -0.01,
        "worst_drop_no_more_than_2_points": deltas["worst_class_map50_95"] >= -0.02,
        "proposal_drop_no_more_than_1_point": deltas["proposal_accessibility"] >= -0.01,
    }
    if semantic_control:
        criteria["conditional_top1_not_lower"] = deltas["conditional_top1_accuracy"] >= 0.0
    else:
        criteria["conditional_top1_gain_at_least_2_points"] = (
            deltas["conditional_top1_accuracy"] >= 0.02
        )
    return {
        "decision": "PASS" if all(criteria.values()) else "FAIL",
        "deltas": deltas,
        "criteria": criteria,
    }


def _evaluate_model(
    code: str,
    checkpoint: Path,
    data_root: Path,
    reports: Path,
    device: str | None,
) -> tuple[dict, dict, dict]:
    evaluation_path = reports / f"{code}_seed42_val.json"
    diagnostic_path = reports / f"{code}_seed42_diagnostic.json"
    fingerprint = _checkpoint_fingerprint(checkpoint)
    evaluation = _cached_report(evaluation_path, fingerprint, "evaluation")
    if evaluation is None:
        evaluation = evaluate(
            checkpoint, data_root, evaluation_path, split="val", device=device
        )
        evaluation["test_images_accessed"] = False
        _persist_provenance(evaluation_path, evaluation, fingerprint, "evaluation")
    else:
        print(f"REUSE EVALUATION: {code} seed 42", flush=True)
    if evaluation["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError(f"Validation {code} kehilangan kelas")
    diagnostic = _cached_report(diagnostic_path, fingerprint, "diagnostic")
    if diagnostic is None:
        diagnostic = run_faruq_v3_diagnostics(
            checkpoint,
            data_root,
            diagnostic_path,
            split="val",
            device=device or "cpu",
        )
        _persist_provenance(diagnostic_path, diagnostic, fingerprint, "diagnostic")
    else:
        print(f"REUSE DIAGNOSTIC: {code} seed 42", flush=True)
    return evaluation, diagnostic, _metrics(evaluation, diagnostic)


def run_faruq_v3_ontology_marginal(
    data_root: str | Path,
    grouped_summary: str | Path,
    support_report: str | Path,
    static_audit: str | Path,
    baseline_root: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    if seed != 42:
        raise RuntimeError("Screening v1 dikunci hanya untuk seed 42")
    data_root = Path(data_root).expanduser().resolve()
    baseline_root = Path(baseline_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    _validate_prerequisites(data_root, grouped_summary, support_report, static_audit)
    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)

    baseline_checkpoint = baseline_root / "D0_seed42/weights/best.pt"
    if not baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint D0 tidak ditemukan: {baseline_checkpoint}")
    _, _, baseline_metrics = _evaluate_model(
        "D0", baseline_checkpoint, data_root, reports, device
    )

    model_metrics = {"D0": baseline_metrics}
    training_actions = {}
    for code, config in CONFIGS.items():
        run_dir = output_root / f"{code}_seed42"
        recover_completed_training_manifest(config, data_root, run_dir, seed)
        complete = is_training_complete(run_dir)
        training_actions[code] = "reuse" if complete else "train_or_resume"
        if not complete:
            print(f"TRAIN/RESUME {code} seed 42", flush=True)
            train_experiment(
                config,
                data_root,
                output_root,
                seed,
                device=device,
                resume=True,
            )
        checkpoint = run_dir / "weights/best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint {code} belum lengkap: {checkpoint}")
        _, _, model_metrics[code] = _evaluate_model(
            code, checkpoint, data_root, reports, device
        )

    comparisons = {
        "S0_vs_D0": compare_screening(
            model_metrics["S0"], model_metrics["D0"], semantic_control=False
        ),
        "S0_vs_C0": compare_screening(
            model_metrics["S0"], model_metrics["C0"], semantic_control=True
        ),
    }
    final = "PASS" if all(row["decision"] == "PASS" for row in comparisons.values()) else "FAIL"
    payload = {
        "protocol": "faruq-v3-ontology-marginal-screening-v1",
        "seed": 42,
        "evaluation_split": "val",
        "models": model_metrics,
        "comparisons": comparisons,
        "decision": final,
        "training_actions": training_actions,
        "test_images_accessed": False,
        "test_opened": False,
        "next_action": (
            "request_review_before_three_seed_confirmation"
            if final == "PASS"
            else "stop_ontology_marginal_without_test_or_extra_seeds"
        ),
    }
    destination = reports / "screening_seed42.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 ontology-marginal seed-42 screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--support-report", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = run_faruq_v3_ontology_marginal(
        args.data_root,
        args.grouped_summary,
        args.support_report,
        args.static_audit,
        args.baseline_root,
        args.output_root,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
