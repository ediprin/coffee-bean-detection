"""Paired three-seed confirmation for ACMC2 entropy+margin gating on Faruq-v3.

Seed 42 is reused from the locked ACMC2 screening result. Seeds 123 and 2026
reuse the already-confirmed D0/D0FT/ACMC1 arms and train only ACMC2 from the
same D0 checkpoint used by the paired ACMC1 protocol. Evaluation is validation
only; the development dataset must not expose a test split.
"""

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
ALL_SEEDS = (42, 123, 2026)
CONFIRMATION_SEEDS = (123, 2026)


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


def _validate_acmc1_paired(path: str | Path) -> dict:
    payload = _load_json(path, "ACMC1 paired confirmation")
    if (
        payload.get("protocol") != "faruq-v3-acmc-paired-optimization-confirmation-v1"
        or tuple(int(seed) for seed in payload.get("seeds", ())) != ALL_SEEDS
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
        or payload.get("decision") != "PASS"
    ):
        raise RuntimeError("ACMC1 paired confirmation tidak kompatibel")
    per_seed = payload.get("per_seed", {})
    for seed in ALL_SEEDS:
        results = per_seed.get(str(seed), {}).get("results", {})
        for arm in ("D0", "D0FT", "ACMC1"):
            if arm not in results:
                raise RuntimeError(f"ACMC1 paired summary kehilangan {arm} seed {seed}")
            _metrics(results[arm])
    return payload


def _validate_seed42_screening(path: str | Path, paired: dict) -> dict:
    payload = _load_json(path, "ACMC2 seed42 screening")
    if (
        payload.get("protocol") != "faruq-v3-acmc2-entropy-margin-v1"
        or payload.get("stage") != "seed42_screening"
        or int(payload.get("seed", -1)) != 42
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
        or payload.get("decision") != "PASS"
    ):
        raise RuntimeError("ACMC2 seed42 screening bukan PASS validation-only yang kompatibel")
    results = payload.get("results", {})
    for arm in ("D0", "D0FT", "ACMC1", "ACMC2"):
        if arm not in results:
            raise RuntimeError(f"ACMC2 seed42 screening kehilangan arm {arm}")
        _metrics(results[arm])
    paired42 = paired["per_seed"]["42"]["results"]
    for arm in ("D0", "D0FT", "ACMC1"):
        for metric in METRICS:
            if abs(float(results[arm][metric]) - float(paired42[arm][metric])) > 1e-12:
                raise RuntimeError(f"Seed42 {arm} tidak identik antara screening ACMC2 dan paired ACMC1")
    return payload


def paired_confirmation_decision(per_seed: dict[str, dict]) -> tuple[dict, dict, str]:
    """Frozen ACMC2 progression gate, defined before seeds 123/2026 training."""
    aggregate: dict[str, dict[str, float | int]] = {}
    for metric in METRICS:
        d0_values = [float(per_seed[str(seed)]["results"]["D0"][metric]) for seed in ALL_SEEDS]
        d0ft_values = [float(per_seed[str(seed)]["results"]["D0FT"][metric]) for seed in ALL_SEEDS]
        acmc1_values = [float(per_seed[str(seed)]["results"]["ACMC1"][metric]) for seed in ALL_SEEDS]
        acmc2_values = [float(per_seed[str(seed)]["results"]["ACMC2"][metric]) for seed in ALL_SEEDS]
        vs_d0ft = [candidate - control for candidate, control in zip(acmc2_values, d0ft_values)]
        vs_acmc1 = [candidate - control for candidate, control in zip(acmc2_values, acmc1_values)]
        aggregate[metric] = {
            "d0_mean": sum(d0_values) / len(d0_values),
            "d0ft_mean": sum(d0ft_values) / len(d0ft_values),
            "acmc1_mean": sum(acmc1_values) / len(acmc1_values),
            "acmc2_mean": sum(acmc2_values) / len(acmc2_values),
            "acmc2_vs_d0ft_mean": sum(vs_d0ft) / len(vs_d0ft),
            "acmc2_vs_d0ft_min": min(vs_d0ft),
            "acmc2_vs_d0ft_improved_seeds": sum(delta > 0.0 for delta in vs_d0ft),
            "acmc2_vs_acmc1_mean": sum(vs_acmc1) / len(vs_acmc1),
            "acmc2_vs_acmc1_min": min(vs_acmc1),
            "acmc2_vs_acmc1_improved_seeds": sum(delta > 0.0 for delta in vs_acmc1),
        }

    macro = aggregate["macro_map50_95"]
    bottom3 = aggregate["bottom3_class_map50_95"]
    worst = aggregate["worst_class_map50_95"]
    criteria = {
        "macro_gain_over_d0ft_mean_at_least_0_5_point": macro["acmc2_vs_d0ft_mean"] >= 0.005,
        "macro_improved_over_d0ft_at_least_2_of_3": macro["acmc2_vs_d0ft_improved_seeds"] >= 2,
        "bottom3_mean_not_lower_than_d0ft": bottom3["acmc2_vs_d0ft_mean"] >= 0.0,
        "bottom3_improved_over_d0ft_at_least_2_of_3": bottom3["acmc2_vs_d0ft_improved_seeds"] >= 2,
        "worst_mean_drop_vs_d0ft_no_more_than_1_point": worst["acmc2_vs_d0ft_mean"] >= -0.01,
        "macro_mean_not_lower_than_acmc1": macro["acmc2_vs_acmc1_mean"] >= 0.0,
        "macro_improved_over_acmc1_at_least_2_of_3": macro["acmc2_vs_acmc1_improved_seeds"] >= 2,
        "at_least_one_tail_mean_improves_over_acmc1": max(
            bottom3["acmc2_vs_acmc1_mean"], worst["acmc2_vs_acmc1_mean"]
        ) > 0.0,
        "at_least_one_tail_improves_over_acmc1_at_least_2_of_3": max(
            bottom3["acmc2_vs_acmc1_improved_seeds"], worst["acmc2_vs_acmc1_improved_seeds"]
        ) >= 2,
        "neither_tail_mean_drops_more_than_1_point_vs_acmc1": min(
            bottom3["acmc2_vs_acmc1_mean"], worst["acmc2_vs_acmc1_mean"]
        ) >= -0.01,
    }
    decision = "PASS" if all(criteria.values()) else "FAIL"
    return aggregate, criteria, decision


def _train_acmc2_seed(
    data_root: Path,
    d0_checkpoint: Path,
    output_root: Path,
    static_root: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[dict[str, float], dict]:
    checkpoint_hash = _sha256_file(d0_checkpoint)
    config_payload = load_experiment(CONFIG)
    static_path = static_root / f"D0_seed{seed}_acmc2_static.json"
    static = static_ambiguity_multilevel_audit(
        MODEL_YAML,
        d0_checkpoint,
        static_path,
        nc=21,
        image_size=128,
        config=config_payload["ambiguity_multilevel"],
    )
    if static["decision"] != "PASS":
        raise RuntimeError(f"Static audit ACMC2 seed {seed} gagal: {static_path}")

    run_dir = output_root / f"ACMC2_seed{seed}"
    recover_completed_training_manifest(CONFIG, data_root, run_dir, seed, weights_override=d0_checkpoint)
    manifest = run_dir / "experiment_manifest.json"
    if manifest.is_file():
        provenance = _load_json(manifest, f"ACMC2 seed {seed} manifest")
        if provenance.get("weights_override_sha256") != checkpoint_hash:
            raise RuntimeError(f"ACMC2 seed {seed} memakai checkpoint D0 berbeda")
        if provenance.get("ambiguity_multilevel", {}).get("ambiguity_mode") != "entropy_margin":
            raise RuntimeError(f"ACMC2 seed {seed} bukan entropy_margin")

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
        output_root / "val_reports" / f"ACMC2_seed{seed}_val.json",
        split="val",
        device=device,
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError(f"Validation seed {seed} kehilangan kelas")
    return _metrics(report["metrics"]), {
        "d0_checkpoint": str(d0_checkpoint),
        "d0_checkpoint_sha256": checkpoint_hash,
        "static_audit": str(static_path),
        "acmc2_report": str(output_root / "val_reports" / f"ACMC2_seed{seed}_val.json"),
        "training_executed_this_call": training_was_run,
    }


def run_faruq_v3_acmc2_paired_confirmation(
    data_root: str | Path,
    grouped_summary: str | Path,
    acmc1_paired_summary: str | Path,
    acmc1_paired_root: str | Path,
    acmc2_seed42_summary: str | Path,
    output_root: str | Path,
    *,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    frozen_seeds = tuple(int(seed) for seed in seeds)
    if frozen_seeds != CONFIRMATION_SEEDS:
        raise ValueError(f"Konfirmasi ACMC2 dikunci pada seed {CONFIRMATION_SEEDS}")
    if not authorize_training:
        raise RuntimeError("Konfirmasi paired ACMC2 belum diotorisasi")

    data_root = Path(data_root).expanduser().resolve()
    acmc1_paired_root = Path(acmc1_paired_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    grouped = load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki split test")

    paired = _validate_acmc1_paired(acmc1_paired_summary)
    seed42 = _validate_seed42_screening(acmc2_seed42_summary, paired)
    reports_root = output_root / "val_reports"
    static_root = output_root / "static_audits"
    reports_root.mkdir(parents=True, exist_ok=True)
    static_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    per_seed: dict[str, dict] = {
        "42": {
            "source": str(Path(acmc2_seed42_summary).expanduser().resolve()),
            "results": {arm: _metrics(seed42["results"][arm]) for arm in ("D0", "D0FT", "ACMC1", "ACMC2")},
            "training_executed_this_call": False,
        }
    }

    for seed in frozen_seeds:
        print(f"\n=== PAIRED ACMC2 CONFIRMATION | SEED {seed} ===", flush=True)
        d0_checkpoint = acmc1_paired_root / "D0_base" / f"D0_seed{seed}" / "weights" / "best.pt"
        if not d0_checkpoint.is_file():
            raise FileNotFoundError(f"D0 paired seed {seed} tidak ditemukan: {d0_checkpoint}")
        candidate, provenance = _train_acmc2_seed(
            data_root, d0_checkpoint, output_root, static_root, seed=seed, device=device
        )
        source_results = paired["per_seed"][str(seed)]["results"]
        per_seed[str(seed)] = {
            "source": str(Path(acmc1_paired_summary).expanduser().resolve()),
            "results": {
                "D0": _metrics(source_results["D0"]),
                "D0FT": _metrics(source_results["D0FT"]),
                "ACMC1": _metrics(source_results["ACMC1"]),
                "ACMC2": candidate,
            },
            **provenance,
        }

    aggregate, criteria, decision = paired_confirmation_decision(per_seed)
    payload = {
        "protocol": "faruq-v3-acmc2-paired-optimization-confirmation-v1",
        "seeds": list(ALL_SEEDS),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "grouped_dataset": {
            "images_by_split": grouped["images_by_split"],
            "annotations_by_split": grouped["annotations_by_split"],
        },
        "acmc1_paired_summary": str(Path(acmc1_paired_summary).expanduser().resolve()),
        "acmc2_seed42_summary": str(Path(acmc2_seed42_summary).expanduser().resolve()),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "SELECT_ACMC2_FOR_SINGLE_LOCKED_TEST_EVALUATION"
            if decision == "PASS"
            else "KEEP_ACMC1_AS_SELECTED_MODEL"
        ),
    }
    summary = reports_root / "acmc2_paired_optimization_confirmation.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 paired ACMC2 optimization confirmation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--acmc1-paired-summary", required=True)
    parser.add_argument("--acmc1-paired-root", required=True)
    parser.add_argument("--acmc2-seed42-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIRMATION_SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_acmc2_paired_confirmation(
        args.data_root,
        args.grouped_summary,
        args.acmc1_paired_summary,
        args.acmc1_paired_root,
        args.acmc2_seed42_summary,
        args.output_root,
        seeds=tuple(args.seeds),
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
