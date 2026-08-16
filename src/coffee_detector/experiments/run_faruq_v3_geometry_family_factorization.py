"""Exploratory exact-capacity shared-vs-family geometry validation on Faruq-v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.dataset import discover_layout
from coffee_detector.evaluate import evaluate
from coffee_detector.geometry_factorization.audit import static_geometry_factorization_audit
from coffee_detector.geometry_factorization.model import (
    FAMILIES,
    GeometryFactorizationConfig,
)
from coffee_detector.geometry_factorization.trainer import make_geometry_factorization_trainer
from coffee_detector.experiments.run_faruq_v3_geometry_conditioning_paired_confirmation import (
    _checkpoint_state,
    _load_json,
    _metrics,
    _run_complete,
    _sha256,
    _size_mean,
    _training_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIG = REPO_ROOT / "configs/geometry_factorization/GEO_family_exact_capacity.yaml"
SEEDS = (42, 123, 2026)
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
PROTOCOL = "faruq-v3-geometry-family-factorization-v1"
SOURCE_PROTOCOL = "faruq-v3-geometry-conditioning-paired-confirmation-v1"
DECOMP_PROTOCOL = "faruq-v3-geometry-family-effect-decomposition-v1"


def _d0_checkpoint(project_root: Path, seed: int) -> Path:
    if seed == 42:
        return project_root / "experiments/faruq-v3-yolo26n-baseline-v1/D0_seed42/weights/best.pt"
    return (
        project_root
        / "experiments/faruq-v3-acmc-paired-confirmation-v1"
        / "D0_base"
        / f"D0_seed{seed}"
        / "weights/best.pt"
    )


def _d0ft_artifacts(project_root: Path, seed: int) -> tuple[Path, Path]:
    if seed == 42:
        root = project_root / "experiments/faruq-v3-acmc-optimization-control-v1"
        return (
            root / "D0FT_seed42/experiment_manifest.json",
            root / "val_reports/D0FT_seed42_val.json",
        )
    root = project_root / "experiments/faruq-v3-acmc-paired-confirmation-v1/D0FT"
    return (
        root / f"D0FT_seed{seed}/experiment_manifest.json",
        root / "val_reports" / f"D0FT_seed{seed}_val.json",
    )


def _validate_sources(confirmation: Path, decomposition: Path) -> tuple[dict, dict]:
    source = _load_json(confirmation, "GEO three-seed confirmation")
    if (
        source.get("protocol") != SOURCE_PROTOCOL
        or tuple(int(seed) for seed in source.get("seeds", ())) != SEEDS
        or source.get("evaluation_split") != "val"
        or source.get("test_images_accessed") is not False
        or source.get("test_opened") is not False
        or source.get("decision") != "PASS"
    ):
        raise RuntimeError("Source GEO confirmation tidak kompatibel")
    decomposition_payload = _load_json(decomposition, "Family decomposition")
    if (
        decomposition_payload.get("protocol") != DECOMP_PROTOCOL
        or tuple(int(seed) for seed in decomposition_payload.get("seeds", ())) != SEEDS
        or decomposition_payload.get("evaluation_split") != "val"
        or decomposition_payload.get("test_images_accessed") is not False
        or decomposition_payload.get("test_opened") is not False
        or decomposition_payload.get("analysis_status")
        != "posthoc_descriptive_decomposition_no_new_gate"
    ):
        raise RuntimeError("Family decomposition tidak kompatibel")
    return source, decomposition_payload


def _verify_seed_provenance(project_root: Path, seed: int) -> dict:
    d0 = _d0_checkpoint(project_root, seed)
    manifest_path, d0ft_report = _d0ft_artifacts(project_root, seed)
    for path in (d0, manifest_path, d0ft_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    d0_hash = _sha256(d0)
    manifest = _load_json(manifest_path, f"D0FT manifest seed{seed}")
    if manifest.get("weights_override_sha256") != d0_hash:
        raise RuntimeError(f"D0FT seed{seed} bukan continuation exact D0")
    report = _load_json(d0ft_report, f"D0FT report seed{seed}")
    if report.get("split") != "val":
        raise RuntimeError(f"D0FT seed{seed} bukan validation")
    _metrics(report)
    return {
        "d0_checkpoint": str(d0),
        "d0_checkpoint_sha256": d0_hash,
        "d0ft_manifest": str(manifest_path),
        "d0ft_report": str(d0ft_report),
    }


def _family_means(size_by_class: dict[str, float]) -> dict[str, float]:
    result = {}
    for family in FAMILIES:
        values = [
            float(value)
            for name, value in size_by_class.items()
            if str(name).startswith(family + "_")
        ]
        if len(values) != 3:
            raise RuntimeError(f"Family {family} harus memiliki 3 size classes, got={len(values)}")
        result[family] = float(np.mean(values))
    return result


def _train_arm(
    *,
    arm: str,
    mode: str,
    seed: int,
    d0_checkpoint: Path,
    data_root: Path,
    output_root: Path,
    config: GeometryFactorizationConfig,
    train_args: dict,
    device: str | None,
) -> tuple[Path, bool]:
    from ultralytics import YOLO

    epochs = int(train_args["epochs"])
    run_dir = output_root / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    executed = False
    if not _run_complete(run_dir, epochs):
        trainer = make_geometry_factorization_trainer(
            config, d0_checkpoint=d0_checkpoint, mode=mode
        )
        epoch, resumable = _checkpoint_state(last)
        with _training_lock(output_root, arm, seed):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {arm} seed{seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True}
            else:
                print(f"START {arm} seed{seed} dari exact D0", flush=True)
                model = YOLO(str(MODEL_YAML))
                args = dict(train_args)
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root),
                    name=f"{arm}_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=True,
                    verbose=True,
                )
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        executed = True
    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run {arm} seed{seed} belum lengkap: {run_dir}")
    return best, executed


def _evaluate_arm(
    arm: str,
    seed: int,
    checkpoint: Path,
    data_root: Path,
    reports: Path,
    device: str | None,
) -> dict:
    report = evaluate(
        checkpoint,
        data_root,
        reports / f"{arm}_seed{seed}_val.json",
        split="val",
        device=device,
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError(f"Validation {arm} seed{seed} kehilangan kelas")
    row = _metrics(report)
    size_mean, by_class = _size_mean(report)
    row["size_class_mean_map50_95"] = size_mean
    row["size_map50_95_by_class"] = by_class
    row["family_mean_map50_95"] = _family_means(by_class)
    row["checkpoint"] = str(checkpoint)
    return row


def _aggregate(per_seed: dict[str, dict]) -> tuple[dict, dict, dict]:
    metric_names = (*METRICS, "size_class_mean_map50_95")
    aggregate = {}
    for metric in metric_names:
        deltas = [
            float(per_seed[str(seed)]["fam_minus_shared"][metric]) for seed in SEEDS
        ]
        aggregate[metric] = {
            "delta_mean": float(np.mean(deltas)),
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "improved_seeds": sum(value > 0.0 for value in deltas),
            "per_seed_deltas": {str(seed): deltas[i] for i, seed in enumerate(SEEDS)},
        }
    family_aggregate = {}
    for family in FAMILIES:
        deltas = [
            float(per_seed[str(seed)]["family_deltas"][family]) for seed in SEEDS
        ]
        family_aggregate[family] = {
            "delta_mean": float(np.mean(deltas)),
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "improved_seeds": sum(value > 0.0 for value in deltas),
            "per_seed_deltas": {str(seed): deltas[i] for i, seed in enumerate(SEEDS)},
        }
    tail_best = max(
        aggregate["bottom3_class_map50_95"]["delta_mean"],
        aggregate["worst_class_map50_95"]["delta_mean"],
    )
    criteria = {
        "macro_mean_gain_at_least_0_2_point": aggregate["macro_map50_95"]["delta_mean"] >= 0.002,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_not_lower": aggregate["bottom3_class_map50_95"]["delta_mean"] >= 0.0,
        "worst_mean_not_lower": aggregate["worst_class_map50_95"]["delta_mean"] >= 0.0,
        "size_mean_gain_at_least_0_5_point": aggregate["size_class_mean_map50_95"]["delta_mean"] >= 0.005,
        "size_mean_improved_at_least_2_of_3": aggregate["size_class_mean_map50_95"]["improved_seeds"] >= 2,
        "at_least_one_tail_mean_gain_at_least_0_5_point": tail_best >= 0.005,
        "kulit_kopi_family_mean_gain_at_least_0_5_point": family_aggregate["kulit_kopi"]["delta_mean"] >= 0.005,
        "kulit_tanduk_family_mean_drop_no_more_than_0_5_point": family_aggregate["kulit_tanduk"]["delta_mean"] >= -0.005,
    }
    return aggregate, family_aggregate, criteria


def run_family_factorization(
    data_root: str | Path,
    project_root: str | Path,
    confirmation_summary: str | Path,
    family_decomposition: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if stage not in {"static", "train"}:
        raise ValueError(stage)
    data_root = Path(data_root).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    confirmation_summary = Path(confirmation_summary).expanduser().resolve()
    family_decomposition = Path(family_decomposition).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    layout = discover_layout(data_root)
    _validate_sources(confirmation_summary, family_decomposition)

    provenance = {str(seed): _verify_seed_provenance(project_root, seed) for seed in SEEDS}
    static_root = output_root / "static_audits"
    static_root.mkdir(parents=True, exist_ok=True)

    if stage == "static":
        audits = {}
        for seed in SEEDS:
            audit_path = static_root / f"seed{seed}_geometry_factorization_static.json"
            audit = static_geometry_factorization_audit(
                MODEL_YAML,
                provenance[str(seed)]["d0_checkpoint"],
                layout.names,
                audit_path,
                nc=len(layout.names),
            )
            audits[str(seed)] = audit
        gates = {
            "all_static_audits_pass": all(audit["decision"] == "PASS" for audit in audits.values()),
            "all_shared_added_parameters_849": all(
                audit["models"]["GEO-SHARED60"]["added_parameters"] == 849 for audit in audits.values()
            ),
            "all_family_added_parameters_849": all(
                audit["models"]["GEO-FAM35x3"]["added_parameters"] == 849 for audit in audits.values()
            ),
        }
        payload = {
            "protocol": "faruq-v3-geometry-family-factorization-preflight-v1",
            "seeds": list(SEEDS),
            "evaluation_split": "val",
            "training_executed": False,
            "test_images_accessed": False,
            "test_opened": False,
            "provenance": provenance,
            "audits": {seed: audit["summary"] for seed, audit in audits.items()},
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
        destination = output_root / "static_preflight.json"
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["summary"] = str(destination)
        return payload

    if not authorize_training:
        raise RuntimeError("Training family factorization belum diotorisasi")
    preflight = _load_json(output_root / "static_preflight.json", "Static preflight")
    if preflight.get("decision") != "PASS":
        raise RuntimeError("Static family-factorization preflight belum PASS")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    config = GeometryFactorizationConfig.from_mapping(payload["geometry_factorization"])
    train_args = dict(payload["train"])
    per_seed = {}
    training_executed = {}

    for seed in SEEDS:
        d0_checkpoint = Path(provenance[str(seed)]["d0_checkpoint"])
        d0ft_report = _load_json(provenance[str(seed)]["d0ft_report"], f"D0FT seed{seed}")
        results = {"D0FT": _metrics(d0ft_report)}
        for arm, mode in (("GEO-SHARED60", "shared60"), ("GEO-FAM35x3", "family35x3")):
            best, executed = _train_arm(
                arm=arm,
                mode=mode,
                seed=seed,
                d0_checkpoint=d0_checkpoint,
                data_root=data_root,
                output_root=output_root,
                config=config,
                train_args=train_args,
                device=device,
            )
            results[arm] = _evaluate_arm(arm, seed, best, data_root, reports, device)
            training_executed[f"{arm}_seed{seed}"] = executed
        shared, family = results["GEO-SHARED60"], results["GEO-FAM35x3"]
        deltas = {metric: family[metric] - shared[metric] for metric in METRICS}
        deltas["size_class_mean_map50_95"] = (
            family["size_class_mean_map50_95"] - shared["size_class_mean_map50_95"]
        )
        family_deltas = {
            fam: family["family_mean_map50_95"][fam] - shared["family_mean_map50_95"][fam]
            for fam in FAMILIES
        }
        per_seed[str(seed)] = {
            "results": results,
            "fam_minus_shared": deltas,
            "family_deltas": family_deltas,
        }

    aggregate, family_aggregate, criteria = _aggregate(per_seed)
    retain = all(criteria.values())
    result = {
        "protocol": PROTOCOL,
        "scientific_status": "exploratory_architecture_validation_not_independent_confirmation",
        "seeds": list(SEEDS),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "source_confirmation": str(confirmation_summary),
        "source_family_decomposition": str(family_decomposition),
        "parameter_match": {"GEO-SHARED60_added": 849, "GEO-FAM35x3_added": 849},
        "per_seed": per_seed,
        "aggregate": aggregate,
        "family_aggregate": family_aggregate,
        "criteria": criteria,
        "decision": "RETAIN" if retain else "REJECT",
        "next_action": (
            "RETAIN_FAMILY_FACTORIZATION_FOR_FINAL_STAGE_REVIEW"
            if retain
            else "KEEP_SHARED_GEOMETRY_STRUCTURE"
        ),
        "training_executed_this_call": training_executed,
    }
    summary = reports / "geometry_family_factorization_three_seed.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact-capacity GEO shared vs family-factorized validation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--confirmation-summary", required=True)
    parser.add_argument("--family-decomposition", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("static", "train"), required=True)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_family_factorization(
        args.data_root,
        args.project_root,
        args.confirmation_summary,
        args.family_decomposition,
        args.output_root,
        stage=args.stage,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
