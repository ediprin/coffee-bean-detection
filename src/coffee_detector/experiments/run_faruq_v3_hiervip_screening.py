"""Validation-only breadth screen for ExpertDet HierVIP on YOLO26."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.hiervip import HierVIPConfig, build_sni_hierarchy, make_hiervip_trainer


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/hiervip/HVIP1_yolo26n_sni_hierarchy.yaml"
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


def _metrics(payload: dict, preferred: str | None = None) -> dict[str, float]:
    source = payload.get("metrics", payload)
    if "results" in source and isinstance(source["results"], dict):
        results = source["results"]
        if preferred and preferred in results:
            source = results[preferred]
        else:
            for name in ("D0FT", "ACMC1", "D0"):
                if name in results:
                    source = results[name]
                    break
    return {name: float(source[name]) for name in METRICS}


def _load_config() -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != "HVIP1" or payload.get("variant") != "hiervip":
        raise RuntimeError("Config HVIP1 salah")
    config = HierVIPConfig.from_mapping(payload.get("hiervip"))
    paper_constants = {
        "temperature": 0.2,
        "loss_weight": 0.001,
        "momentum_low": 0.5,
        "momentum_base": 0.8,
        "drift_alpha": 0.2,
    }
    for key, value in paper_constants.items():
        if float(getattr(config, key)) != value:
            raise RuntimeError(f"HVIP1 {key} harus tetap {value}")
    return payload


def run_hiervip_screening(
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
        raise ValueError("HVIP1 discovery screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh memiliki test")
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)

    d0ft_payload = _load_json(d0ft_report, "D0FT report")
    acmc1_payload = _load_json(acmc1_report, "ACMC1 report")
    d0ft = _metrics(d0ft_payload, "D0FT")
    acmc1 = _metrics(acmc1_payload, "ACMC1")
    checkpoint_hash = _sha256(d0_checkpoint)

    config = _load_config()
    ontology = (REPO_ROOT / config["ontology"]).resolve()
    data_yaml = yaml.safe_load((data_root / "data.yaml").read_text(encoding="utf-8")) or {}
    hierarchy = build_sni_hierarchy(data_yaml["names"], ontology)
    if hierarchy.num_classes != 21:
        raise RuntimeError("HVIP1 memerlukan 21 fine classes")

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    run_name = f"HVIP1_seed{seed}"
    run_dir = output_root / run_name
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False
    if not best.is_file():
        from ultralytics import YOLO

        trainer = make_hiervip_trainer(
            config["hiervip"], ontology_path=ontology, d0_checkpoint=d0_checkpoint
        )
        if last.is_file():
            model = YOLO(str(last))
            train_args = {"resume": True}
            if device is not None:
                train_args["device"] = device
            model.train(**train_args)
        else:
            model = YOLO(str(REPO_ROOT / config["model"]))
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
            model.train(trainer=trainer, **train_args)
        training_executed = True

    if not best.is_file():
        raise FileNotFoundError(f"HVIP1 best.pt tidak ditemukan: {best}")
    report_path = reports_root / f"HVIP1_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation kehilangan kelas")
    hvip = _metrics(report)

    delta_d0ft = {name: hvip[name] - d0ft[name] for name in METRICS}
    delta_acmc1 = {name: hvip[name] - acmc1[name] for name in METRICS}
    discovery_signal = (
        delta_d0ft["macro_map50_95"] >= 0.002
        or delta_d0ft["bottom3_class_map50_95"] >= 0.005
        or delta_d0ft["worst_class_map50_95"] >= 0.005
    )
    safeguards = {
        "macro_drop_no_more_than_1_point": delta_d0ft["macro_map50_95"] >= -0.010,
        "bottom3_drop_no_more_than_2_points": delta_d0ft["bottom3_class_map50_95"] >= -0.020,
        "worst_drop_no_more_than_2_points": delta_d0ft["worst_class_map50_95"] >= -0.020,
    }
    decision = "RETAIN" if discovery_signal and all(safeguards.values()) else "REJECT"
    payload = {
        "protocol": "faruq-v3-expertdet-hiervip-search-v1",
        "stage": "breadth_discovery",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "candidate": "HVIP1",
        "paper_scope": "ExpertDet_HierVIP_only_Eqs_4_to_10",
        "excluded_component": "VMAM requires explicit per-class textual attribute supervision not present in this benchmark",
        "adaptation_boundary": (
            "ExpertDet extracts GT-box RRoI instance features. HVIP1 applies the same adaptive prototype-tree and HSC equations "
            "to 128-D embeddings of positively assigned YOLO26 P3/P4/P5 one-to-many locations. The hierarchy is frozen as "
            "fine_class -> primary_condition -> entity_family -> root from the SNI ontology; no validation confusion information is used."
        ),
        "embedding_dimension_note": "128 is a project-matched transfer choice for fair comparison with PCL/APCL; the paper does not specify this dimension in the retrieved method text.",
        "d0_checkpoint_sha256": checkpoint_hash,
        "hierarchy": hierarchy.to_dict(),
        "results": {"D0FT": d0ft, "ACMC1": acmc1, "HVIP1": hvip},
        "deltas": {"HVIP1_vs_D0FT": delta_d0ft, "HVIP1_vs_ACMC1": delta_acmc1},
        "criteria": {"discovery_signal": discovery_signal, **safeguards},
        "decision": decision,
        "next_action": "KEEP_HVIP1_IN_BREADTH_POOL" if decision == "RETAIN" else "ARCHIVE_HVIP1_SEED42",
        "training_executed_this_call": training_executed,
    }
    summary = reports_root / "hiervip_seed42_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 ExpertDet HierVIP breadth screen")
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
    result = run_hiervip_screening(
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
