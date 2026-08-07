from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.drnet_refinement import (
    DRNetInteractionConfig,
    build_entity_family_mapping,
    make_drnet_interaction_trainer,
)
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/drnet_refinement/DRIV1_yolo26n_entity_family_verification.yaml"
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


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def _load_config() -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    required = {"code", "model", "ontology", "drnet_interaction", "train"}
    if not required <= set(payload):
        raise ValueError("Config DRIV1 tidak lengkap")
    if payload["code"] != "DRIV1":
        raise ValueError("Config code harus DRIV1")
    DRNetInteractionConfig.from_mapping(payload["drnet_interaction"])
    return payload


def _gate(candidate: dict[str, float], drf1: dict[str, float], d0ft: dict[str, float]) -> dict:
    delta_drf1 = {name: candidate[name] - drf1[name] for name in METRICS}
    delta_d0ft = {name: candidate[name] - d0ft[name] for name in METRICS}
    criteria = {
        "iv_has_incremental_signal_over_drf1": (
            delta_drf1["macro_map50_95"] >= 0.002
            or delta_drf1["bottom3_class_map50_95"] >= 0.005
            or delta_drf1["worst_class_map50_95"] >= 0.005
        ),
        "macro_drop_vs_drf1_no_more_than_0_2_point": delta_drf1["macro_map50_95"] >= -0.002,
        "bottom3_drop_vs_drf1_no_more_than_1_point": delta_drf1["bottom3_class_map50_95"] >= -0.010,
        "worst_drop_vs_drf1_no_more_than_2_points": delta_drf1["worst_class_map50_95"] >= -0.020,
        "macro_not_below_d0ft_by_more_than_0_2_point": delta_d0ft["macro_map50_95"] >= -0.002,
    }
    return {
        "delta_DRIV1_vs_DRF1": delta_drf1,
        "delta_DRIV1_vs_D0FT": delta_d0ft,
        "criteria": criteria,
        "decision": "RETAIN" if all(criteria.values()) else "REJECT",
    }


def run_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    control_summary: str | Path,
    drnet_summary: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("DRIV1 discovery screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh memiliki split test")
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)

    control = _load_json(control_summary, "D0FT/ACMC1 control summary")
    previous = _load_json(drnet_summary, "DRF1/DRC1 summary")
    for payload, label in ((control, "control"), (previous, "DRNet")):
        if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
            raise RuntimeError(f"{label} summary tidak membuktikan test lock")
    checkpoint_hash = _sha256(d0_checkpoint)
    if control.get("d0_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("D0 checkpoint berbeda dari control summary")
    if previous.get("d0_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("D0 checkpoint berbeda dari DRNet predecessor summary")
    if "DRF1" not in previous.get("candidates", {}):
        raise RuntimeError("DRNet predecessor summary tidak memiliki DRF1")

    config = _load_config()
    ontology = (REPO_ROOT / config["ontology"]).resolve()
    data_yaml = yaml.safe_load((data_root / "data.yaml").read_text(encoding="utf-8")) or {}
    class_to_group, group_names, members = build_entity_family_mapping(data_yaml["names"], ontology)
    if len(class_to_group) != 21:
        raise RuntimeError("DRIV1 memerlukan 21 fine classes")

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    run_dir = output_root / f"DRIV1_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False
    if not best.is_file():
        from ultralytics import YOLO

        if last.is_file():
            model = YOLO(str(last))
            args = {"resume": True}
            if device is not None:
                args["device"] = device
            model.train(**args)
        else:
            model = YOLO(str(REPO_ROOT / config["model"]))
            train_args = dict(config["train"])
            train_args.update({
                "data": str(data_root / "data.yaml"),
                "project": str(output_root),
                "name": f"DRIV1_seed{seed}",
                "exist_ok": True,
                "seed": seed,
                "deterministic": True,
                "plots": True,
                "verbose": True,
            })
            if device is not None:
                train_args["device"] = device
            trainer = make_drnet_interaction_trainer(
                config["drnet_interaction"],
                ontology_path=ontology,
                d0_checkpoint=d0_checkpoint,
            )
            model.train(trainer=trainer, **train_args)
        training_executed = True

    if not best.is_file():
        raise FileNotFoundError(f"DRIV1 best.pt tidak ditemukan: {best}")
    report_path = reports_root / f"DRIV1_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation DRIV1 kehilangan kelas")
    driv1 = _metrics(report)
    d0ft = _metrics(control["results"]["D0FT"])
    acmc1 = _metrics(control["results"]["ACMC1"])
    drf1 = _metrics(previous["candidates"]["DRF1"])
    decision = _gate(driv1, drf1, d0ft)
    decision["delta_DRIV1_vs_ACMC1"] = {
        name: driv1[name] - acmc1[name] for name in METRICS
    }

    result = {
        "protocol": "faruq-v3-drnet-interaction-verification-discovery-v1",
        "stage": "breadth_discovery",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": checkpoint_hash,
        "official_code_operator": (
            "coarse argmax selects an allowed fine-class subset; fine scores outside that subset "
            "are suppressed before detector post-processing"
        ),
        "transfer_boundary": (
            "Official DRNet uses a two-stage RoI coarse bbox head plus fine classifier. DRIV1 instead "
            "adds a dense P3/P4/P5 coarse classifier to the existing YOLO26 DRF1 transfer, trains it "
            "on ontology-derived entity_family targets for positive one-to-many assignments, and applies "
            "the same coarse-to-fine score restriction on one-to-one inference logits."
        ),
        "coarse_taxonomy_source": str(ontology),
        "coarse_groups": list(group_names),
        "coarse_members": members,
        "class_to_group": list(class_to_group),
        "controls": {"D0FT": d0ft, "ACMC1": acmc1, "DRF1": drf1},
        "candidate": {
            "DRIV1": {
                "metrics": driv1,
                "checkpoint": str(best),
                "report": str(report_path),
                "training_executed_this_call": training_executed,
            }
        },
        "decision": decision,
    }
    summary = reports_root / "drnet_interaction_seed42_screening.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 DRNet Interaction Verification screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--drnet-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_screening(
        args.data_root,
        args.grouped_summary,
        args.control_summary,
        args.drnet_summary,
        args.d0_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
