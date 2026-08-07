"""Validation-only broad-search screening for PCLDet learned prototypes on YOLO26."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.pcl import make_pcl_trainer


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/pcl/PCL1_yolo26n_learned_prototype.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
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


def _metrics(payload: dict, preferred_arm: str | None = None) -> dict[str, float]:
    source = payload.get("metrics", payload)
    if "results" in source and isinstance(source["results"], dict):
        results = source["results"]
        if preferred_arm and preferred_arm in results:
            source = results[preferred_arm]
        else:
            for name in ("ACMC1", "D0FT", "D0"):
                if name in results:
                    source = results[name]
                    break
    return {name: float(source[name]) for name in METRICS}


def _load_config() -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("variant") != "pcl":
        raise RuntimeError("Config PCL1 memiliki variant yang salah")
    if not isinstance(payload.get("pcl"), dict):
        raise RuntimeError("Config PCL1 tidak memiliki mapping pcl")
    if float(payload["pcl"].get("temperature", -1)) != 1.0 / 32.0:
        raise RuntimeError("PCL1 temperature harus tetap 1/32 sesuai PCLDet")
    return payload


def run_faruq_v3_pcl_screening(
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
        raise ValueError("PCL1 discovery screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki split test")
    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)

    d0ft = _metrics(_load_json(d0ft_report, "D0FT report"), "D0FT")
    acmc1 = _metrics(_load_json(acmc1_report, "ACMC1 report"), "ACMC1")
    checkpoint_hash = _sha256_file(d0_checkpoint)

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(
        data_root, reports_root / "dataset_audit.json", near_threshold=-1
    )
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    config = _load_config()
    run_name = f"PCL1_seed{seed}"
    run_dir = output_root / run_name
    best_checkpoint = run_dir / "weights/best.pt"
    last_checkpoint = run_dir / "weights/last.pt"
    training_executed_this_call = False

    if not best_checkpoint.is_file():
        from ultralytics import YOLO

        trainer = make_pcl_trainer(config["pcl"], d0_checkpoint=d0_checkpoint)
        if last_checkpoint.is_file():
            model = YOLO(str(last_checkpoint))
            train_args = {"resume": True}
            if device is not None:
                train_args["device"] = device
        else:
            model = YOLO(str(MODEL_YAML))
            model.load(str(d0_checkpoint))
            train_args = dict(config["train"])
            train_args.update(
                {
                    "data": str(data_root / "data.yaml"),
                    "project": str(output_root),
                    "name": run_name,
                    "exist_ok": True,
                    "seed": int(seed),
                    "deterministic": True,
                    "plots": True,
                    "verbose": True,
                    "pretrained": True,
                }
            )
            if device is not None:
                train_args["device"] = device
        print(
            f"{'RESUME' if last_checkpoint.is_file() else 'START'} PCL1 | "
            f"learned prototype ProtoCL | seed={seed}",
            flush=True,
        )
        model.train(trainer=trainer, **train_args)
        training_executed_this_call = True

    if not best_checkpoint.is_file():
        raise FileNotFoundError(f"PCL1 best.pt tidak ditemukan: {best_checkpoint}")

    report = evaluate(
        best_checkpoint,
        data_root,
        reports_root / f"PCL1_seed{seed}_val.json",
        split="val",
        device=device,
    )
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError("Validation kehilangan kelas")
    pcl1 = _metrics(report)

    versus_d0ft = {name: pcl1[name] - d0ft[name] for name in METRICS}
    versus_acmc1 = {name: pcl1[name] - acmc1[name] for name in METRICS}
    discovery_signal = (
        versus_d0ft["macro_map50_95"] >= 0.002
        or versus_d0ft["bottom3_class_map50_95"] >= 0.005
        or versus_d0ft["worst_class_map50_95"] >= 0.005
    )
    safeguards = {
        "macro_drop_no_more_than_1_point": versus_d0ft["macro_map50_95"] >= -0.010,
        "bottom3_drop_no_more_than_2_points": versus_d0ft["bottom3_class_map50_95"] >= -0.020,
        "worst_drop_no_more_than_2_points": versus_d0ft["worst_class_map50_95"] >= -0.020,
    }
    decision = "RETAIN" if discovery_signal and all(safeguards.values()) else "REJECT"

    payload = {
        "protocol": "faruq-v3-pcldet-learned-prototype-search-v1",
        "stage": "broad_search_screening",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "candidate": "PCL1",
        "mechanistic_hypothesis": "fine_grained_embedding_overlap_limits_leaf_classification",
        "paper_operator": "PCLDet_Eqs_1_3_4",
        "paper_temperature": 1.0 / 32.0,
        "adaptation_boundary": (
            "PCLDet learned prototype bank and ProtoCL are applied to positive YOLO26 one-to-many "
            "dense assignments through the same 128-D P3/P4/P5 projection layout used by the APCL "
            "screen. The original ReDet/RPN and class-balanced sampler are not reproduced. CBS is "
            "excluded because the coffee benchmark is approximately balanced and YOLO26 has no RPN. "
            "The auxiliary branch is training-only; inference remains native YOLO26."
        ),
        "prototype_initialization_note": (
            "paper specifies normal-distribution initialization but not its standard deviation; "
            "this implementation predeclares standard-normal directions (std=1.0)"
        ),
        "d0_checkpoint_sha256": checkpoint_hash,
        "results": {"D0FT": d0ft, "ACMC1": acmc1, "PCL1": pcl1},
        "deltas": {"PCL1_vs_D0FT": versus_d0ft, "PCL1_vs_ACMC1": versus_acmc1},
        "criteria": {"discovery_signal": discovery_signal, **safeguards},
        "decision": decision,
        "next_action": (
            "KEEP_PCL1_AS_PREDECESSOR_CONTROL_FOR_APCL"
            if decision == "RETAIN"
            else "ARCHIVE_PCL1_BUT_KEEP_AS_APCL_METHOD_CONTROL"
        ),
        "training_executed_this_call": training_executed_this_call,
    }
    summary = reports_root / "pcl1_seed42_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 PCL1 broad-search screening")
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
    result = run_faruq_v3_pcl_screening(
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
