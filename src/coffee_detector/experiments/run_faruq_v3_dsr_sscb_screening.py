"""Validation-only breadth screening for DSRDet-inspired SSCB transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.dsr_sscb import make_sscb_trainer
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIGS = {
    "M0": REPO_ROOT / "configs/dsr_sscb/M0_msda.yaml",
    "S0": REPO_ROOT / "configs/dsr_sscb/S0_semantic_aux_msda.yaml",
    "S1": REPO_ROOT / "configs/dsr_sscb/S1_calibrated_sscb.yaml",
}
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


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    if "results" in source and isinstance(source["results"], dict):
        for name in ("ACMC1", "D0FT", "D0"):
            if name in source["results"]:
                source = source["results"][name]
                break
    return {name: float(source[name]) for name in METRICS}


def _load_config(arm: str) -> dict:
    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    if payload.get("arm") != arm:
        raise RuntimeError(f"Config {arm} memiliki arm yang salah")
    if not isinstance(payload.get("sscb"), dict) or not isinstance(payload.get("train"), dict):
        raise RuntimeError(f"Config {arm} tidak lengkap")
    return payload


def _delta(candidate: dict[str, float], control: dict[str, float]) -> dict[str, float]:
    return {name: candidate[name] - control[name] for name in METRICS}


def _decision(delta_vs_d0ft: dict[str, float]) -> dict:
    signal = (
        delta_vs_d0ft["macro_map50_95"] >= 0.002
        or delta_vs_d0ft["bottom3_class_map50_95"] >= 0.005
        or delta_vs_d0ft["worst_class_map50_95"] >= 0.005
    )
    safeguards = {
        "macro_drop_no_more_than_1_point": delta_vs_d0ft["macro_map50_95"] >= -0.010,
        "bottom3_drop_no_more_than_2_points": delta_vs_d0ft["bottom3_class_map50_95"] >= -0.020,
        "worst_drop_no_more_than_2_points": delta_vs_d0ft["worst_class_map50_95"] >= -0.020,
    }
    keep = signal and all(safeguards.values())
    return {"decision": "RETAIN" if keep else "REJECT", "discovery_signal": signal, **safeguards}


def run_faruq_v3_dsr_sscb_screening(
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
        raise ValueError("DSR-SSCB breadth screening dikunci untuk seed 42")
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

    d0ft = _metrics(_load_json(d0ft_report, "D0FT report"))
    acmc1 = _metrics(_load_json(acmc1_report, "ACMC1 report"))
    checkpoint_hash = _sha256_file(d0_checkpoint)

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    results: dict[str, dict[str, float]] = {"D0FT": d0ft, "ACMC1": acmc1}
    training_executed: dict[str, bool] = {}
    configs: dict[str, dict] = {}

    for arm in ("M0", "S0", "S1"):
        config = _load_config(arm)
        configs[arm] = config
        run_name = f"{arm}_seed{seed}"
        run_dir = output_root / run_name
        best = run_dir / "weights/best.pt"
        last = run_dir / "weights/last.pt"
        executed = False

        if not best.is_file():
            from ultralytics import YOLO

            trainer = make_sscb_trainer(config["sscb"], d0_checkpoint=d0_checkpoint)
            if last.is_file():
                model = YOLO(str(last))
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
            print(f"{'RESUME' if last.is_file() else 'START'} {arm} | mode={config['sscb']['mode']} | seed={seed}", flush=True)
            model.train(trainer=trainer, **train_args)
            executed = True

        if not best.is_file():
            raise FileNotFoundError(f"{arm} best.pt tidak ditemukan: {best}")
        report = evaluate(best, data_root, reports_root / f"{arm}_seed{seed}_val.json", split="val", device=device)
        if report["metrics"].get("classes_without_ground_truth", []):
            raise RuntimeError(f"Validation {arm} kehilangan kelas")
        results[arm] = _metrics(report)
        training_executed[arm] = executed

    deltas = {
        arm: {
            "vs_D0FT": _delta(results[arm], d0ft),
            "vs_ACMC1": _delta(results[arm], acmc1),
        }
        for arm in ("M0", "S0", "S1")
    }
    attribution = {
        "S0_minus_M0": _delta(results["S0"], results["M0"]),
        "S1_minus_S0": _delta(results["S1"], results["S0"]),
        "S1_minus_M0": _delta(results["S1"], results["M0"]),
    }
    decisions = {arm: _decision(deltas[arm]["vs_D0FT"]) for arm in ("M0", "S0", "S1")}

    payload = {
        "protocol": "faruq-v3-dsr-sscb-bbox-transfer-v1",
        "stage": "broad_search_screening",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "paper_mechanism": "DSRDet_SSCB_MSDA_Eqs_16_to_19",
        "adaptation_boundary": (
            "DSRDet CLIP-attention semantic labels are replaced by train-bbox foreground masks; "
            "YOLO26 localization and TAL remain native; SSCB affects residual classification only"
        ),
        "transfer_choices": {
            "semantic_supervisor": "bbox_union_foreground_BCE",
            "hidden_dim": 64,
            "sampling_points_per_level": 4,
            "max_offset_pixels": 2.0,
            "semantic_aux_weight": 0.2,
            "sampler": "torch_grid_sample_bilinear",
        },
        "d0_checkpoint_sha256": checkpoint_hash,
        "results": results,
        "deltas": deltas,
        "attribution": attribution,
        "decisions": decisions,
        "training_executed_this_call": training_executed,
    }
    summary = reports_root / "dsr_sscb_seed42_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 DSR SSCB breadth screening")
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
    result = run_faruq_v3_dsr_sscb_screening(
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
