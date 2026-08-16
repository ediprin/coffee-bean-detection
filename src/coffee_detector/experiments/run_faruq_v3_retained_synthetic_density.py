"""Evaluate every retained Faruq-v3 model on the frozen synthetic ladder.

Custom checkpoints were produced on several research branches.  The parent
process therefore resolves artifacts and launches one isolated worker per
model with the matching checkout on ``PYTHONPATH``.  No training is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import yaml


ARMS = tuple(f"B{i}_balanced_mild" for i in range(4))
MODEL_SPECS = {
    "D0FT": ("agent/af2-igem-paired-confirmation", "faruq-v3-acmc-optimization-control-v1/D0FT_seed42/weights/best.pt"),
    "AF1": ("agent/af2-igem-paired-confirmation", "faruq-v3-breadth-screening-batch-v1/candidates/AFAB/AF1_seed42/weights/best.pt"),
    "AF2": ("agent/af2-igem-paired-confirmation", "faruq-v3-breadth-screening-batch-v1/candidates/AFAB/AF2_seed42/weights/best.pt"),
    "IGEM1": ("agent/af2-igem-paired-confirmation", "faruq-v3-breadth-screening-batch-v1/candidates/IGEM/IGEM1_seed42/weights/best.pt"),
    "STB1": ("agent/af2-igem-paired-confirmation", "faruq-v3-breadth-screening-batch-v1/candidates/STB1/STB1_seed42/weights/best.pt"),
    "SAF1": ("agent/safpn-classification-alignment", "faruq-v3-breadth-screening-batch-v1/candidates/SAF1/SAF1_seed42/weights/best.pt"),
    "LPS1": ("agent/leaf-preserving-semantic-screening", "faruq-v3-breadth-screening-batch-v1/candidates/SEMAUX/LPS1_seed42/weights/best.pt"),
    "CPE0": ("agent/circle-cpe-screening", "faruq-v3-breadth-screening-batch-v1/candidates/CPE/CPE0_seed42/weights/best.pt"),
    "CPE7": ("agent/circle-cpe-screening", "faruq-v3-breadth-screening-batch-v1/candidates/CPE/CPE7_seed42/weights/best.pt"),
    "ACMC1": ("agent/af2-igem-paired-confirmation", "faruq-v3-acmc-one-stage-v1/ACMC1_seed42/weights/best.pt"),
    "GEO1": ("agent/circle-cpe-screening", "faruq-v3-geometry-conditioning-screening-v1/GEO1_seed42/weights/best.pt"),
}


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_checkpoint(project_root: Path, relative_path: str) -> Path:
    checkpoint = project_root / "experiments" / relative_path
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    return checkpoint


def _checkout(repo_root: Path, branch: str, repository: str) -> Path:
    destination = repo_root / branch.replace("/", "__")
    if (destination / ".git").is_dir():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "git", "clone", "--depth", "1", "--branch", branch,
        repository, str(destination),
    ]
    subprocess.run(command, check=True)
    return destination


def _runtime_yaml(arm_root: Path, output: Path) -> Path:
    from coffee_detector.dataset import discover_layout
    from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES

    layout = discover_layout(arm_root)
    expected = {index: name for index, name in enumerate(SNI21_CLASSES)}
    if layout.names != expected or "val" not in layout.splits:
        raise RuntimeError(f"Arm sintetis tidak valid: {arm_root}")
    payload = yaml.safe_load(layout.yaml_path.read_text(encoding="utf-8")) or {}
    relative = layout.splits["val"][0].relative_to(layout.root).as_posix()
    payload.update(
        {"path": str(layout.root), "names": expected, "train": relative,
         "val": relative, "test": relative}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def _worker(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    from coffee_detector.run_sni21_density_evaluation import _metric_summary
    from coffee_detector.dataset import discover_layout

    checkpoint = Path(args.checkpoint).resolve()
    arm_root = Path(args.arm_root).resolve()
    output = Path(args.output).resolve()
    checkpoint_hash = _sha256(checkpoint)
    manifest_hash = _sha256(arm_root / "metadata/generation_manifest.json")
    if output.is_file():
        cached = _load(output)
        if (
            cached.get("checkpoint_sha256") == checkpoint_hash
            and cached.get("generation_manifest_sha256") == manifest_hash
            and cached.get("complete") is True
        ):
            print(f"REUSE {args.model}/{args.arm}", flush=True)
            return
        raise RuntimeError(f"Cache konflik: {output}")
    runtime_yaml = _runtime_yaml(arm_root, output.parent / "runtime_data.yaml")
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=str(runtime_yaml), split="val", imgsz=640, batch=8, workers=2,
        conf=0.001, iou=0.7, max_det=500, plots=False, verbose=False,
        project=str(output.parent / "ultralytics"), name="validation", exist_ok=True,
        device=args.device,
    )
    layout = discover_layout(arm_root)
    payload = {
        "format": "coffee_detector.faruq_v3_retained_synthetic_density.v1",
        "model": args.model,
        "arm": args.arm,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "generation_manifest_sha256": manifest_hash,
        "metrics": _metric_summary(metrics, layout.names),
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "complete": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    del metrics, model
    gc.collect()


def run(args: argparse.Namespace) -> dict:
    setup = _load(args.setup_summary)
    if (
        setup.get("ready_for_frozen_screening") is not True
        or setup.get("training_executed") is not False
        or setup.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Setup benchmark tidak aman untuk development screening")
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    checkout_root = Path(args.checkout_root).expanduser().resolve()
    selected = list(MODEL_SPECS) if not args.models else args.models
    unknown = sorted(set(selected) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Model tidak dikenal: {unknown}")

    rows = []
    hashes = {}
    for model_name in selected:
        branch, relative_checkpoint = MODEL_SPECS[model_name]
        checkout = _checkout(checkout_root, branch, args.repository)
        checkpoint = _find_checkpoint(project_root, relative_checkpoint)
        hashes[model_name] = _sha256(checkpoint)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(checkout / "src")
        for arm in ARMS:
            arm_info = setup.get("arms", {}).get(arm)
            if not arm_info:
                raise RuntimeError(f"Arm benchmark hilang: {arm}")
            report = output_root / "reports" / model_name / f"{arm}.json"
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--model", model_name, "--checkpoint", str(checkpoint),
                "--arm", arm, "--arm-root", str(arm_info["root"]),
                "--output", str(report), "--device", args.device,
            ]
            print(f"EVALUATE {model_name} / {arm}", flush=True)
            subprocess.run(command, cwd=checkout, env=env, check=True)
            metrics = _load(report)["metrics"]
            rows.append({
                "model": model_name,
                "condition": arm,
                "density": arm_info["density"],
                "macro_map50_95": metrics["macro_map50_95"],
                "bottom3_map50_95": metrics["bottom3_map50_95"],
                "worst_map50_95": metrics["worst_map50_95"],
                "recall": metrics.get("metrics/recall(B)"),
            })

    baseline = {row["condition"]: row for row in rows if row["model"] == "D0FT"}
    for row in rows:
        reference = baseline.get(row["condition"])
        for metric in ("macro_map50_95", "bottom3_map50_95", "worst_map50_95"):
            row[f"delta_{metric}_vs_d0ft"] = (
                None if reference is None else row[metric] - reference[metric]
            )
    summaries = []
    for model_name in selected:
        model_rows = [row for row in rows if row["model"] == model_name]
        deltas = [row["delta_macro_map50_95_vs_d0ft"] for row in model_rows]
        summaries.append({
            "model": model_name,
            "mean_macro_map50_95": statistics.mean(row["macro_map50_95"] for row in model_rows),
            "mean_macro_delta_vs_d0ft": statistics.mean(deltas),
            "minimum_macro_delta_vs_d0ft": min(deltas),
            "improved_conditions": sum(value > 0 for value in deltas),
        })
    summaries.sort(key=lambda row: row["mean_macro_delta_vs_d0ft"], reverse=True)
    payload = {
        "format": "coffee_detector.faruq_v3_retained_synthetic_density_summary.v1",
        "status": "complete",
        "models": selected,
        "checkpoint_hashes": hashes,
        "rows": rows,
        "ranking": summaries,
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "retained_synthetic_density_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUMMARY: {path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--setup-summary")
    parser.add_argument("--project-root")
    parser.add_argument("--output-root")
    parser.add_argument("--checkout-root", default="/content/retained-model-branches")
    parser.add_argument("--repository", default="https://github.com/ediprin/coffee-bean-detection.git")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--device", default="0")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--arm")
    parser.add_argument("--arm-root")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.worker:
        _worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
