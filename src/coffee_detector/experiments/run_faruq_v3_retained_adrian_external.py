"""Evaluate retained Faruq-v3 seed-42 models on real Adrian validation."""

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


MODEL_SPECS = {
    "D0FT": ("agent/af2-igem-paired-confirmation", "faruq-v3-acmc-optimization-control-v1/D0FT_seed42/weights/best.pt"),
    "SAF1": ("agent/safpn-classification-alignment", "faruq-v3-breadth-screening-batch-v1/candidates/SAF1/SAF1_seed42/weights/best.pt"),
    "IGEM1": ("agent/af2-igem-paired-confirmation", "faruq-v3-breadth-screening-batch-v1/candidates/IGEM/IGEM1_seed42/weights/best.pt"),
    "ACMC1": ("agent/af2-igem-paired-confirmation", "faruq-v3-acmc-one-stage-v1/ACMC1_seed42/weights/best.pt"),
    "AF2": ("agent/af2-igem-paired-confirmation", "faruq-v3-breadth-screening-batch-v1/candidates/AFAB/AF2_seed42/weights/best.pt"),
    "GEO1": ("agent/circle-cpe-screening", "faruq-v3-geometry-conditioning-screening-v1/GEO1_seed42/weights/best.pt"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _checkout(root: Path, branch: str, repository: str) -> Path:
    destination = root / branch.replace("/", "__")
    if not (destination / ".git").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, repository, str(destination)],
            check=True,
        )
    return destination


def _runtime_yaml(data_root: Path, output: Path) -> Path:
    from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES

    payload = {
        "path": str(data_root), "train": "val/images", "val": "val/images",
        "names": {index: name for index, name in enumerate(SNI21_CLASSES)},
        "external_development_only": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def _worker(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    from coffee_detector.evaluate import _classwise_summary
    from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES

    checkpoint = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    checkpoint_hash = _sha256(checkpoint)
    manifest_hash = _sha256(Path(args.dataset_manifest).resolve())
    if output.is_file():
        cached = _load(output)
        if (
            cached.get("checkpoint_sha256") == checkpoint_hash
            and cached.get("dataset_manifest_sha256") == manifest_hash
            and cached.get("complete") is True
        ):
            print(f"REUSE {args.model}", flush=True)
            return
        raise RuntimeError(f"Cache konflik: {output}")
    model = YOLO(str(checkpoint))
    runtime_yaml = _runtime_yaml(data_root, output.parent / "runtime_data.yaml")
    metrics = model.val(
        data=str(runtime_yaml), split="val", imgsz=640, batch=8, workers=2,
        conf=0.001, iou=0.7, max_det=500, plots=False, verbose=False,
        project=str(output.parent / "ultralytics"), name="validation", exist_ok=True,
        device=args.device,
    )
    classwise = _classwise_summary(
        metrics.box, {index: name for index, name in enumerate(SNI21_CLASSES)}
    )
    worst_class = min(
        classwise["map50_95_by_class"], key=classwise["map50_95_by_class"].get
    )
    result_dict = {key: float(value) for key, value in metrics.results_dict.items()}
    payload = {
        "format": "coffee_detector.faruq_v3_retained_adrian_external.v1",
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_manifest_sha256": manifest_hash,
        "metrics": {
            **result_dict,
            "macro_map50_95": classwise["macro_map50_95"],
            "bottom3_class_map50_95": classwise["bottom3_class_map50_95"],
            "worst_class_map50_95": classwise["worst_class_map50_95"],
            "worst_class": worst_class,
        },
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
    setup = _load(args.adrian_setup)
    if (
        setup.get("status") != "complete"
        or setup.get("source_dataset") != "adrian_detection"
        or setup.get("test_images_accessed") is not False
        or not all(setup.get("gates", {}).values())
    ):
        raise RuntimeError("Adrian validation setup tidak aman")
    project_root = Path(args.project_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    manifest = data_root / "adrian_external_validation_manifest.json"
    output_root = Path(args.output_root).expanduser().resolve()
    checkout_root = Path(args.checkout_root).expanduser().resolve()
    selected = list(MODEL_SPECS) if not args.models else args.models
    unknown = sorted(set(selected) - set(MODEL_SPECS))
    if unknown:
        raise ValueError(f"Model tidak dikenal: {unknown}")
    rows = []
    for model_name in selected:
        branch, relative = MODEL_SPECS[model_name]
        checkout = _checkout(checkout_root, branch, args.repository)
        checkpoint = project_root / "experiments" / relative
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        report = output_root / "reports" / model_name / "evaluation.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(checkout / "src")
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--model", model_name, "--checkpoint", str(checkpoint),
            "--data-root", str(data_root), "--dataset-manifest", str(manifest),
            "--output", str(report), "--device", args.device,
        ]
        print(f"EVALUATE ADRIAN / {model_name}", flush=True)
        subprocess.run(command, cwd=checkout, env=env, check=True)
        metrics = _load(report)["metrics"]
        rows.append({
            "model": model_name,
            "macro_map50_95": metrics["macro_map50_95"],
            "bottom3_class_map50_95": metrics["bottom3_class_map50_95"],
            "worst_class_map50_95": metrics["worst_class_map50_95"],
            "worst_class": metrics["worst_class"],
            "recall": metrics.get("metrics/recall(B)"),
        })
    baseline = next(row for row in rows if row["model"] == "D0FT")
    for row in rows:
        for metric in ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95"):
            row[f"delta_{metric}_vs_d0ft"] = row[metric] - baseline[metric]
    rows.sort(key=lambda row: row["delta_macro_map50_95_vs_d0ft"], reverse=True)
    payload = {
        "format": "coffee_detector.faruq_v3_retained_adrian_external_summary.v1",
        "status": "complete", "models": selected, "rows": rows,
        "adrian_setup": setup,
        "training_executed": False, "test_images_accessed": False,
        "development_only": True,
        "claim_limit": "Independent-source post-hoc evaluation with only eight Adrian parent identities.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "retained_adrian_external_summary.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUMMARY: {path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--project-root")
    parser.add_argument("--data-root")
    parser.add_argument("--adrian-setup")
    parser.add_argument("--output-root")
    parser.add_argument("--checkout-root", default="/content/retained-model-branches")
    parser.add_argument("--repository", default="https://github.com/ediprin/coffee-bean-detection.git")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--device", default="0")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    _worker(args) if args.worker else run(args)


if __name__ == "__main__":
    main()
