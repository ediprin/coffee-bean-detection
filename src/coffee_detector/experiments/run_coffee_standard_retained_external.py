"""Evaluate retained Faruq-v3 models on the Coffee Standard external benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


# Keep this registry local: workers execute with the historical branch that
# defines each custom checkpoint class, and those branches do not necessarily
# contain the newer cross-model orchestration modules.
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkout(root: Path, branch: str, repository: str) -> Path:
    destination = root / branch.replace("/", "__")
    if not (destination / ".git").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, repository, str(destination)], check=True)
    return destination


def _worker(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    from coffee_detector.evaluate import _classwise_summary
    from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES

    checkpoint, root, output = Path(args.checkpoint), Path(args.data_root), Path(args.output)
    checkpoint_hash, manifest_hash = _sha(checkpoint), _sha(root / "external_manifest.json")
    if output.is_file():
        cached = _load(output)
        if cached.get("checkpoint_sha256") == checkpoint_hash and cached.get("manifest_sha256") == manifest_hash and cached.get("complete"):
            print(f"REUSE {args.model}", flush=True)
            return
        raise RuntimeError(f"Cache konflik: {output}")
    runtime = output.parent / "runtime_data.yaml"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(yaml.safe_dump({
        "path": str(root), "train": "external/images", "val": "external/images",
        "names": {index: name for index, name in enumerate(SNI21_CLASSES)},
    }, sort_keys=False), encoding="utf-8")
    metrics = YOLO(str(checkpoint)).val(
        data=str(runtime), split="val", imgsz=640, batch=8, workers=2,
        conf=0.001, iou=0.7, max_det=500, plots=False, verbose=False,
        project=str(output.parent / "ultralytics"), name="external", exist_ok=True, device=args.device,
    )
    classwise = _classwise_summary(metrics.box, {index: name for index, name in enumerate(SNI21_CLASSES)})
    payload = {
        "model": args.model, "checkpoint_sha256": checkpoint_hash, "manifest_sha256": manifest_hash,
        "metrics": {
            "macro_map50_95": classwise["macro_map50_95"],
            "bottom3_class_map50_95": classwise["bottom3_class_map50_95"],
            "worst_class_map50_95": classwise["worst_class_map50_95"],
            "worst_class": min(classwise["map50_95_by_class"], key=classwise["map50_95_by_class"].get),
            "recall": float(metrics.results_dict.get("metrics/recall(B)", 0.0)),
            "classwise": classwise,
        },
        "training_executed": False, "test_images_accessed": False,
        "external_posthoc_only": True, "complete": True,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    del metrics
    gc.collect()


def run(args: argparse.Namespace) -> dict:
    root, project, output_root = Path(args.data_root), Path(args.project_root), Path(args.output_root)
    setup = _load(root / "external_summary.json")
    if setup.get("role") != "external_posthoc_diagnostic_only" or setup.get("training_authorized") is not False:
        raise RuntimeError("External benchmark tidak valid")
    selected = args.models or list(MODEL_SPECS)
    rows = []
    for name in selected:
        branch, relative = MODEL_SPECS[name]
        checkout = _checkout(Path(args.checkout_root), branch, args.repository)
        checkpoint = project / "experiments" / relative
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        report = output_root / "reports" / name / "evaluation.json"
        env = os.environ.copy(); env["PYTHONPATH"] = str(checkout / "src")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model", name,
                   "--checkpoint", str(checkpoint), "--data-root", str(root), "--output", str(report), "--device", args.device]
        print(f"EVALUATE COFFEE-STANDARD / {name}", flush=True)
        subprocess.run(command, cwd=checkout, env=env, check=True)
        metric = _load(report)["metrics"]
        rows.append({"model": name, **{key: metric[key] for key in ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95", "worst_class", "recall")}})
    baseline = next(row for row in rows if row["model"] == "D0FT")
    for row in rows:
        for metric in ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95"):
            row[f"delta_{metric}_vs_d0ft"] = row[metric] - baseline[metric]
    rows.sort(key=lambda row: row["macro_map50_95"], reverse=True)
    summary = {
        "status": "complete", "rows": rows, "training_executed": False, "test_images_accessed": False,
        "role": "external_posthoc_diagnostic_only",
        "claim_limit": "Cross-dataset transfer on 148 independent parent identities and 18 directly mapped classes.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "coffee_standard_retained_external_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--worker", action="store_true"); value.add_argument("--project-root"); value.add_argument("--data-root")
    value.add_argument("--output-root"); value.add_argument("--checkout-root", default="/content/coffee-standard-model-branches")
    value.add_argument("--repository", default="https://github.com/ediprin/coffee-bean-detection.git")
    value.add_argument("--models", nargs="*"); value.add_argument("--device", default="0")
    value.add_argument("--model"); value.add_argument("--checkpoint"); value.add_argument("--output")
    return value


def main() -> None:
    args = parser().parse_args()
    result = _worker(args) if args.worker else run(args)
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
