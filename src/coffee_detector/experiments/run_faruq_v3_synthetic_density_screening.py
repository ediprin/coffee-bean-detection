"""Evaluate frozen D0FT and ACMC1 on a Faruq-v3 synthetic density ladder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from coffee_detector.dataset import discover_layout
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES
from coffee_detector.run_sni21_density_evaluation import _metric_summary


ARM_ORDER = tuple(f"B{index}_balanced_mild" for index in range(4))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str | Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_yaml(root: Path, output: Path) -> Path:
    layout = discover_layout(root)
    expected = {index: name for index, name in enumerate(SNI21_CLASSES)}
    if layout.names != expected or "val" not in layout.splits:
        raise RuntimeError(f"Dataset density tidak valid: {root}")
    payload = yaml.safe_load(layout.yaml_path.read_text(encoding="utf-8")) or {}
    payload["path"] = str(layout.root)
    payload["names"] = expected
    relative = layout.splits["val"][0].relative_to(layout.root).as_posix()
    # Ultralytics checks all schema keys even though this is val-only.
    payload.update({"train": relative, "val": relative, "test": relative})
    payload["development_diagnostic_only"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return output


def _evaluate(
    checkpoint: Path,
    checkpoint_hash: str,
    arm: str,
    arm_root: Path,
    output: Path,
    *,
    device: str | None,
) -> dict:
    manifest_hash = _sha256(arm_root / "metadata/generation_manifest.json")
    if output.is_file():
        cached = _load(output, "Density report cache")
        if (
            cached.get("checkpoint_sha256") != checkpoint_hash
            or cached.get("generation_manifest_sha256") != manifest_hash
            or cached.get("complete") is not True
        ):
            raise RuntimeError(f"Cache density konflik: {output}")
        print(f"REUSE {output.name}", flush=True)
        return cached
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang") from error
    runtime_yaml = _runtime_yaml(arm_root, output.parent / "runtime_data.yaml")
    model = YOLO(str(checkpoint))
    kwargs = {
        "data": str(runtime_yaml),
        "split": "val",
        "imgsz": 640,
        "batch": 8,
        "workers": 2,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 500,
        "plots": False,
        "verbose": True,
        "project": str(output.parent / "ultralytics"),
        "name": "validation",
        "exist_ok": True,
    }
    if device is not None:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    layout = discover_layout(arm_root)
    payload = {
        "format": "coffee_detector.faruq_v3_synthetic_density_report.v1",
        "arm": arm,
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
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
    return payload


def run_faruq_v3_synthetic_density_screening(
    setup_summary: str | Path,
    d0ft_checkpoint: str | Path,
    acmc_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = "0",
) -> dict:
    if seed != 42:
        raise ValueError("Screening sintetis dibekukan pada seed 42")
    setup = _load(setup_summary, "Synthetic density setup")
    if (
        setup.get("format") != "coffee_detector.faruq_v3_synthetic_density_setup.v1"
        or setup.get("ready_for_frozen_screening") is not True
        or setup.get("training_executed") is not False
        or setup.get("test_images_accessed") is not False
        or setup.get("source_split") != "faruq_v3_validation"
    ):
        raise RuntimeError("Setup synthetic density tidak mengotorisasi screening")
    checkpoints = {
        "D0FT": Path(d0ft_checkpoint).expanduser().resolve(),
        "ACMC1": Path(acmc_checkpoint).expanduser().resolve(),
    }
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {name: _sha256(path) for name, path in checkpoints.items()}
    output_root = Path(output_root).expanduser().resolve()
    reports = {}
    rows = []
    for arm in ARM_ORDER:
        arm_info = setup.get("arms", {}).get(arm)
        if not arm_info:
            raise RuntimeError(f"Arm hilang: {arm}")
        arm_root = Path(arm_info["root"]).expanduser().resolve()
        reports[arm] = {}
        metrics = {}
        for model_name in ("D0FT", "ACMC1"):
            print(f"EVALUATE {arm} / {model_name}", flush=True)
            report = _evaluate(
                checkpoints[model_name],
                hashes[model_name],
                arm,
                arm_root,
                output_root / "reports" / arm / f"{model_name}_seed42.json",
                device=device,
            )
            reports[arm][model_name] = str(
                output_root / "reports" / arm / f"{model_name}_seed42.json"
            )
            metrics[model_name] = report["metrics"]
        row = {
            "condition": arm,
            "density": arm_info["density"],
        }
        for metric in ("macro_map50_95", "bottom3_map50_95", "worst_map50_95"):
            left = float(metrics["D0FT"][metric])
            right = float(metrics["ACMC1"][metric])
            row[f"d0ft_{metric}"] = left
            row[f"acmc1_{metric}"] = right
            row[f"delta_{metric}"] = right - left
        rows.append(row)

    macro_deltas = [row["delta_macro_map50_95"] for row in rows]
    payload = {
        "format": "coffee_detector.faruq_v3_synthetic_density_screening.v1",
        "status": "complete",
        "seed": seed,
        "checkpoint_hashes": hashes,
        "rows": rows,
        "reports": reports,
        "summary": {
            "mean_macro_delta": statistics.mean(macro_deltas),
            "minimum_macro_delta": min(macro_deltas),
            "macro_improved_conditions": sum(value > 0 for value in macro_deltas),
            "conditions": len(macro_deltas),
        },
        "interpretation_status": "DEVELOPMENT_SYNTHETIC_DIAGNOSTIC_ONLY",
        "training_executed": False,
        "test_images_accessed": False,
        "locked_test_conclusion_changed": False,
        "further_tuning_authorized": False,
        "next_action": "REPORT_DENSITY_TREND_WITHOUT_CHANGING_LOCKED_TEST",
    }
    summary_path = output_root / "synthetic_density_seed42_summary.json"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary_path"] = str(summary_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen ACMC synthetic density screening")
    parser.add_argument("--setup-summary", required=True)
    parser.add_argument("--d0ft-checkpoint", required=True)
    parser.add_argument("--acmc-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run_faruq_v3_synthetic_density_screening(
        args.setup_summary,
        args.d0ft_checkpoint,
        args.acmc_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
