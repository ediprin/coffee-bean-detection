"""Locked, no-training illumination diagnostic for a confirmed AF2 spectral winner.

This intentionally measures a *paired degradation difference*: for each exact
validation image and deterministic illumination transform, the candidate's
drop from its own clean score is compared with AF2C's drop from AF2C clean
score.  It is a development diagnostic only and never reopens Faruq test.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import yaml

from coffee_detector.evaluate import _classwise_summary
from coffee_detector.illumination_stress import (
    CONDITIONS,
    IlluminationCondition,
    make_illumination_preview,
    make_illumination_validator,
)


METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
SEEDS = (42, 123, 2026)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _names(data_root: Path) -> dict[int, str]:
    payload = yaml.safe_load((data_root / "data.yaml").read_text(encoding="utf-8"))["names"]
    return {index: value for index, value in enumerate(payload)} if isinstance(payload, list) else {int(k): v for k, v in payload.items()}


def _evaluate(
    checkpoint: Path,
    data_root: Path,
    output: Path,
    *,
    model: str,
    seed: int,
    condition: IlluminationCondition,
    device: str,
) -> dict:
    checkpoint_hash = _sha256(checkpoint)
    condition_hash = hashlib.sha256(json.dumps(asdict(condition), sort_keys=True).encode()).hexdigest()
    if output.is_file():
        cached = _read(output)
        if (
            cached.get("complete") is True
            and cached.get("checkpoint_sha256") == checkpoint_hash
            and cached.get("condition_sha256") == condition_hash
            and cached.get("model") == model
            and cached.get("seed") == seed
        ):
            print(f"REUSE {model}-{seed}/{condition.code}", flush=True)
            return cached
        raise RuntimeError(f"Cache illumination konflik: {output}")
    from ultralytics import YOLO

    print(f"EVALUATE {model}-{seed}/{condition.code}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = YOLO(str(checkpoint)).val(
        validator=make_illumination_validator(condition),
        data=str(data_root / "data.yaml"),
        split="val",
        imgsz=640,
        batch=8,
        workers=2,
        conf=0.001,
        iou=0.7,
        max_det=500,
        plots=False,
        verbose=False,
        project=str(output.parent / "ultralytics"),
        name="validation",
        exist_ok=True,
        device=device,
    )
    classwise = _classwise_summary(result.box, _names(data_root))
    if classwise["classes_without_ground_truth"]:
        raise RuntimeError("Validation illumination kehilangan kelas")
    payload = {
        "format": "coffee_detector.af2_spectral.illumination_row.v1",
        "model": model,
        "seed": seed,
        "condition": condition.code,
        "condition_spec": asdict(condition),
        "checkpoint_sha256": checkpoint_hash,
        "condition_sha256": condition_hash,
        "metrics": {
            metric: float(classwise[metric]) for metric in METRICS
        },
        "training_executed": False,
        "test_images_accessed": False,
        "complete": True,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    del result
    gc.collect()
    return payload


def run_spectral_illumination(
    confirmation_summary: str | Path,
    data_root: str | Path,
    af2_checkpoints: tuple[str | Path, ...],
    candidate_checkpoints: tuple[str | Path, ...],
    output_root: str | Path,
    *,
    device: str = "0",
) -> dict:
    confirmation = _read(confirmation_summary)
    if (
        confirmation.get("decision") != "PASS"
        or confirmation.get("next") != "AUTHORIZE_POSTHOC_EXTERNAL_EVALUATION"
        or confirmation.get("test_opened") is not False
    ):
        raise RuntimeError("Confirmation PASS diperlukan sebelum illumination diagnostic")
    if len(af2_checkpoints) != 3 or len(candidate_checkpoints) != 3:
        raise ValueError("Harus tersedia tepat tiga checkpoint AF2C dan kandidat")
    data_root = Path(data_root).expanduser().resolve()
    if (data_root / "test").exists() or not (data_root / "val/images").is_dir():
        raise RuntimeError("Illumination hanya menerima development root tanpa test")
    output_root = Path(output_root).expanduser().resolve()
    candidate_arm = str(confirmation["arm"])
    rows: list[dict] = []
    for seed, af2_value, candidate_value in zip(SEEDS, af2_checkpoints, candidate_checkpoints):
        for model, checkpoint_value in (("AF2C", af2_value), (candidate_arm, candidate_value)):
            checkpoint = Path(checkpoint_value).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            for condition in CONDITIONS:
                rows.append(_evaluate(
                    checkpoint,
                    data_root,
                    output_root / "reports" / f"seed{seed}" / condition.code / f"{model}.json",
                    model=model,
                    seed=seed,
                    condition=condition,
                    device=device,
                ))
    lookup = {(item["seed"], item["condition"], item["model"]): item["metrics"] for item in rows}
    effects = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            if condition.is_clean:
                continue
            value = {"seed": seed, "condition": condition.code, "family": condition.family}
            for metric in METRICS:
                af2_drop = lookup[(seed, condition.code, "AF2C")][metric] - lookup[(seed, "clean", "AF2C")][metric]
                candidate_drop = lookup[(seed, condition.code, candidate_arm)][metric] - lookup[(seed, "clean", candidate_arm)][metric]
                value[f"af2c_degradation_{metric}"] = af2_drop
                value[f"candidate_degradation_{metric}"] = candidate_drop
                value[f"robustness_advantage_{metric}"] = candidate_drop - af2_drop
            effects.append(value)
    aggregate = []
    for metric in METRICS:
        key = f"robustness_advantage_{metric}"
        values = [float(row[key]) for row in effects]
        aggregate.append({
            "metric": metric,
            "mean_robustness_advantage": statistics.fmean(values),
            "minimum_robustness_advantage": min(values),
            "positive_pairs": sum(value > 0 for value in values),
            "pairs": len(values),
            "seed_means": {str(seed): statistics.fmean(row[key] for row in effects if row["seed"] == seed) for seed in SEEDS},
        })
    result = {
        "format": "coffee_detector.af2_spectral.illumination_summary.v1",
        "status": "POSTHOC_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "arm": candidate_arm,
        "seeds": list(SEEDS),
        "conditions": [condition.code for condition in CONDITIONS],
        "effects": effects,
        "aggregate": aggregate,
        "preview": str(make_illumination_preview(data_root, output_root / "illumination_preview.jpg")),
        "training_executed": False,
        "test_images_accessed": False,
        "changes_confirmation_decision": False,
        "claim_limit": "Synthetic photometric stress on validation identities; not a real controlled-lux study.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "af2_spectral_illumination_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="No-training AF2 spectral illumination diagnostic")
    parser.add_argument("--confirmation-summary", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--af2-checkpoints", nargs=3, required=True)
    parser.add_argument("--candidate-checkpoints", nargs=3, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run_spectral_illumination(
        args.confirmation_summary,
        args.data_root,
        tuple(args.af2_checkpoints),
        tuple(args.candidate_checkpoints),
        args.output_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
