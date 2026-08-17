"""Paired D0FT-vs-AF2 synthetic illumination robustness evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import yaml

from coffee_detector.illumination_stress import (
    CONDITIONS,
    IlluminationCondition,
    make_illumination_preview,
    make_illumination_validator,
)


CHECKPOINTS = {
    "D0FT": {
        42: "faruq-v3-acmc-optimization-control-v1/D0FT_seed42/weights/best.pt",
        123: "faruq-v3-acmc-paired-confirmation-v1/D0FT/D0FT_seed123/weights/best.pt",
        2026: "faruq-v3-acmc-paired-confirmation-v1/D0FT/D0FT_seed2026/weights/best.pt",
    },
    "AF2": {
        42: "faruq-v3-breadth-screening-batch-v1/candidates/AFAB/AF2_seed42/weights/best.pt",
        123: "faruq-v3-af2-igem-paired-confirmation-v1/AF2/AF2_seed123/weights/best.pt",
        2026: "faruq-v3-af2-igem-paired-confirmation-v1/AF2/AF2_seed2026/weights/best.pt",
    },
}
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _validate_development_dataset(root: Path, summary_path: Path) -> dict:
    summary = _load(summary_path)
    if (
        summary.get("status") != "complete"
        or summary.get("test_images_accessed") is not False
        or not all(summary.get("gates", {}).values())
        or (root / "test").exists()
    ):
        raise RuntimeError("Faruq-v3 development dataset tidak aman")
    if int(summary.get("images_by_split", {}).get("val", -1)) != 294:
        raise RuntimeError("Jumlah validation berubah dari protokol")
    return summary


def _evaluate_one(
    checkpoint: Path,
    data_root: Path,
    dataset_summary: Path,
    output: Path,
    *,
    model_name: str,
    seed: int,
    condition: IlluminationCondition,
    device: str,
) -> dict:
    checkpoint_hash = _sha256(checkpoint)
    dataset_hash = _sha256(dataset_summary)
    condition_hash = hashlib.sha256(
        json.dumps(asdict(condition), sort_keys=True).encode("utf-8")
    ).hexdigest()
    if output.is_file():
        cached = _load(output)
        if (
            cached.get("complete")
            and cached.get("checkpoint_sha256") == checkpoint_hash
            and cached.get("dataset_summary_sha256") == dataset_hash
            and cached.get("condition_sha256") == condition_hash
        ):
            print(f"REUSE {model_name}-{seed} / {condition.code}", flush=True)
            return cached
        raise RuntimeError(f"Cache evaluasi konflik: {output}")
    from ultralytics import YOLO

    from coffee_detector.evaluate import _classwise_summary

    names_payload = yaml.safe_load(
        (data_root / "data.yaml").read_text(encoding="utf-8")
    )["names"]
    if isinstance(names_payload, list):
        names = {index: name for index, name in enumerate(names_payload)}
    else:
        names = {int(index): name for index, name in names_payload.items()}
    print(f"EVALUATE {model_name}-{seed} / {condition.code}", flush=True)
    model = YOLO(str(checkpoint))
    validator = make_illumination_validator(condition)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = model.val(
        validator=validator,
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
    classwise = _classwise_summary(result.box, names)
    if classwise["classes_without_ground_truth"]:
        raise RuntimeError("Validation kehilangan kelas")
    metrics = {
        "macro_map50_95": classwise["macro_map50_95"],
        "bottom3_class_map50_95": classwise["bottom3_class_map50_95"],
        "worst_class_map50_95": classwise["worst_class_map50_95"],
        "worst_class": min(
            classwise["map50_95_by_class"],
            key=classwise["map50_95_by_class"].get,
        ),
        "recall": float(result.results_dict.get("metrics/recall(B)", 0.0)),
        "precision": float(result.results_dict.get("metrics/precision(B)", 0.0)),
    }
    payload = {
        "format": "coffee_detector.af2_illumination_evaluation.v1",
        "model": model_name,
        "seed": seed,
        "condition": condition.code,
        "condition_spec": asdict(condition),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_summary_sha256": dataset_hash,
        "condition_sha256": condition_hash,
        "metrics": metrics,
        "training_executed": False,
        "test_images_accessed": False,
        "complete": True,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    del result, model
    gc.collect()
    return payload


def summarize_illumination_rows(rows: list[dict], seeds: list[int]) -> dict:
    lookup = {
        (int(row["seed"]), row["condition"], row["model"]): row["metrics"]
        for row in rows
    }
    effects = []
    for seed in seeds:
        clean_d0 = lookup[(seed, "clean", "D0FT")]
        clean_af2 = lookup[(seed, "clean", "AF2")]
        for condition in CONDITIONS:
            if condition.is_clean:
                continue
            d0 = lookup[(seed, condition.code, "D0FT")]
            af2 = lookup[(seed, condition.code, "AF2")]
            effect = {
                "seed": seed,
                "condition": condition.code,
                "family": condition.family,
                "severity": condition.severity,
            }
            for metric in METRICS:
                d0_degradation = d0[metric] - clean_d0[metric]
                af2_degradation = af2[metric] - clean_af2[metric]
                effect[f"d0ft_{metric}"] = d0[metric]
                effect[f"af2_{metric}"] = af2[metric]
                effect[f"d0ft_degradation_{metric}"] = d0_degradation
                effect[f"af2_degradation_{metric}"] = af2_degradation
                effect[f"robustness_advantage_{metric}"] = (
                    af2_degradation - d0_degradation
                )
            effects.append(effect)
    aggregate = []
    for metric in METRICS:
        key = f"robustness_advantage_{metric}"
        values = [row[key] for row in effects]
        seed_means = {
            str(seed): statistics.mean(row[key] for row in effects if row["seed"] == seed)
            for seed in seeds
        }
        aggregate.append(
            {
                "metric": metric,
                "mean_robustness_advantage": statistics.mean(values),
                "minimum_robustness_advantage": min(values),
                "positive_pairs": sum(value > 0 for value in values),
                "pairs": len(values),
                "seed_means": seed_means,
                "positive_seed_means": sum(value > 0 for value in seed_means.values()),
            }
        )
    by_metric = {row["metric"]: row for row in aggregate}
    clean_rows = []
    for seed in seeds:
        clean_rows.append(
            {
                "seed": seed,
                "d0ft_macro_map50_95": lookup[(seed, "clean", "D0FT")][
                    "macro_map50_95"
                ],
                "af2_macro_map50_95": lookup[(seed, "clean", "AF2")][
                    "macro_map50_95"
                ],
            }
        )
    pair_requirement = 6 if seeds == [42] else 18
    seed_requirement = 1 if seeds == [42] else 2
    criteria = {
        "af2_clean_macro_not_lower": statistics.mean(
            row["af2_macro_map50_95"] for row in clean_rows
        )
        >= statistics.mean(row["d0ft_macro_map50_95"] for row in clean_rows),
        "macro_mean_degradation_advantage_positive": by_metric[
            "macro_map50_95"
        ]["mean_robustness_advantage"]
        > 0,
        "macro_positive_condition_seed_pairs": by_metric["macro_map50_95"][
            "positive_pairs"
        ]
        >= pair_requirement,
        "macro_positive_seed_means": by_metric["macro_map50_95"][
            "positive_seed_means"
        ]
        >= seed_requirement,
        "bottom3_mean_degradation_advantage_not_lower": by_metric[
            "bottom3_class_map50_95"
        ]["mean_robustness_advantage"]
        >= 0,
        "worst_mean_degradation_advantage_drop_no_more_than_1_point": by_metric[
            "worst_class_map50_95"
        ]["mean_robustness_advantage"]
        >= -0.01,
    }
    return {
        "clean_rows": clean_rows,
        "effects": effects,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": "PASS" if all(criteria.values()) else "FAIL",
    }


def run(args: argparse.Namespace) -> dict:
    seeds = [int(seed) for seed in args.seeds]
    if seeds not in ([42], [42, 123, 2026]):
        raise ValueError("Seeds harus screening 42 atau konfirmasi 42 123 2026")
    project_root = Path(args.project_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    grouped_summary_path = Path(args.grouped_summary).expanduser().resolve()
    dataset_summary = _validate_development_dataset(data_root, grouped_summary_path)
    if seeds != [42]:
        if not args.screen_summary:
            raise RuntimeError("Konfirmasi memerlukan summary screening PASS")
        screen = _load(args.screen_summary)
        if screen.get("decision") != "PASS" or screen.get("confirmation_authorized") is not True:
            raise RuntimeError("Screening tidak mengotorisasi konfirmasi")
    output_root.mkdir(parents=True, exist_ok=True)
    preview = make_illumination_preview(
        data_root, output_root / "illumination_preview.jpg"
    )
    rows = []
    for condition in CONDITIONS:
        for seed in seeds:
            for model_name in ("D0FT", "AF2"):
                checkpoint = (
                    project_root / "experiments" / CHECKPOINTS[model_name][seed]
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                report = (
                    output_root
                    / "reports"
                    / f"seed{seed}"
                    / condition.code
                    / f"{model_name}.json"
                )
                evaluated = _evaluate_one(
                    checkpoint,
                    data_root,
                    grouped_summary_path,
                    report,
                    model_name=model_name,
                    seed=seed,
                    condition=condition,
                    device=args.device,
                )
                rows.append(evaluated)
    summary = summarize_illumination_rows(rows, seeds)
    result = {
        "format": "coffee_detector.af2_illumination_summary.v1",
        "stage": "screen" if seeds == [42] else "confirmation",
        "seeds": seeds,
        "conditions": [condition.code for condition in CONDITIONS],
        "dataset_summary": str(Path(args.grouped_summary).expanduser().resolve()),
        "preview": str(preview),
        **summary,
        "confirmation_authorized": summary["decision"] == "PASS" and seeds == [42],
        "training_executed": False,
        "test_images_accessed": False,
        "claim_limit": (
            "Deterministic synthetic photometric robustness on validation identities; "
            "not controlled real-lux evidence."
        ),
    }
    filename = (
        "illumination_screen_seed42.json"
        if seeds == [42]
        else "illumination_confirmation_three_seed.json"
    )
    path = output_root / filename
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"SUMMARY: {path}", flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--project-root", required=True)
    value.add_argument("--data-root", required=True)
    value.add_argument("--grouped-summary", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--seeds", nargs="+", type=int, default=[42])
    value.add_argument("--screen-summary")
    value.add_argument("--device", default="0")
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
