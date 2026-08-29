"""Validation-only clustered parent bootstrap for AF2CUE1.

The analysis reuses fixed seed-42 checkpoints and resamples complete Faruq
source-parent clusters.  It never trains a model or reads a test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from coffee_detector.evaluate import _classwise_summary


ARMS = ("AF2BASE", "AF2SPDS", "AF2CUE1")
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _result_path(original_root: Path, refinement_root: Path, arm: str) -> Path:
    root = original_root if arm in {"AF2BASE", "AF2SPDS"} else refinement_root
    return root / "val_reports" / f"{arm}_seed42_result.json"


def _resolve_checkpoint(raw: str, project_root: Path) -> Path:
    source = Path(raw).expanduser()
    if source.is_file():
        return source.resolve()
    parts = list(source.parts)
    if "Coffee_Bean_Detection" in parts:
        index = parts.index("Coffee_Bean_Detection")
        candidate = project_root.joinpath(*parts[index + 1 :])
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Checkpoint laporan tidak ditemukan: {raw}")


def _load_contracts(
    project_root: Path, original_root: Path, refinement_root: Path
) -> dict[str, dict[str, Any]]:
    contracts = {}
    for arm in ARMS:
        path = _result_path(original_root, refinement_root, arm)
        payload = _load_json(path, f"Laporan {arm}")
        if (
            payload.get("arm") != arm
            or int(payload.get("seed", -1)) != 42
            or payload.get("evaluation_split") != "val"
            or payload.get("test_images_accessed") is not False
        ):
            raise RuntimeError(f"Kontrak laporan {arm} tidak kompatibel: {path}")
        metrics = payload.get("metrics", {})
        if (
            len(metrics.get("map50_95_by_class", {})) != 21
            or metrics.get("classes_without_ground_truth")
        ):
            raise RuntimeError(f"Laporan {arm} tidak memiliki 21 kelas validation")
        checkpoint = _resolve_checkpoint(str(payload.get("checkpoint", "")), project_root)
        contracts[arm] = {
            "report": path,
            "report_sha256": _sha256(path),
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256(checkpoint),
            "historical_metrics": {metric: float(metrics[metric]) for metric in METRICS},
        }
    return contracts


def _parse_yolo_labels(path: Path, width: int, height: int, *, prediction: bool) -> list[dict]:
    rows = []
    if not path.is_file():
        if prediction:
            return rows
        raise FileNotFoundError(path)
    expected = 6 if prediction else 5
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != expected:
            raise ValueError(f"Format YOLO tidak cocok ({len(fields)} != {expected}): {path}")
        class_id = int(fields[0])
        x, y, box_width, box_height = map(float, fields[1:5])
        row = {
            "class_id": class_id,
            "xyxy": [
                (x - box_width / 2) * width,
                (y - box_height / 2) * height,
                (x + box_width / 2) * width,
                (y + box_height / 2) * height,
            ],
        }
        if prediction:
            row["score"] = float(fields[5])
        rows.append(row)
    return rows


def _collect_observations(data_root: Path, prediction_labels: Path) -> list[dict]:
    image_paths = sorted(path for path in (data_root / "val/images").glob("*") if path.is_file())
    if not image_paths:
        raise FileNotFoundError("Validation images tidak ditemukan")
    observations = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        label_name = image_path.with_suffix(".txt").name
        observations.append(
            {
                "image_name": image_path.name,
                "targets": _parse_yolo_labels(
                    data_root / "val/labels" / label_name, width, height, prediction=False
                ),
                "predictions": _parse_yolo_labels(
                    prediction_labels / label_name, width, height, prediction=True
                ),
            }
        )
    observations.sort(key=lambda row: row["image_name"].lower())
    return observations


def _evaluate_arm(
    arm: str,
    contract: Mapping[str, Any],
    data_root: Path,
    manifest_hash: str,
    output: Path,
    *,
    device: str | None,
) -> dict[str, Any]:
    if output.is_file():
        cached = _load_json(output, f"Cache inference {arm}")
        if (
            cached.get("checkpoint_sha256") != contract["checkpoint_sha256"]
            or cached.get("manifest_sha256") != manifest_hash
            or cached.get("split") != "val"
            or cached.get("training_executed") is not False
            or cached.get("test_opened") is not False
            or not isinstance(cached.get("prediction_observations"), list)
        ):
            raise RuntimeError(f"Cache inference tidak kompatibel: {output}")
        print(f"REUSE VALIDATION REPORT: {arm}", flush=True)
        return cached

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang") from error

    with tempfile.TemporaryDirectory(prefix=f"af2cue1_{arm.lower()}_") as temporary:
        kwargs: dict[str, Any] = {
            "data": str(data_root / "data.yaml"),
            "split": "val",
            "imgsz": 640,
            "batch": 16,
            "workers": 2,
            "max_det": 500,
            "conf": 0.001,
            "iou": 0.7,
            "plots": False,
            "verbose": False,
            "save_txt": True,
            "save_conf": True,
            "project": temporary,
            "name": "evaluation",
            "exist_ok": True,
        }
        if device is not None:
            kwargs["device"] = device
        yolo = YOLO(str(contract["checkpoint"]))
        metrics = yolo.val(**kwargs)
        values = {key: float(value) for key, value in metrics.results_dict.items()}
        box = getattr(metrics, "box", None)
        if box is None or getattr(box, "ap", None) is None:
            raise RuntimeError(f"Evaluator tidak menghasilkan AP: {arm}")
        names = getattr(yolo.model, "names", None)
        if isinstance(names, list):
            names = dict(enumerate(names))
        if not isinstance(names, dict):
            raise RuntimeError(f"Evaluator tidak menghasilkan ontology: {arm}")
        names = {int(key): str(value) for key, value in names.items()}
        values.update(_classwise_summary(box, names))
        if values.get("classes_without_ground_truth") or len(values["map50_95_by_class"]) != 21:
            raise RuntimeError(f"Validation kehilangan kelas: {arm}")
        save_dir = Path(getattr(metrics, "save_dir", Path(temporary) / "evaluation"))
        observations = _collect_observations(data_root, save_dir / "labels")

    endpoint_deltas = {
        metric: float(values[metric]) - float(contract["historical_metrics"][metric])
        for metric in METRICS
    }
    if max(abs(value) for value in endpoint_deltas.values()) > 1.0e-6:
        raise RuntimeError(f"Endpoint {arm} tidak mereproduksi laporan historis: {endpoint_deltas}")
    payload = {
        "format": "coffee_detector.af2cue1.parent_bootstrap.inference.v1",
        "arm": arm,
        "checkpoint": str(contract["checkpoint"]),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "manifest_sha256": manifest_hash,
        "split": "val",
        "metrics": values,
        "historical_endpoint_deltas": endpoint_deltas,
        "prediction_observations": observations,
        "training_executed": False,
        "test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None).prod(axis=2)
    left_area = np.clip(left[:, 2:] - left[:, :2], 0.0, None).prod(axis=1)
    right_area = np.clip(right[:, 2:] - right[:, :2], 0.0, None).prod(axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / np.clip(union, 1.0e-12, None)


def _prepare_observations(observations: list[dict], class_count: int) -> list[list[dict]]:
    prepared = []
    for observation in observations:
        class_rows = []
        for class_id in range(class_count):
            targets = np.asarray(
                [row["xyxy"] for row in observation["targets"] if row["class_id"] == class_id],
                dtype=np.float64,
            ).reshape(-1, 4)
            predictions = [
                row for row in observation["predictions"] if row["class_id"] == class_id
            ]
            predictions.sort(key=lambda row: row["score"], reverse=True)
            boxes = np.asarray([row["xyxy"] for row in predictions], dtype=np.float64).reshape(-1, 4)
            scores = np.asarray([row["score"] for row in predictions], dtype=np.float64)
            overlaps = _iou(boxes, targets)
            true_positive = np.zeros((len(predictions), len(IOU_THRESHOLDS)), dtype=bool)
            for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
                used: set[int] = set()
                for prediction_index in range(len(predictions)):
                    order = np.argsort(overlaps[prediction_index])[::-1]
                    for target_index in order:
                        if overlaps[prediction_index, target_index] < threshold:
                            break
                        if int(target_index) not in used:
                            used.add(int(target_index))
                            true_positive[prediction_index, threshold_index] = True
                            break
            class_rows.append(
                {"targets": len(targets), "scores": scores, "tp": true_positive}
            )
        prepared.append(class_rows)
    return prepared


def _average_precision(tp: np.ndarray, scores: np.ndarray, targets: int) -> float:
    if targets <= 0:
        return float("nan")
    if not len(scores):
        return 0.0
    order = np.argsort(scores)[::-1]
    true_positive = np.cumsum(tp[order].astype(np.float64))
    false_positive = np.cumsum((~tp[order]).astype(np.float64))
    recall = true_positive / targets
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0e-12)
    recall_curve = np.concatenate(([0.0], recall, [1.0]))
    precision_curve = np.concatenate(([1.0], precision, [0.0]))
    precision_curve = np.maximum.accumulate(precision_curve[::-1])[::-1]
    return float(np.interp(np.linspace(0.0, 1.0, 101), recall_curve, precision_curve).mean())


def _headline(prepared: list[list[dict]], sample: np.ndarray, class_count: int) -> dict[str, float] | None:
    class_values = []
    for class_id in range(class_count):
        rows = [prepared[int(index)][class_id] for index in sample]
        targets = sum(int(row["targets"]) for row in rows)
        if targets <= 0:
            return None
        scores = np.concatenate([row["scores"] for row in rows])
        true_positive = np.concatenate([row["tp"] for row in rows], axis=0)
        aps = [
            _average_precision(true_positive[:, index], scores, targets)
            for index in range(len(IOU_THRESHOLDS))
        ]
        class_values.append(float(np.mean(aps)))
    ordered = np.sort(np.asarray(class_values, dtype=np.float64))
    return {
        "macro_map50_95": float(ordered.mean()),
        "bottom3_class_map50_95": float(ordered[:3].mean()),
        "worst_class_map50_95": float(ordered[0]),
    }


def paired_parent_bootstrap(
    observations_by_arm: Mapping[str, list[dict]],
    parent_by_image: Mapping[str, str],
    *,
    iterations: int = 1000,
    seed: int = 20260829,
    class_count: int = 21,
) -> dict[str, Any]:
    """Resample complete source-parent clusters, paired across all arms."""

    if set(observations_by_arm) != set(ARMS):
        raise ValueError(f"Arm harus tepat {ARMS}")
    names = [row["image_name"] for row in observations_by_arm[ARMS[0]]]
    if len(names) != len(set(name.lower() for name in names)):
        raise RuntimeError("Nama validation image tidak unik")
    for arm in ARMS[1:]:
        if [row["image_name"] for row in observations_by_arm[arm]] != names:
            raise RuntimeError("Urutan validation image berbeda antar-arm")
    missing = sorted(set(names) - set(parent_by_image))
    extra = sorted(set(parent_by_image) - set(names))
    if missing or extra:
        raise RuntimeError(f"Parent map tidak cocok; missing={missing[:3]}, extra={extra[:3]}")

    parent_indices: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        parent_indices[str(parent_by_image[name])].append(index)
    parents = sorted(parent_indices)
    if len(parents) < 21:
        raise RuntimeError("Parent independen terlalu sedikit")
    clusters = [np.asarray(parent_indices[parent], dtype=np.int64) for parent in parents]
    prepared = {
        arm: _prepare_observations(observations_by_arm[arm], class_count) for arm in ARMS
    }
    full = np.arange(len(names), dtype=np.int64)
    point = {arm: _headline(prepared[arm], full, class_count) for arm in ARMS}
    if any(value is None for value in point.values()):
        raise RuntimeError("Validation point estimate kehilangan kelas")

    comparisons = ("AF2CUE1_vs_AF2BASE", "AF2CUE1_vs_AF2SPDS")
    deltas = {
        comparison: {metric: np.empty(iterations, dtype=np.float64) for metric in METRICS}
        for comparison in comparisons
    }
    rng = np.random.default_rng(seed)
    accepted = 0
    attempts = 0
    maximum_attempts = iterations * 50
    while accepted < iterations and attempts < maximum_attempts:
        attempts += 1
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        sample = np.concatenate([clusters[int(index)] for index in chosen])
        sample_values = {
            arm: _headline(prepared[arm], sample, class_count) for arm in ARMS
        }
        if any(value is None for value in sample_values.values()):
            continue
        for comparator, comparison in (
            ("AF2BASE", "AF2CUE1_vs_AF2BASE"),
            ("AF2SPDS", "AF2CUE1_vs_AF2SPDS"),
        ):
            for metric in METRICS:
                deltas[comparison][metric][accepted] = (
                    sample_values["AF2CUE1"][metric] - sample_values[comparator][metric]
                )
        accepted += 1
        if accepted % 100 == 0:
            print(f"PAIRED PARENT BOOTSTRAP {accepted}/{iterations}", flush=True)
    if accepted != iterations:
        raise RuntimeError(f"Bootstrap valid tidak cukup: {accepted}/{iterations}")

    summaries = {}
    for comparator, comparison in (
        ("AF2BASE", "AF2CUE1_vs_AF2BASE"),
        ("AF2SPDS", "AF2CUE1_vs_AF2SPDS"),
    ):
        summaries[comparison] = {}
        for metric in METRICS:
            values = deltas[comparison][metric]
            summaries[comparison][metric] = {
                "point_delta": point["AF2CUE1"][metric] - point[comparator][metric],
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "probability_positive": float(np.mean(values > 0.0)),
                "probability_nonnegative": float(np.mean(values >= 0.0)),
            }
    return {
        "iterations": iterations,
        "seed": seed,
        "unit": "source_parent_cluster",
        "independent_parents": len(parents),
        "validation_images": len(names),
        "images_per_parent_min": min(map(len, clusters)),
        "images_per_parent_max": max(map(len, clusters)),
        "attempts": attempts,
        "rejected_missing_class_samples": attempts - accepted,
        "custom_point_metrics": point,
        "comparisons": summaries,
    }


def _parent_map(data_root: Path) -> tuple[dict[str, str], str]:
    manifest_path = data_root / "faruq_grouped_manifest.json"
    rows = _load_json(manifest_path, "Faruq grouped manifest")
    mapping = {}
    for row in rows:
        if row.get("output_split") != "val":
            continue
        name = Path(str(row["output_image"])).name
        if name in mapping and mapping[name] != str(row["source_parent_id"]):
            raise RuntimeError(f"Parent ambigu untuk {name}")
        mapping[name] = str(row["source_parent_id"])
    return mapping, _sha256(manifest_path)


def run_af2cue1_parent_bootstrap(
    project_root: str | Path,
    data_root: str | Path,
    original_root: str | Path,
    refinement_root: str | Path,
    output: str | Path,
    *,
    device: str | None = None,
    iterations: int = 1000,
    seed: int = 20260829,
) -> dict[str, Any]:
    project_root = Path(project_root).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    original_root = Path(original_root).expanduser().resolve()
    refinement_root = Path(refinement_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if (data_root / "test").exists():
        raise RuntimeError("Test tidak boleh tersedia pada audit validation-only")
    for required in (data_root / "data.yaml", data_root / "val/images", data_root / "val/labels"):
        if not required.exists():
            raise FileNotFoundError(required)

    contracts = _load_contracts(project_root, original_root, refinement_root)
    parent_by_image, manifest_hash = _parent_map(data_root)
    reports = {}
    report_root = output.parent / "af2cue1_parent_bootstrap_reports"
    for arm in ARMS:
        print(f"VALIDATION INFERENCE: {arm}", flush=True)
        reports[arm] = _evaluate_arm(
            arm,
            contracts[arm],
            data_root,
            manifest_hash,
            report_root / f"{arm}_seed42_observations.json",
            device=device,
        )
    bootstrap = paired_parent_bootstrap(
        {arm: reports[arm]["prediction_observations"] for arm in ARMS},
        parent_by_image,
        iterations=iterations,
        seed=seed,
    )
    comparisons = bootstrap["comparisons"]
    supportive = (
        comparisons["AF2CUE1_vs_AF2BASE"]["macro_map50_95"]["probability_positive"] >= 0.95
        and comparisons["AF2CUE1_vs_AF2BASE"]["bottom3_class_map50_95"]["point_delta"] >= 0.0
        and comparisons["AF2CUE1_vs_AF2BASE"]["worst_class_map50_95"]["point_delta"] >= 0.0
        and comparisons["AF2CUE1_vs_AF2SPDS"]["macro_map50_95"]["probability_positive"] >= 0.95
    )
    result = {
        "format": "coffee_detector.af2cue1.parent_bootstrap.validation.v1",
        "scope": "reused_validation_posthoc_fixed_checkpoints",
        "formal_frozen_decision": "FAIL_KILL_GATE",
        "exploratory_research_status": (
            "PARENT_BOOTSTRAP_SUPPORTIVE" if supportive else "PARENT_BOOTSTRAP_INCONCLUSIVE"
        ),
        "contracts": {
            arm: {
                "report": str(contracts[arm]["report"]),
                "report_sha256": contracts[arm]["report_sha256"],
                "checkpoint": str(contracts[arm]["checkpoint"]),
                "checkpoint_sha256": contracts[arm]["checkpoint_sha256"],
            }
            for arm in ARMS
        },
        "endpoint_calibration": {
            arm: reports[arm]["historical_endpoint_deltas"] for arm in ARMS
        },
        "paired_parent_bootstrap": bootstrap,
        "training_executed": False,
        "test_opened": False,
        "claim_boundary": (
            "Supportive post-hoc uncertainty analysis on reused validation; not an independent "
            "confirmation dataset and does not override the frozen decision."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AF2CUE1 validation parent bootstrap")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--refinement-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    result = run_af2cue1_parent_bootstrap(
        args.project_root,
        args.data_root,
        args.original_root,
        args.refinement_root,
        args.output,
        device=args.device,
        iterations=args.iterations,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
