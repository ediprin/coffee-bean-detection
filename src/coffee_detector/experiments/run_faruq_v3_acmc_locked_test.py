"""One-time, inference-only locked-test comparison of D0FT and ACMC1."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from coffee_detector.evaluate import _classwise_summary
from coffee_detector.experiments.run_faruq_v3_acmc import METRICS
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES


FROZEN_SEEDS = (42, 123, 2026)


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_authority(
    confirmation: dict,
    eligibility: dict,
    amendment: dict | None,
    eligibility_hash: str,
) -> str:
    if (
        confirmation.get("protocol")
        != "faruq-v3-acmc-paired-optimization-confirmation-v1"
        or confirmation.get("decision") != "PASS"
        or confirmation.get("next_action")
        != "AUTHORIZE_SINGLE_LOCKED_TEST_EVALUATION"
        or tuple(confirmation.get("seeds", [])) != FROZEN_SEEDS
        or confirmation.get("evaluation_split") != "val"
        or confirmation.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Summary validation tidak mengotorisasi locked test")
    if eligibility.get("format") != "coffee_detector.faruq_locked_test_eligibility.v1":
        raise RuntimeError("Format eligibility test tidak dikenal")
    if eligibility.get("training_executed") is not False or eligibility.get("inference_executed") is not False:
        raise RuntimeError("Eligibility bukan audit pra-inference")
    if (
        eligibility.get("decision") == "PASS"
        and eligibility.get("next_action") == "AUTHORIZE_FROZEN_ACMC_TEST_INFERENCE"
        and all(eligibility.get("gates", {}).values())
    ):
        return "v1"
    if amendment is None:
        raise RuntimeError("Eligibility v1 FAIL memerlukan amendemen support v2")
    if (
        amendment.get("format")
        != "coffee_detector.faruq_locked_test_amendment.v2"
        or amendment.get("decision") != "PASS"
        or amendment.get("next_action")
        != "AUTHORIZE_V2_FROZEN_ACMC_TEST_INFERENCE"
        or amendment.get("model_inference_executed") is not False
        or amendment.get("training_executed") is not False
        or amendment.get("further_tuning_authorized") is not False
        or amendment.get("source_v1_eligibility_sha256") != eligibility_hash
        or not all(amendment.get("gates", {}).values())
    ):
        raise RuntimeError("Amendemen support v2 tidak mengotorisasi inference")
    return "v2"


def _ground_truth(label_path: Path, width: int, height: int) -> list[dict]:
    targets = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Locked test hanya menerima box YOLO: {label_path}")
        class_id = int(fields[0])
        x, y, box_width, box_height = map(float, fields[1:])
        targets.append(
            {
                "class_id": class_id,
                "xyxy": [
                    (x - box_width / 2) * width,
                    (y - box_height / 2) * height,
                    (x + box_width / 2) * width,
                    (y + box_height / 2) * height,
                ],
            }
        )
    return targets


def _collect_prediction_observations(model, test_root: Path, device: str | None) -> list[dict]:
    image_paths = sorted((test_root / "test/images").glob("*"))
    image_paths = [path for path in image_paths if path.is_file()]
    if not image_paths:
        raise FileNotFoundError("Locked test tidak memiliki gambar")
    kwargs = {
        "source": [str(path) for path in image_paths],
        "imgsz": 640,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 500,
        "stream": True,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    observations = []
    for result in model.predict(**kwargs):
        image_path = Path(result.path)
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = test_root / "test/labels" / image_path.with_suffix(".txt").name
        targets = _ground_truth(label_path, width, height)
        boxes = result.boxes
        predictions = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            predictions = [
                {
                    "class_id": int(class_id),
                    "score": float(score),
                    "xyxy": [float(value) for value in box],
                }
                for box, score, class_id in zip(xyxy, scores, classes)
            ]
        observations.append(
            {"image_name": image_path.name, "targets": targets, "predictions": predictions}
        )
    observations.sort(key=lambda row: row["image_name"].lower())
    if len(observations) != len(image_paths):
        raise RuntimeError("Jumlah hasil predict tidak sama dengan locked-test images")
    return observations


def _iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None).prod(axis=2)
    left_area = np.clip(left[:, 2:] - left[:, :2], 0.0, None).prod(axis=1)
    right_area = np.clip(right[:, 2:] - right[:, :2], 0.0, None).prod(axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / np.clip(union, 1e-12, None)


IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _precompute_observation_stats(observations: list[dict]) -> list[list[dict]]:
    prepared = []
    for observation in observations:
        class_rows = []
        for class_id in range(len(SNI21_CLASSES)):
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
                used = set()
                for prediction_index in range(len(predictions)):
                    if not len(targets):
                        break
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
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-12)
    recall_curve = np.concatenate(([0.0], recall, [1.0]))
    precision_curve = np.concatenate(([1.0], precision, [0.0]))
    precision_curve = np.maximum.accumulate(precision_curve[::-1])[::-1]
    grid = np.linspace(0.0, 1.0, 101)
    return float(np.interp(grid, recall_curve, precision_curve).mean())


def _macro_map(prepared: list[list[dict]], sample: np.ndarray) -> float:
    class_values = []
    for class_id in range(len(SNI21_CLASSES)):
        rows = [prepared[int(index)][class_id] for index in sample]
        targets = sum(int(row["targets"]) for row in rows)
        if targets <= 0:
            continue
        scores = np.concatenate([row["scores"] for row in rows]) if rows else np.empty(0)
        matrices = [row["tp"] for row in rows]
        true_positive = np.concatenate(matrices, axis=0) if matrices else np.zeros((0, 10), dtype=bool)
        aps = [
            _average_precision(true_positive[:, threshold], scores, targets)
            for threshold in range(len(IOU_THRESHOLDS))
        ]
        class_values.append(float(np.mean(aps)))
    return float(np.mean(class_values)) if class_values else float("nan")


def _paired_parent_bootstrap(
    observations_by_seed: dict[str, dict], *, iterations: int = 1000, seed: int = 20260809
) -> dict:
    prepared = {}
    image_names = None
    for seed_key, record in observations_by_seed.items():
        prepared[seed_key] = {}
        for arm in ("D0FT", "ACMC1"):
            observations = record[arm]
            names = [row["image_name"] for row in observations]
            if image_names is None:
                image_names = names
            elif names != image_names:
                raise RuntimeError("Urutan parent observations berbeda antar-checkpoint")
            prepared[seed_key][arm] = _precompute_observation_stats(observations)
    assert image_names is not None
    count = len(image_names)
    full = np.arange(count)
    per_seed_point = {
        seed_key: {
            arm: _macro_map(prepared[seed_key][arm], full)
            for arm in ("D0FT", "ACMC1")
        }
        for seed_key in prepared
    }
    point_delta = statistics.mean(
        values["ACMC1"] - values["D0FT"] for values in per_seed_point.values()
    )
    generator = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = generator.integers(0, count, size=count)
        seed_deltas = []
        for seed_key in prepared:
            left = _macro_map(prepared[seed_key]["D0FT"], sample)
            right = _macro_map(prepared[seed_key]["ACMC1"], sample)
            seed_deltas.append(right - left)
        deltas[iteration] = statistics.mean(seed_deltas)
        if (iteration + 1) % 100 == 0:
            print(f"PAIRED PARENT BOOTSTRAP {iteration + 1}/{iterations}", flush=True)
    return {
        "iterations": iterations,
        "seed": seed,
        "independent_parents": count,
        "custom_macro_point_delta": point_delta,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "probability_positive": float(np.mean(deltas > 0.0)),
        "per_seed_custom_macro": per_seed_point,
    }


def _evaluate_checkpoint(
    checkpoint: Path,
    data_yaml: Path,
    output: Path,
    *,
    checkpoint_hash: str,
    test_manifest_hash: str,
    device: str | None,
) -> dict:
    if output.is_file():
        cached = _load_json(output, "Cached locked-test report")
        if (
            cached.get("checkpoint_sha256") != checkpoint_hash
            or cached.get("test_manifest_sha256") != test_manifest_hash
            or cached.get("split") != "test"
            or not isinstance(cached.get("prediction_observations"), list)
        ):
            raise RuntimeError(f"Report test cache tidak kompatibel: {output}")
        print(f"REUSE LOCKED TEST REPORT: {output.name}", flush=True)
        return cached

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang") from error

    kwargs = {
        "data": str(data_yaml),
        "split": "test",
        "imgsz": 640,
        "batch": 16,
        "workers": 2,
        "max_det": 500,
        "plots": False,
        "verbose": True,
    }
    if device is not None:
        kwargs["device"] = device
    model = YOLO(str(checkpoint))
    metrics = model.val(**kwargs)
    results = {key: float(value) for key, value in metrics.results_dict.items()}
    box = getattr(metrics, "box", None)
    if box is None or getattr(box, "ap", None) is None:
        raise RuntimeError("Evaluator tidak menghasilkan box AP")
    results.update(
        _classwise_summary(box, {index: name for index, name in enumerate(SNI21_CLASSES)})
    )
    if results.get("classes_without_ground_truth"):
        raise RuntimeError("Locked test kehilangan ground truth kelas")
    observations = _collect_prediction_observations(model, data_yaml.parent, device)
    payload = {
        "protocol": "faruq-v3-acmc-locked-test-report-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "test_manifest_sha256": test_manifest_hash,
        "data": str(data_yaml.parent),
        "split": "test",
        "training_executed": False,
        "test_images_accessed": True,
        "metrics": results,
        "prediction_observations": observations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_faruq_v3_acmc_locked_test(
    test_root: str | Path,
    eligibility_summary: str | Path,
    confirmation_summary: str | Path,
    output_root: str | Path,
    d0ft_checkpoints: tuple[str | Path, ...],
    acmc_checkpoints: tuple[str | Path, ...],
    *,
    amendment_summary: str | Path | None = None,
    seeds: tuple[int, ...] = FROZEN_SEEDS,
    device: str | None = None,
    authorize_test: bool = False,
) -> dict:
    """Evaluate the six frozen checkpoints once; never train or tune."""

    frozen_seeds = tuple(int(seed) for seed in seeds)
    if frozen_seeds != FROZEN_SEEDS:
        raise ValueError(f"Locked test dikunci pada seed {FROZEN_SEEDS}")
    if not authorize_test:
        raise RuntimeError("Locked test belum diotorisasi secara eksplisit")
    if len(d0ft_checkpoints) != 3 or len(acmc_checkpoints) != 3:
        raise ValueError("Diperlukan tepat tiga checkpoint D0FT dan tiga ACMC1")

    test_root = Path(test_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    data_yaml = test_root / "data.yaml"
    manifest = test_root / "faruq_locked_test_manifest.json"
    if not data_yaml.is_file() or not manifest.is_file():
        raise FileNotFoundError("Paket locked test belum lengkap")
    confirmation = _load_json(confirmation_summary, "ACMC paired confirmation")
    eligibility = _load_json(eligibility_summary, "Locked-test eligibility")
    amendment = (
        _load_json(amendment_summary, "Locked-test amendment v2")
        if amendment_summary is not None
        else None
    )
    eligibility_hash = _sha256(eligibility_summary)
    protocol_version = _validate_authority(
        confirmation, eligibility, amendment, eligibility_hash
    )
    test_manifest_hash = _sha256(manifest)
    authority_hashes = {
        "eligibility": eligibility_hash,
        "confirmation": _sha256(confirmation_summary),
        "amendment": _sha256(amendment_summary) if amendment_summary is not None else None,
    }

    checkpoint_map = {
        "D0FT": tuple(Path(path).expanduser().resolve() for path in d0ft_checkpoints),
        "ACMC1": tuple(Path(path).expanduser().resolve() for path in acmc_checkpoints),
    }
    for arm, paths in checkpoint_map.items():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint {arm} tidak ditemukan: {path}")
    checkpoint_hashes = {
        arm: tuple(_sha256(path) for path in paths)
        for arm, paths in checkpoint_map.items()
    }

    summary_path = output_root / "faruq_v3_acmc_locked_test_summary.json"
    if summary_path.is_file():
        cached = _load_json(summary_path, "Final locked-test summary")
        expected_hashes = {
            arm: list(values) for arm, values in checkpoint_hashes.items()
        }
        if (
            cached.get("status") != "complete"
            or cached.get("test_manifest_sha256") != test_manifest_hash
            or cached.get("checkpoint_hashes") != expected_hashes
            or cached.get("authority_hashes") != authority_hashes
        ):
            raise RuntimeError("Final locked-test cache tidak kompatibel")
        print(f"REUSE FINAL LOCKED TEST: {summary_path}", flush=True)
        return cached

    reports_root = output_root / "reports"
    per_seed: dict[str, dict] = {}
    bootstrap_observations: dict[str, dict] = {}
    for index, seed in enumerate(frozen_seeds):
        results = {}
        bootstrap_observations[str(seed)] = {}
        for arm in ("D0FT", "ACMC1"):
            checkpoint = checkpoint_map[arm][index]
            print(f"LOCKED TEST {arm} seed={seed}", flush=True)
            report = _evaluate_checkpoint(
                checkpoint,
                data_yaml,
                reports_root / f"{arm}_seed{seed}_test.json",
                checkpoint_hash=checkpoint_hashes[arm][index],
                test_manifest_hash=test_manifest_hash,
                device=device,
            )
            results[arm] = report["metrics"]
            bootstrap_observations[str(seed)][arm] = report["prediction_observations"]
        deltas = {
            metric: float(results["ACMC1"][metric]) - float(results["D0FT"][metric])
            for metric in METRICS
        }
        per_seed[str(seed)] = {"results": results, "head_deltas_acmc1_vs_d0ft": deltas}

    aggregate = {}
    for metric in METRICS:
        d0ft_values = [float(per_seed[str(seed)]["results"]["D0FT"][metric]) for seed in frozen_seeds]
        acmc_values = [float(per_seed[str(seed)]["results"]["ACMC1"][metric]) for seed in frozen_seeds]
        deltas = [right - left for left, right in zip(d0ft_values, acmc_values)]
        aggregate[metric] = {
            "d0ft_mean": statistics.mean(d0ft_values),
            "d0ft_std": statistics.stdev(d0ft_values),
            "acmc1_mean": statistics.mean(acmc_values),
            "acmc1_std": statistics.stdev(acmc_values),
            "head_delta_mean": statistics.mean(deltas),
            "head_delta_std": statistics.stdev(deltas),
            "head_delta_min": min(deltas),
            "head_improved_seeds": sum(delta > 0.0 for delta in deltas),
        }

    classwise = {}
    for class_name in SNI21_CLASSES:
        arm_values = {}
        for arm in ("D0FT", "ACMC1"):
            values = [
                float(per_seed[str(seed)]["results"][arm]["map50_95_by_class"][class_name])
                for seed in frozen_seeds
            ]
            arm_values[arm] = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
        arm_values["delta_mean"] = arm_values["ACMC1"]["mean"] - arm_values["D0FT"]["mean"]
        classwise[class_name] = arm_values

    bootstrap = (
        _paired_parent_bootstrap(bootstrap_observations)
        if protocol_version == "v2"
        else None
    )
    criteria = {
        "macro_head_gain_positive": aggregate["macro_map50_95"]["head_delta_mean"] > 0.0,
        "macro_head_improved_at_least_2_of_3": aggregate["macro_map50_95"]["head_improved_seeds"] >= 2,
    }
    if bootstrap is not None:
        criteria["paired_parent_bootstrap_probability_at_least_95_percent"] = (
            bootstrap["probability_positive"] >= 0.95
        )
    else:
        criteria.update(
            {
                "bottom3_head_mean_not_lower": aggregate["bottom3_class_map50_95"]["head_delta_mean"] >= 0.0,
                "worst_head_mean_drop_no_more_than_1_point": aggregate["worst_class_map50_95"]["head_delta_mean"] >= -0.01,
            }
        )
    conclusion = "CONFIRMED" if all(criteria.values()) else "NOT_CONFIRMED"
    payload = {
        "protocol": f"faruq-v3-acmc-single-locked-test-{protocol_version}",
        "protocol_version": protocol_version,
        "status": "complete",
        "conclusion": conclusion,
        "seeds": list(frozen_seeds),
        "test_manifest_sha256": test_manifest_hash,
        "authority_hashes": authority_hashes,
        "checkpoint_hashes": {arm: list(values) for arm, values in checkpoint_hashes.items()},
        "test_images_accessed": True,
        "test_opened": True,
        "training_executed": False,
        "further_tuning_authorized": False,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "classwise": classwise,
        "paired_parent_bootstrap": bootstrap,
        "metric_roles": {
            "primary": "macro_map50_95",
            "secondary_descriptive_only": (
                ["bottom3_class_map50_95", "worst_class_map50_95"]
                if protocol_version == "v2"
                else []
            ),
        },
        "criteria": criteria,
        "next_action": "FINALIZE_THESIS_RESULT_NO_FURTHER_TUNING",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time Faruq-v3 ACMC locked-test evaluation")
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--eligibility-summary", required=True)
    parser.add_argument("--confirmation-summary", required=True)
    parser.add_argument("--amendment-summary")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--d0ft-checkpoints", nargs=3, required=True)
    parser.add_argument("--acmc-checkpoints", nargs=3, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, default=list(FROZEN_SEEDS))
    parser.add_argument("--device")
    parser.add_argument("--authorize-test", action="store_true")
    args = parser.parse_args()
    result = run_faruq_v3_acmc_locked_test(
        args.test_root,
        args.eligibility_summary,
        args.confirmation_summary,
        args.output_root,
        tuple(args.d0ft_checkpoints),
        tuple(args.acmc_checkpoints),
        amendment_summary=args.amendment_summary,
        seeds=tuple(args.seeds),
        device=args.device,
        authorize_test=args.authorize_test,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
