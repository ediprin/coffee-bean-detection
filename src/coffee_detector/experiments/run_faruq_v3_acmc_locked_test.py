"""One-time, inference-only locked-test comparison of D0FT and ACMC1."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

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


def _validate_authority(confirmation: dict, eligibility: dict) -> None:
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
    if (
        eligibility.get("format")
        != "coffee_detector.faruq_locked_test_eligibility.v1"
        or eligibility.get("decision") != "PASS"
        or eligibility.get("next_action")
        != "AUTHORIZE_FROZEN_ACMC_TEST_INFERENCE"
        or eligibility.get("training_executed") is not False
        or eligibility.get("inference_executed") is not False
        or not all(eligibility.get("gates", {}).values())
    ):
        raise RuntimeError("Test Faruq bebas-leakage tidak lolos eligibility gate")


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
    metrics = YOLO(str(checkpoint)).val(**kwargs)
    results = {key: float(value) for key, value in metrics.results_dict.items()}
    box = getattr(metrics, "box", None)
    if box is None or getattr(box, "ap", None) is None:
        raise RuntimeError("Evaluator tidak menghasilkan box AP")
    results.update(
        _classwise_summary(box, {index: name for index, name in enumerate(SNI21_CLASSES)})
    )
    if results.get("classes_without_ground_truth"):
        raise RuntimeError("Locked test kehilangan ground truth kelas")
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
    _validate_authority(confirmation, eligibility)
    test_manifest_hash = _sha256(manifest)

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
        ):
            raise RuntimeError("Final locked-test cache tidak kompatibel")
        print(f"REUSE FINAL LOCKED TEST: {summary_path}", flush=True)
        return cached

    reports_root = output_root / "reports"
    per_seed: dict[str, dict] = {}
    for index, seed in enumerate(frozen_seeds):
        results = {}
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

    criteria = {
        "macro_head_gain_positive": aggregate["macro_map50_95"]["head_delta_mean"] > 0.0,
        "macro_head_improved_at_least_2_of_3": aggregate["macro_map50_95"]["head_improved_seeds"] >= 2,
        "bottom3_head_mean_not_lower": aggregate["bottom3_class_map50_95"]["head_delta_mean"] >= 0.0,
        "worst_head_mean_drop_no_more_than_1_point": aggregate["worst_class_map50_95"]["head_delta_mean"] >= -0.01,
    }
    conclusion = "CONFIRMED" if all(criteria.values()) else "NOT_CONFIRMED"
    payload = {
        "protocol": "faruq-v3-acmc-single-locked-test-v1",
        "status": "complete",
        "conclusion": conclusion,
        "seeds": list(frozen_seeds),
        "test_manifest_sha256": test_manifest_hash,
        "checkpoint_hashes": {arm: list(values) for arm, values in checkpoint_hashes.items()},
        "test_images_accessed": True,
        "test_opened": True,
        "training_executed": False,
        "further_tuning_authorized": False,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "classwise": classwise,
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
        seeds=tuple(args.seeds),
        device=args.device,
        authorize_test=args.authorize_test,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
