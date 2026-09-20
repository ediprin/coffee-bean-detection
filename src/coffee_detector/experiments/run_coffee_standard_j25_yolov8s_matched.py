"""Matched YOLOv8s prior-study baseline on the frozen J25 development split.

This runner is descriptive: it does not promote or reject CWCF1.  It exists to
separate detector-family effects from the historical thesis split/protocol.
Locked test images are never extracted or evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    METRICS,
    SEED,
    _epochs,
    _json,
    validate_j25_development,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import _sha256
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/V8S_MATCHED.yaml"
SAFEAUG_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/SAFEAUG0.yaml"
ARM = "V8S_MATCHED"
PROTOCOL = "coffee-standard-j25-yolov8s-matched-seed42-v1"
TARGET_CLASS = "Biji Hitam Pecah"
NC = 25


def _yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(path)
    return payload


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen V8S-MATCHED pertama dikunci seed 42")
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    config, safeaug = _yaml(CONFIG), _yaml(SAFEAUG_CONFIG)
    from ultralytics import YOLO

    source = YOLO(str(checkpoint))
    source_model = source.model
    source_nc = int(getattr(source_model.model[-1], "nc", -1))
    names = getattr(source_model, "names", {})
    source_class_count = len(names) if isinstance(names, (dict, list)) else source_nc

    gates = {
        "seed_is_42": seed == SEED,
        "dataset_matches_safeaug0": config.get("dataset") == safeaug.get("dataset"),
        "training_schedule_exactly_matches_safeaug0": config.get("train") == safeaug.get("train"),
        "sampler_is_none": config.get("sampler") == "none",
        "pretrained_checkpoint_exists": checkpoint.is_file(),
        "pretrained_source_has_80_classes": source_nc == 80 and source_class_count == 80,
        "target_ontology_is_25_classes": NC == 25,
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.coffee_standard_j25.yolov8s_matched.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": _sha256(checkpoint),
        "source_parameters": _parameter_count(source_model),
        "config_sha256": _sha256(CONFIG),
        "safeaug0_config_sha256": _sha256(SAFEAUG_CONFIG),
        "gates": gates,
        "training_authorized": all(gates.values()),
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["decision"] != "PASS":
        raise RuntimeError(
            f"Static preflight gagal: {[key for key, value in gates.items() if not value]}"
        )
    return payload


def run_arm(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
    pretrained_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != SEED:
        raise RuntimeError("V8S-MATCHED screening dikunci seed 42")
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training")

    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("V8S-MATCHED hanya untuk J25 train-siblings v2")

    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static = run_static_preflight(checkpoint, destination / "static_preflight.json", seed=seed)

    config = _yaml(CONFIG)
    train_args = dict(config["train"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = destination / "val_reports" / f"{ARM}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    contract = {
        "format": "coffee_detector.coffee_standard_j25.yolov8s_matched.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "source_archive_sha256": dataset["source_archive_sha256"],
        "development_contract_sha256": dataset["development_contract_sha256"],
        "provenance_summary_sha256": dataset["provenance_summary_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "config_sha256": static["config_sha256"],
        "safeaug0_config_sha256": static["safeaug0_config_sha256"],
        "train": train_args,
        "sampler": "none",
        "comparison_role": "matched prior-study detector baseline; not a CWCF promotion gate",
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _json(result_path, "V8S-MATCHED result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing V8S-MATCHED result berbeda kontrak")
        return old

    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _json(contract_path, "Run contract") != contract:
        raise RuntimeError("Run directory berbeda kontrak")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False
    if not _run_complete(run_dir, int(train_args["epochs"])):
        from ultralytics import YOLO

        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            destination, lock_name=f"{ARM}_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                model = YOLO(str(last))
                model.train(resume=True, device=device)
            else:
                if last.is_file():
                    raise RuntimeError("last.pt tidak resumable")
                model = YOLO(str(checkpoint))
                args = dict(train_args)
                args.update(
                    data=str(root / "data.yaml"),
                    project=str(destination / ARM),
                    name=f"{ARM}_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=False,
                    verbose=False,
                    device=device,
                )
                model.train(**args)
        training_executed = True

    if not _run_complete(run_dir, int(train_args["epochs"])) or not best.is_file():
        raise RuntimeError("V8S-MATCHED belum selesai secara valid")

    evaluation = evaluate(
        best,
        root,
        destination / "val_reports" / f"{ARM}_seed{seed}_val.json",
        split="val",
        device=device,
    )
    metrics = evaluation["metrics"]
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")

    result = {
        "format": "coffee_detector.coffee_standard_j25.yolov8s_matched.arm_result.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "metrics": {metric: float(metrics[metric]) for metric in METRICS},
        "target_class": TARGET_CLASS,
        "target_class_map50_95": float(metrics["map50_95_by_class"][TARGET_CLASS]),
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "completed_epochs": _epochs(run_dir / "results.csv"),
        "maximum_epochs": int(train_args["epochs"]),
        "training_executed_this_call": training_executed,
        "evaluation_split": "val",
        "comparison_role": "matched prior-study detector baseline; descriptive only",
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_comparison(
    v8s_result: str | Path,
    safeaug0_result: str | Path,
    cwcf1_result: str | Path,
    output: str | Path,
) -> dict:
    v8s = _json(v8s_result, "V8S-MATCHED")
    safeaug = _json(safeaug0_result, "SAFEAUG0")
    cwcf = _json(cwcf1_result, "CWCF1")
    for label, payload in (("V8S_MATCHED", v8s), ("SAFEAUG0", safeaug), ("CWCF1", cwcf)):
        if payload.get("seed") != SEED:
            raise RuntimeError(f"{label} bukan seed 42")
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"{label} melanggar test lock")

    values = {
        "V8S_MATCHED": {metric: float(v8s["metrics"][metric]) for metric in METRICS},
        "SAFEAUG0": {metric: float(safeaug["metrics"][metric]) for metric in METRICS},
        "CWCF1": {metric: float(cwcf["metrics"][metric]) for metric in METRICS},
    }
    target_values = {
        "V8S_MATCHED": float(v8s["target_class_map50_95"]),
        "SAFEAUG0": float(safeaug["target_class_map50_95"]),
        "CWCF1": float(cwcf["target_class_map50_95"]),
    }
    payload = {
        "format": "coffee_detector.coffee_standard_j25.yolov8s_matched.comparison.v1",
        "protocol": PROTOCOL,
        "seed": SEED,
        "values": values,
        "target_class": TARGET_CLASS,
        "target_values": target_values,
        "v8s_minus_safeaug0": {
            metric: values["V8S_MATCHED"][metric] - values["SAFEAUG0"][metric]
            for metric in METRICS
        },
        "cwcf1_minus_v8s": {
            metric: values["CWCF1"][metric] - values["V8S_MATCHED"][metric]
            for metric in METRICS
        },
        "historical_reference": {
            "model": "YOLOv8s",
            "reported_validation_images": 113,
            "reported_validation_instances": 1606,
            "reported_map50": 0.773,
            "reported_map50_95": 0.616,
            "reported_biji_hitam_pecah_map50_95": 0.827,
            "head_to_head_comparable": False,
        },
        "decision": "DESCRIPTIVE_BASELINE_COMPLETE",
        "claim_boundary": (
            "V8S-MATCHED is a matched prior-study detector baseline. "
            "It is not a promotion or kill gate for CWCF1; CWCF1 must be judged "
            "against its matched YOLO26 control across frozen seeds."
        ),
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--development-contract")
    parser.add_argument("--provenance-summary")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--output-root")
    parser.add_argument("--v8s-result")
    parser.add_argument("--safeaug0-result")
    parser.add_argument("--cwcf1-result")
    parser.add_argument("--comparison-output")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()

    if args.comparison_output:
        required = (args.v8s_result, args.safeaug0_result, args.cwcf1_result)
        if any(value is None for value in required):
            parser.error("Comparison mode requires V8S, SAFEAUG0, and CWCF1 results")
        payload = build_comparison(*required, args.comparison_output)
        print(json.dumps(payload, indent=2), flush=True)
        return

    required = (
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.pretrained_checkpoint,
        args.output_root,
    )
    if any(value is None for value in required):
        parser.error("Training mode requires data, contracts, pretrained, and output")
    run_arm(
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.pretrained_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
