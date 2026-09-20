"""Fresh J25 CWCF1 conditional hard-negative repair screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coffee_detector.data.prepare_coffee_standard_primary import J25_CLASSES
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    METRICS,
    MODEL_YAML,
    NC,
    SEED,
    _epochs,
    _json,
    _yaml,
    validate_j25_development,
)
from coffee_detector.experiments.run_coffee_standard_j25_cwcf import (
    PROTOCOL as CWCF1_PROTOCOL,
    _build_native,
    _raw_predictions,
    _trainer,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import (
    OFFICIAL_YOLO26N_SHA256,
    _parameter_count,
    _require_official_pretrained,
    _sha256,
)
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)
from coffee_detector.j25_cwcf import (
    BLACK_BROKEN_CONFUSION_CLASSES,
    CWCFConfig,
    build_cwcf_model,
    conditional_confusion_cross_entropy,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/CWCFHNC1.yaml"
CWCF1_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/CWCF1.yaml"
ARM = "CWCFHNC1"
PROTOCOL = "coffee-standard-j25-cwcf-hnc-seed42-v1"
TARGET_CLASS_INDEX = 7
TARGET_CLASS = J25_CLASSES[TARGET_CLASS_INDEX]


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen pertama dikunci seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    candidate_yaml, baseline_yaml = _yaml(CONFIG), _yaml(CWCF1_CONFIG)
    if candidate_yaml["model"] != baseline_yaml["model"]:
        raise RuntimeError("Model YAML CWCFHNC1 dan CWCF1 berbeda")
    if candidate_yaml["train"] != baseline_yaml["train"]:
        raise RuntimeError("Jadwal CWCFHNC1 dan CWCF1 berbeda")
    candidate_config = CWCFConfig.from_mapping(candidate_yaml["cwcf"])
    baseline_config = CWCFConfig.from_mapping(baseline_yaml["cwcf"])
    native, source = _build_native(checkpoint, seed)
    baseline = build_cwcf_model(
        str(MODEL_YAML), nc=NC, source=source, seed=seed, config=baseline_config
    )
    candidate = build_cwcf_model(
        str(MODEL_YAML), nc=NC, source=source, seed=seed, config=candidate_config
    )
    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)
    baseline_raw = _raw_predictions(baseline, probe)
    candidate_raw = _raw_predictions(candidate, probe)
    initial_boxes_exact = torch.equal(
        baseline_raw["boxes"], candidate_raw["boxes"]
    )
    initial_scores_exact = torch.equal(
        baseline_raw["scores"], candidate_raw["scores"]
    )

    logits = torch.randn(5, NC, requires_grad=True)
    labels = torch.tensor([7, 8, 9, 12, 0])
    conditional = conditional_confusion_cross_entropy(logits, labels)
    conditional.backward()
    selected_gradient = logits.grad[:4]
    outside_object_gradient = logits.grad[4]
    outside_columns = sorted(
        set(range(NC)).difference(BLACK_BROKEN_CONFUSION_CLASSES)
    )
    gates = {
        "official_pretrained_sha256_exact": (
            pretrained_sha == OFFICIAL_YOLO26N_SHA256
        ),
        "same_model_yaml_as_cwcf1": candidate_yaml["model"] == baseline_yaml["model"],
        "same_50_epoch_schedule_as_cwcf1": (
            candidate_yaml["train"] == baseline_yaml["train"]
        ),
        "sampler_is_none": candidate_yaml.get("sampler") == "none",
        "conditional_gain_exactly_0_05": (
            candidate_config.conditional_confusion_gain == 0.05
        ),
        "confusion_family_exact": (
            BLACK_BROKEN_CONFUSION_CLASSES == (7, 8, 9, 12)
        ),
        "same_parameter_count_as_cwcf1": (
            _parameter_count(candidate) == _parameter_count(baseline)
        ),
        "same_state_schema_as_cwcf1": (
            tuple(candidate.state_dict()) == tuple(baseline.state_dict())
        ),
        "initial_boxes_bitwise_exact_to_cwcf1": bool(initial_boxes_exact),
        "initial_scores_bitwise_exact_to_cwcf1": bool(initial_scores_exact),
        "conditional_loss_finite": bool(torch.isfinite(conditional)),
        "conditional_selected_gradients_finite_nonzero": bool(
            torch.isfinite(selected_gradient).all()
            and selected_gradient.abs().sum() > 0
        ),
        "conditional_outside_family_columns_zero": bool(
            selected_gradient[:, outside_columns].abs().sum() == 0
        ),
        "conditional_outside_family_objects_zero": bool(
            outside_object_gradient.abs().sum() == 0
        ),
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.coffee_standard_j25.cwcf_hnc.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "config_sha256": _sha256(CONFIG),
        "cwcf1_config_sha256": _sha256(CWCF1_CONFIG),
        "native_parameters": _parameter_count(native),
        "cwcf1_parameters": _parameter_count(baseline),
        "candidate_parameters": _parameter_count(candidate),
        "inference_parameters_added_vs_cwcf1": (
            _parameter_count(candidate) - _parameter_count(baseline)
        ),
        "cwcf": candidate_config.to_dict(),
        "confusion_class_indices": list(BLACK_BROKEN_CONFUSION_CLASSES),
        "confusion_class_names": [
            J25_CLASSES[index] for index in BLACK_BROKEN_CONFUSION_CLASSES
        ],
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
    if seed != SEED or not authorize_training:
        raise RuntimeError("Seed-42 dan authorization wajib valid")
    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("CWCFHNC1 hanya untuk J25 train-siblings v2")
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static = run_static_preflight(
        checkpoint, destination / "static_preflight.json", seed=seed
    )
    config = _yaml(CONFIG)
    frozen = CWCFConfig.from_mapping(config["cwcf"])
    train_args = dict(config["train"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = destination / "val_reports" / f"{ARM}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.coffee_standard_j25.cwcf_hnc.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "source_archive_sha256": dataset["source_archive_sha256"],
        "development_contract_sha256": dataset["development_contract_sha256"],
        "provenance_summary_sha256": dataset["provenance_summary_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "config_sha256": static["config_sha256"],
        "cwcf1_config_sha256": static["cwcf1_config_sha256"],
        "cwcf": frozen.to_dict(),
        "confusion_class_indices": list(BLACK_BROKEN_CONFUSION_CLASSES),
        "train": train_args,
        "sampler": "none",
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _json(result_path, "CWCFHNC1 result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing result berbeda kontrak")
        return old
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _json(contract_path, "Run contract") != contract:
        raise RuntimeError("Run directory berbeda kontrak")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    if not _run_complete(run_dir, int(train_args["epochs"])):
        from ultralytics import YOLO

        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            destination, lock_name=f"{ARM}_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                if last.is_file():
                    raise RuntimeError("last.pt tidak resumable")
                model, args = YOLO(str(MODEL_YAML)), dict(train_args)
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
            model.train(
                trainer=_trainer(pretrained=checkpoint, seed=seed, config=frozen),
                **args,
            )
        training_executed = True
    if not _run_complete(run_dir, int(train_args["epochs"])) or not best.is_file():
        raise RuntimeError("Run belum selesai secara valid")
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
        "format": "coffee_detector.coffee_standard_j25.cwcf_hnc.arm_result.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "metrics": {metric: float(metrics[metric]) for metric in METRICS},
        "map50_95_by_class": metrics["map50_95_by_class"],
        "target_class": TARGET_CLASS,
        "target_class_map50_95": float(metrics["map50_95_by_class"][TARGET_CLASS]),
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "completed_epochs": _epochs(run_dir / "results.csv"),
        "maximum_epochs": int(train_args["epochs"]),
        "training_executed_this_call": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_decision(
    cwcf1_result: str | Path,
    candidate_result: str | Path,
    output: str | Path,
) -> dict:
    baseline = _json(cwcf1_result, "CWCF1 result")
    candidate = _json(candidate_result, "CWCFHNC1 result")
    if baseline.get("protocol") != CWCF1_PROTOCOL:
        raise RuntimeError("Protocol CWCF1 reference salah")
    if candidate.get("protocol") != PROTOCOL:
        raise RuntimeError("Protocol CWCFHNC1 salah")
    if any(
        payload.get("test_images_accessed") is not False
        for payload in (baseline, candidate)
    ):
        raise RuntimeError("Test lock gagal")
    values = {"CWCF1": baseline["metrics"], ARM: candidate["metrics"]}
    deltas = {
        metric: float(candidate["metrics"][metric])
        - float(baseline["metrics"][metric])
        for metric in METRICS
    }
    baseline_target = float(baseline["map50_95_by_class"][TARGET_CLASS])
    candidate_target = float(candidate["map50_95_by_class"][TARGET_CLASS])
    target_delta = candidate_target - baseline_target
    criteria = {
        "macro_drop_no_more_than_0_5_point": deltas[METRICS[0]] >= -0.005,
        "bottom3_drop_no_more_than_0_5_point": deltas[METRICS[1]] >= -0.005,
        "target_gain_at_least_0_5_point": target_delta >= 0.005,
        "all_25_validation_classes_present": len(candidate["map50_95_by_class"]) == 25,
        "test_not_opened": True,
    }
    passed = all(criteria.values())
    payload = {
        "format": "coffee_detector.coffee_standard_j25.cwcf_hnc.seed42_decision.v1",
        "protocol": PROTOCOL,
        "values": values,
        "deltas_vs_cwcf1": deltas,
        "target_class": TARGET_CLASS,
        "target_values": {"CWCF1": baseline_target, ARM: candidate_target},
        "target_delta": target_delta,
        "criteria": criteria,
        "decision": "PASS" if passed else "FAIL_KILL_GATE",
        "next": "FREEZE_CONFIRMATION" if passed else "RETAIN_CWCF1",
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--development-contract", required=True)
    parser.add_argument("--provenance-summary", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
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
