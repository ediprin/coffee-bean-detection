"""Fresh semantic-safe augmentation control without AF2 or repeat sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coffee_detector.af2_luminance import (
    AF2LuminanceSafeDetectionModel,
    load_af2_luminance_weights,
)
from coffee_detector.afab import AFABConfig
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    METRICS,
    MODEL_YAML,
    SEED,
    _epochs,
    _json,
    _trainer,
    _yaml,
    validate_j25_development,
)
from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_strength_sweep import (
    PROTOCOL as STRENGTH_PROTOCOL,
    TARGET_CLASS,
)
from coffee_detector.experiments.run_coffee_standard_j25_safe_d0 import (
    CONFIG as SAFE_D0_CONFIG,
    DIRECT_PROTOCOL,
    PROTOCOL as SAFE_D0_PROTOCOL,
    _build_native,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import (
    EXPECTED_AF2,
    OFFICIAL_YOLO26N_SHA256,
    _parameter_count,
    _require_official_pretrained,
    _sha256,
    _state_fingerprint,
)
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/SAFEAUG0.yaml"
ARM = "SAFEAUG0"
PROTOCOL = "coffee-standard-j25-safe-augmentation-seed42-v1"
NC = 25


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Kontrol pertama dikunci seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    config, safe_d0 = _yaml(CONFIG), _yaml(SAFE_D0_CONFIG)
    native = _build_native(checkpoint, seed)
    af2 = AFABConfig.from_mapping(EXPECTED_AF2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = AF2LuminanceSafeDetectionModel(
            str(MODEL_YAML), ch=3, nc=NC, verbose=False, afab=af2
        )
        from ultralytics import YOLO

        load_af2_luminance_weights(reference, YOLO(str(checkpoint)).model)
    native_state, reference_state = native.state_dict(), reference.state_dict()
    same_keys = list(native_state) == list(reference_state)
    same_tensors = same_keys and all(
        torch.equal(native_state[key].cpu(), reference_state[key].cpu())
        for key in native_state
    )
    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "same_model_yaml_as_safe_d0": config["model"] == safe_d0["model"],
        "same_training_schedule_as_safe_d0": config["train"] == safe_d0["train"],
        "sampler_is_explicitly_none": config.get("sampler") == "none",
        "frontend_is_none": config.get("frontend") == "none",
        "native_state_keys_match_safe_d0_detector": same_keys,
        "native_state_tensors_match_safe_d0_detector": bool(same_tensors),
        "native_parameter_count_matches_safe_d0_detector": _parameter_count(native)
        == _parameter_count(reference),
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.coffee_standard_j25.safe_augmentation.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "common_initialized_detector_state_sha256": _state_fingerprint(native),
        "config_sha256": _sha256(CONFIG),
        "safe_d0_config_sha256": _sha256(SAFE_D0_CONFIG),
        "native_parameters": _parameter_count(native),
        "gates": gates,
        "training_authorized": all(gates.values()),
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["decision"] != "PASS":
        raise RuntimeError(f"Static preflight gagal: {[k for k,v in gates.items() if not v]}")
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
        raise RuntimeError("SAFEAUG0 hanya untuk J25 train-siblings v2")
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static = run_static_preflight(checkpoint, destination / "static_preflight.json", seed=seed)
    config = _yaml(CONFIG)
    train_args = dict(config["train"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = destination / "val_reports" / f"{ARM}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.coffee_standard_j25.safe_augmentation.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "source_archive_sha256": dataset["source_archive_sha256"],
        "development_contract_sha256": dataset["development_contract_sha256"],
        "provenance_summary_sha256": dataset["provenance_summary_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "common_initialized_detector_state_sha256": static[
            "common_initialized_detector_state_sha256"
        ],
        "config_sha256": static["config_sha256"],
        "safe_d0_config_sha256": static["safe_d0_config_sha256"],
        "train": train_args,
        "sampler": "none",
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _json(result_path, "SAFEAUG0 result")
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
        with _exclusive_training_lock(destination, lock_name=f"{ARM}_seed{seed}.training.lock"):
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
                trainer=_trainer(
                    use_af2=False,
                    pretrained=checkpoint,
                    af2=AFABConfig.from_mapping(EXPECTED_AF2),
                    seed=seed,
                    fingerprint=static["common_initialized_detector_state_sha256"],
                ),
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
        "format": "coffee_detector.coffee_standard_j25.safe_augmentation.arm_result.v1",
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
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def _dominates(left: dict, right: dict) -> bool:
    pairs = [(float(left[key]), float(right[key])) for key in METRICS]
    return all(a >= b for a, b in pairs) and any(a > b for a, b in pairs)


def build_decision(
    direct_result: str | Path,
    safe_d0_result: str | Path,
    safe_strength_summary: str | Path,
    augmentation_result: str | Path,
    output: str | Path,
) -> dict:
    direct = _json(direct_result, "D0DIRECT")
    safe_d0 = _json(safe_d0_result, "SAFED0")
    strength = _json(safe_strength_summary, "AF2LUMSAFE strength summary")
    augmentation = _json(augmentation_result, "SAFEAUG0")
    if direct.get("protocol") != DIRECT_PROTOCOL:
        raise RuntimeError("D0 protocol salah")
    if safe_d0.get("protocol") != SAFE_D0_PROTOCOL:
        raise RuntimeError("SAFED0 protocol salah")
    if strength.get("protocol") != STRENGTH_PROTOCOL:
        raise RuntimeError("Strength summary protocol salah")
    if augmentation.get("protocol") != PROTOCOL:
        raise RuntimeError("SAFEAUG0 protocol salah")
    if (
        direct.get("test_images_accessed") is not False
        or safe_d0.get("test_images_accessed") is not False
        or augmentation.get("test_images_accessed") is not False
        or strength.get("test_opened") is not False
    ):
        raise RuntimeError("Test lock gagal")
    values = {
        "D0DIRECT": direct["metrics"],
        "SAFEAUG0": augmentation["metrics"],
        "SAFED0": safe_d0["metrics"],
        "AF2LUMSAFE_lambda0": {
            metric: float(strength["values"]["0p00"][metric]) for metric in METRICS
        },
    }
    target_values = {
        "SAFEAUG0": float(augmentation["target_class_map50_95"]),
        "SAFED0": float(safe_d0["target_class_map50_95"]),
        "AF2LUMSAFE_lambda0": float(
            strength["values"]["0p00"]["target_class_map50_95"]
        ),
    }
    augmentation_effect = {
        metric: values["SAFEAUG0"][metric] - values["D0DIRECT"][metric]
        for metric in METRICS
    }
    sampler_effect = {
        metric: values["SAFED0"][metric] - values["SAFEAUG0"][metric]
        for metric in METRICS
    }
    recovered = target_values["SAFEAUG0"] > target_values["SAFED0"]
    if recovered:
        decision = "SAMPLER_IMPLICATED_IN_TARGET_COLLAPSE"
    elif target_values["SAFEAUG0"] <= 0.0:
        decision = "TARGET_FAILURE_PERSISTS_WITHOUT_SAMPLER"
    else:
        decision = "TARGET_CHANGE_INCONCLUSIVE"
    payload = {
        "format": "coffee_detector.coffee_standard_j25.safe_augmentation.seed42_decision.v1",
        "protocol": PROTOCOL,
        "values": values,
        "target_class": TARGET_CLASS,
        "target_values": target_values,
        "safe_augmentation_effect_vs_d0": augmentation_effect,
        "sampler_effect_safe_d0_minus_safeaug0": sampler_effect,
        "safeaug0_dominates_safe_d0": _dominates(values["SAFEAUG0"], values["SAFED0"]),
        "safe_d0_dominates_safeaug0": _dominates(values["SAFED0"], values["SAFEAUG0"]),
        "target_recovered_without_sampler": recovered,
        "decision": decision,
        "claim_boundary": "single-seed causal component control; no locked-test claim",
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--development-contract")
    parser.add_argument("--provenance-summary")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--output-root")
    parser.add_argument("--direct-result")
    parser.add_argument("--safe-d0-result")
    parser.add_argument("--safe-strength-summary")
    parser.add_argument("--augmentation-result")
    parser.add_argument("--decision-output")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    if args.decision_output:
        required = (
            args.direct_result,
            args.safe_d0_result,
            args.safe_strength_summary,
            args.augmentation_result,
        )
        if any(value is None for value in required):
            parser.error("Decision mode requires four result paths")
        build_decision(*required, args.decision_output)
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
