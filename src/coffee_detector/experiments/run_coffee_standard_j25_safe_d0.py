"""Fresh native-YOLO causal control for the J25 AF2LUMSAFE package."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from coffee_detector.af2_luminance import (
    AF2LuminanceSafeDetectionModel,
    EpochWeightedSampler,
    identity_repeat_factor_weights,
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
    _yaml,
    validate_j25_development,
)
from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_safe import (
    CONFIG as SAFE_CONFIG,
    PROTOCOL as SAFE_PROTOCOL,
    _class_sets_from_root,
)
from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_strength_sweep import (
    PROTOCOL as STRENGTH_PROTOCOL,
    TARGET_CLASS,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import (
    EXPECTED_AF2,
    OFFICIAL_YOLO26N_SHA256,
    _parameter_count,
    _require_official_pretrained,
    _sha256,
    _source_class_count,
    _state_fingerprint,
)
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/SAFED0.yaml"
ARM = "SAFED0"
PROTOCOL = "coffee-standard-j25-safe-d0-seed42-v1"
DIRECT_PROTOCOL = "coffee-standard-j25-train-siblings-af2-direct-seed42-v2"
NC = 25


def _build_native(pretrained_checkpoint: Path, seed: int):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("Official pretrained source bukan nc=80")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DetectionModel(str(MODEL_YAML), ch=3, nc=NC, verbose=False)
        model.load(source)
    return model


def build_sampler_audit(data_root: str | Path, output: str | Path) -> dict:
    root = Path(data_root).expanduser().resolve()
    config = _yaml(CONFIG)
    images, class_sets = _class_sets_from_root(root)
    _, summary = identity_repeat_factor_weights(
        images,
        class_sets,
        class_count=NC,
        maximum_repeat=float(config["sampler"]["maximum_repeat"]),
    )
    gates = {
        "exact_695_train_derivatives": summary["images"] == 695,
        "exact_315_train_source_identities": summary["source_identities"] == 315,
        "all_25_train_classes_present": len(summary["class_identity_counts"]) == NC
        and all(value > 0 for value in summary["class_identity_counts"].values()),
        "repeat_factors_finite_positive": summary["identity_repeat_min"] >= 1.0
        and summary["identity_repeat_max"]
        <= float(config["sampler"]["maximum_repeat"]),
        "validation_and_test_not_read": True,
    }
    payload = {
        **summary,
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["decision"] != "PASS":
        raise RuntimeError(f"Sampler audit gagal: {[k for k,v in gates.items() if not v]}")
    return payload


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Kontrol pertama dikunci seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    config, safe_config = _yaml(CONFIG), _yaml(SAFE_CONFIG)
    if config.get("frontend") != "none":
        raise RuntimeError("SAFED0 harus tanpa frontend")
    if config.get("train") != safe_config.get("train"):
        raise RuntimeError("Schedule/augmentation berbeda dari AF2LUMSAFE")
    if config.get("sampler") != safe_config.get("sampler"):
        raise RuntimeError("Sampler berbeda dari AF2LUMSAFE")
    if safe_config.get("afab") != EXPECTED_AF2:
        raise RuntimeError("Reference AF2LUMSAFE berubah")

    native = _build_native(checkpoint, seed)
    af2 = AFABConfig.from_mapping(safe_config["afab"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        safe = AF2LuminanceSafeDetectionModel(
            str(MODEL_YAML), ch=3, nc=NC, verbose=False, afab=af2
        )
        from ultralytics import YOLO

        source = YOLO(str(checkpoint)).model
        load_af2_luminance_weights(safe, source)
    native_state, safe_state = native.state_dict(), safe.state_dict()
    same_keys = list(native_state) == list(safe_state)
    same_tensors = same_keys and all(
        torch.equal(native_state[key].cpu(), safe_state[key].cpu()) for key in native_state
    )
    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "same_model_yaml_as_af2lumsafe": config["model"] == safe_config["model"],
        "same_training_schedule_as_af2lumsafe": config["train"] == safe_config["train"],
        "same_sampler_as_af2lumsafe": config["sampler"] == safe_config["sampler"],
        "native_state_keys_match_safe_detector": same_keys,
        "native_state_tensors_match_safe_detector": bool(same_tensors),
        "native_parameter_count_match_safe_detector": _parameter_count(native)
        == _parameter_count(safe),
        "no_af2_frontend": not hasattr(native, "af2_luminance_safe")
        and not hasattr(native, "af2_luminance"),
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.coffee_standard_j25.safe_d0.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "common_initialized_detector_state_sha256": _state_fingerprint(native),
        "config_sha256": _sha256(CONFIG),
        "af2lumsafe_config_sha256": _sha256(SAFE_CONFIG),
        "af2lumsafe_reference_protocol": SAFE_PROTOCOL,
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


def _trainer(
    *,
    pretrained: Path,
    seed: int,
    fingerprint: str,
    expected_sampler: dict,
    maximum_repeat: float,
):
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import DetectionModel

    class J25SafeD0Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if bool(getattr(self.args, "resume", False)):
                model = DetectionModel(
                    str(MODEL_YAML),
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                )
                if weights:
                    model.load(weights)
                return self.set_model_names_for_load(model)
            model = _build_native(pretrained, seed)
            if _state_fingerprint(model) != fingerprint:
                raise RuntimeError("Initial state berbeda dari static preflight")
            return self.set_model_names_for_load(model)

        def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if rank != -1:
                raise RuntimeError("SAFED0 dikunci single GPU")
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            class_sets = [
                {int(value) for value in label["cls"].reshape(-1).tolist()}
                for label in dataset.labels
            ]
            weights, summary = identity_repeat_factor_weights(
                dataset.im_files,
                class_sets,
                class_count=NC,
                maximum_repeat=maximum_repeat,
            )
            for key in ("images", "source_identities", "class_identity_counts", "threshold"):
                if summary[key] != expected_sampler[key]:
                    raise RuntimeError(f"Runtime sampler berbeda pada {key}")
            sampler = EpochWeightedSampler(weights, len(dataset), seed=20260919 + seed)
            workers = min(
                os.cpu_count() or 1,
                self.args.workers,
                max(1, (len(dataset) + batch_size - 1) // batch_size),
            )
            loader = InfiniteDataLoader(
                dataset=dataset,
                batch_size=min(batch_size, len(dataset)),
                shuffle=False,
                num_workers=workers,
                sampler=sampler,
                prefetch_factor=4 if workers > 0 else None,
                pin_memory=torch.cuda.device_count() > 0,
                collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(6148914691236517205),
                drop_last=False,
            )
            if not getattr(self, "_safe_d0_epoch_callback", False):
                def set_sampler_epoch(trainer):
                    trainer.train_loader.sampler.set_epoch(trainer.epoch)
                    trainer.train_loader.reset()

                self.add_callback("on_train_epoch_start", set_sampler_epoch)
                self._safe_d0_epoch_callback = True
            return loader

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    return J25SafeD0Trainer


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
        raise RuntimeError("SAFED0 hanya untuk J25 train-siblings v2")
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static = run_static_preflight(checkpoint, destination / "static_preflight.json", seed=seed)
    sampler = build_sampler_audit(root, destination / "sampler_audit.json")
    config = _yaml(CONFIG)
    train_args = dict(config["train"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = destination / "val_reports" / f"{ARM}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.coffee_standard_j25.safe_d0.arm_contract.v1",
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
        "af2lumsafe_config_sha256": static["af2lumsafe_config_sha256"],
        "af2lumsafe_reference_protocol": static["af2lumsafe_reference_protocol"],
        "sampler_audit_sha256": _sha256(destination / "sampler_audit.json"),
        "train": train_args,
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _json(result_path, "SAFED0 result")
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
                    pretrained=checkpoint,
                    seed=seed,
                    fingerprint=static["common_initialized_detector_state_sha256"],
                    expected_sampler=sampler,
                    maximum_repeat=float(config["sampler"]["maximum_repeat"]),
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
        "format": "coffee_detector.coffee_standard_j25.safe_d0.arm_result.v1",
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
    safe_strength_summary: str | Path,
    control_result: str | Path,
    output: str | Path,
) -> dict:
    direct = _json(direct_result, "D0DIRECT")
    strength = _json(safe_strength_summary, "AF2LUMSAFE strength summary")
    control = _json(control_result, "SAFED0")
    if direct.get("protocol") != DIRECT_PROTOCOL:
        raise RuntimeError("D0 protocol salah")
    if strength.get("protocol") != STRENGTH_PROTOCOL:
        raise RuntimeError("Strength summary protocol salah")
    if control.get("protocol") != PROTOCOL:
        raise RuntimeError("SAFED0 protocol salah")
    if direct.get("test_images_accessed") is not False:
        raise RuntimeError("D0 test lock gagal")
    if strength.get("test_opened") is not False or control.get("test_images_accessed") is not False:
        raise RuntimeError("Test lock gagal")
    values = {
        "D0DIRECT": direct["metrics"],
        "SAFED0": control["metrics"],
        "AF2LUMSAFE_lambda0": {
            metric: float(strength["values"]["0p00"][metric]) for metric in METRICS
        },
    }
    safe_target = float(strength["values"]["0p00"]["target_class_map50_95"])
    target_values = {
        "AF2LUMSAFE_lambda0": safe_target,
        "SAFED0": float(control["target_class_map50_95"]),
    }
    policy_effect = {
        metric: values["SAFED0"][metric] - values["D0DIRECT"][metric] for metric in METRICS
    }
    stochastic_af2_effect = {
        metric: values["AF2LUMSAFE_lambda0"][metric] - values["SAFED0"][metric]
        for metric in METRICS
    }
    if _dominates(values["SAFED0"], values["AF2LUMSAFE_lambda0"]):
        decision = "SAFE_POLICY_DOMINATES_STOCHASTIC_AF2"
    elif _dominates(values["AF2LUMSAFE_lambda0"], values["SAFED0"]):
        decision = "STOCHASTIC_AF2_ADDS_AGGREGATE_BENEFIT"
    else:
        decision = "PARETO_TRADEOFF"
    payload = {
        "format": "coffee_detector.coffee_standard_j25.safe_d0.seed42_decision.v1",
        "protocol": PROTOCOL,
        "values": values,
        "target_class": TARGET_CLASS,
        "target_values": target_values,
        "safe_policy_effect_vs_d0": policy_effect,
        "stochastic_af2_training_effect_at_raw_inference": stochastic_af2_effect,
        "target_recovered_without_af2": target_values["SAFED0"] > safe_target,
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
    parser.add_argument("--safe-strength-summary")
    parser.add_argument("--control-result")
    parser.add_argument("--decision-output")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    if args.decision_output:
        required = (args.direct_result, args.safe_strength_summary, args.control_result)
        if any(value is None for value in required):
            parser.error("Decision mode requires three result paths")
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
