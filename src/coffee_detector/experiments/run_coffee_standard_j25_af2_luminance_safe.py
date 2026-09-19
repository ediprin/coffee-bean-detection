"""Fresh J25 AF2LUM-SAFE final-package screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from coffee_detector.af2_luminance import (
    AF2LuminanceInputEnhancer,
    AF2LuminanceSafeDetectionModel,
    AF2LuminanceStochasticInputEnhancer,
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
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/AF2LUMSAFE.yaml"
ARM = "AF2LUMSAFE"
PROTOCOL = "coffee-standard-j25-af2-luminance-safe-seed42-v1"
REFERENCE_PROTOCOL = "coffee-standard-j25-train-siblings-af2-direct-seed42-v2"
LUMINANCE_PROTOCOL = "coffee-standard-j25-af2-luminance-isolation-seed42-v1"
NC = 25


def _build_detector(pretrained_checkpoint: Path, af2: AFABConfig, seed: int):
    from ultralytics import YOLO

    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("Official pretrained source bukan nc=80")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AF2LuminanceSafeDetectionModel(
            str(MODEL_YAML), ch=3, nc=NC, verbose=False, afab=af2
        )
        transfer = load_af2_luminance_weights(model, source)
    return model, transfer


def _class_sets_from_root(root: Path) -> tuple[list[Path], list[set[int]]]:
    images = sorted(
        path
        for path in (root / "train/images").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    class_sets: list[set[int]] = []
    for image in images:
        label = root / "train/labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        classes = {
            int(line.split()[0])
            for line in label.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        class_sets.append(classes)
    return images, class_sets


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
        and summary["identity_repeat_max"] <= float(config["sampler"]["maximum_repeat"]),
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
        raise ValueError("Screen pertama dikunci seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    config = _yaml(CONFIG)
    if config.get("afab") != EXPECTED_AF2:
        raise RuntimeError("Operator AF2 berubah")
    if config.get("frontend") != "luminance_shared_stochastic":
        raise RuntimeError("Frontend AF2LUM-SAFE tidak dibekukan")
    frozen_train = config.get("train", {})
    intended = {
        "mosaic": 0.0,
        "scale": 0.0,
        "hsv_h": 0.0,
        "erasing": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
    }
    if any(float(frozen_train.get(key, -1)) != value for key, value in intended.items()):
        raise RuntimeError("Augmentasi semantic-safe berubah")
    af2 = AFABConfig.from_mapping(config["afab"])
    candidate, transfer = _build_detector(checkpoint, af2, seed)

    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    source = YOLO(str(checkpoint)).model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        native = DetectionModel(str(MODEL_YAML), ch=3, nc=NC, verbose=False)
        native.load(source)
    native_state, candidate_state = native.state_dict(), candidate.state_dict()
    same_keys = list(native_state) == list(candidate_state)
    same_tensors = same_keys and all(
        torch.equal(native_state[key].cpu(), candidate_state[key].cpu())
        for key in native_state
    )

    probe = 0.15 + 0.7 * torch.rand(2, 3, 64, 64)
    safe = AF2LuminanceStochasticInputEnhancer(af2)
    deterministic = AF2LuminanceInputEnhancer(af2)
    with torch.inference_mode():
        raw = safe.forward_with_strength(probe, 0.0)
        half = safe.forward_with_strength(probe, 0.5)
        full = safe.forward_with_strength(probe, 1.0)
        expected_full = deterministic(probe)
        safe.eval()
        evaluation = safe(probe)
        ratio_before = probe[:, 0] / probe[:, 1]
        ratio_after = half[:, 0] / half[:, 1]
    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "same_model_yaml": config["model"] == str(MODEL_YAML.relative_to(REPO_ROOT)).replace("\\", "/"),
        "fresh_50_epoch_seed42_schedule": frozen_train.get("epochs") == 50,
        "same_legacy_af2_hyperparameters": config["afab"] == EXPECTED_AF2,
        "detector_state_keys_exact": same_keys,
        "detector_state_tensors_exact": bool(same_tensors),
        "detector_parameter_count_exact": _parameter_count(native)
        == _parameter_count(candidate),
        "frontend_trainable_parameters_zero": _parameter_count(safe) == 0,
        "zero_strength_is_exact_raw": torch.equal(raw, probe),
        "unit_strength_is_exact_af2_luminance": torch.equal(full, expected_full),
        "half_strength_is_exact_midpoint": torch.allclose(
            half, probe + 0.5 * (expected_full - probe), atol=2e-7, rtol=0
        ),
        "shared_strength_preserves_chromaticity": torch.allclose(
            ratio_before, ratio_after, atol=2e-6, rtol=2e-6
        ),
        "evaluation_uses_unit_strength": torch.equal(evaluation, expected_full),
        "frontend_outputs_finite": bool(torch.isfinite(half).all()),
        "semantic_safe_augmentation_exact": all(
            float(frozen_train[key]) == value for key, value in intended.items()
        ),
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.coffee_standard_j25.af2_luminance_safe.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "common_initialized_detector_state_sha256": _state_fingerprint(native),
        "config_sha256": _sha256(CONFIG),
        "native_parameters": _parameter_count(native),
        "candidate_parameters": _parameter_count(candidate),
        "weight_transfer": transfer,
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
    af2: AFABConfig,
    seed: int,
    fingerprint: str,
    expected_sampler: dict,
    maximum_repeat: float,
):
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.models.yolo.detect import DetectionTrainer

    class J25AF2LuminanceSafeTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if bool(getattr(self.args, "resume", False)):
                model = AF2LuminanceSafeDetectionModel(
                    str(MODEL_YAML), nc=self.data["nc"], ch=self.data["channels"],
                    verbose=verbose, afab=af2,
                )
                if weights:
                    load_af2_luminance_weights(model, weights)
                return self.set_model_names_for_load(model)
            model, _ = _build_detector(pretrained, af2, seed)
            if _state_fingerprint(model) != fingerprint:
                raise RuntimeError("Initial state berbeda dari static preflight")
            return self.set_model_names_for_load(model)

        def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if rank != -1:
                raise RuntimeError("AF2LUM-SAFE screen dikunci single GPU")
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
            nd = torch.cuda.device_count()
            batches = max(1, (len(dataset) + batch_size - 1) // batch_size)
            workers = min(os.cpu_count() or 1, self.args.workers, batches)
            generator = torch.Generator().manual_seed(6148914691236517205)
            loader = InfiniteDataLoader(
                dataset=dataset,
                batch_size=min(batch_size, len(dataset)),
                shuffle=False,
                num_workers=workers,
                sampler=sampler,
                prefetch_factor=4 if workers > 0 else None,
                pin_memory=nd > 0,
                collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=seed_worker,
                generator=generator,
                drop_last=False,
            )
            if not getattr(self, "_af2lum_safe_epoch_callback", False):
                def set_sampler_epoch(trainer):
                    trainer.train_loader.sampler.set_epoch(trainer.epoch)
                    # InfiniteDataLoader owns a persistent iterator.  Rebuild it
                    # after changing the epoch or the new weighted sequence
                    # would not take effect until that iterator wrapped around.
                    trainer.train_loader.reset()

                self.add_callback("on_train_epoch_start", set_sampler_epoch)
                self._af2lum_safe_epoch_callback = True
            return loader

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    return J25AF2LuminanceSafeTrainer


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
        raise RuntimeError("AF2LUM-SAFE hanya untuk J25 train-siblings v2")
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static = run_static_preflight(checkpoint, destination / "static_preflight.json", seed=seed)
    sampler = build_sampler_audit(root, destination / "sampler_audit.json")
    config = _yaml(CONFIG)
    af2 = AFABConfig.from_mapping(config["afab"])
    train_args = dict(config["train"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = destination / "val_reports" / f"{ARM}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.coffee_standard_j25.af2_luminance_safe.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "source_archive_sha256": dataset["source_archive_sha256"],
        "development_contract_sha256": dataset["development_contract_sha256"],
        "provenance_summary_sha256": dataset["provenance_summary_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "common_initialized_detector_state_sha256": static["common_initialized_detector_state_sha256"],
        "config_sha256": static["config_sha256"],
        "sampler_audit_sha256": _sha256(destination / "sampler_audit.json"),
        "train": train_args,
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _json(result_path, "AF2LUM-SAFE result")
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
                    data=str(root / "data.yaml"), project=str(destination / ARM),
                    name=f"{ARM}_seed{seed}", exist_ok=True, seed=seed,
                    deterministic=True, plots=False, verbose=False, device=device,
                )
            model.train(
                trainer=_trainer(
                    pretrained=checkpoint,
                    af2=af2,
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
    if evaluation["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.coffee_standard_j25.af2_luminance_safe.arm_result.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "metrics": {metric: float(evaluation["metrics"][metric]) for metric in METRICS},
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
    values = [(float(left[key]), float(right[key])) for key in METRICS]
    return all(a >= b for a, b in values) and any(a > b for a, b in values)


def build_decision(
    direct_root: str | Path,
    luminance_root: str | Path,
    candidate_root: str | Path,
    output: str | Path,
) -> dict:
    rows = {
        "D0DIRECT": _json(Path(direct_root) / "val_reports/D0DIRECT_seed42_result.json", "D0"),
        "AF2LUMDIRECT": _json(
            Path(luminance_root) / "val_reports/AF2LUMDIRECT_seed42_result.json", "AF2LUM"
        ),
        ARM: _json(Path(candidate_root) / f"val_reports/{ARM}_seed42_result.json", ARM),
    }
    if rows["D0DIRECT"].get("protocol") != REFERENCE_PROTOCOL:
        raise RuntimeError("D0 reference protocol salah")
    if rows["AF2LUMDIRECT"].get("protocol") != LUMINANCE_PROTOCOL:
        raise RuntimeError("AF2LUM reference protocol salah")
    if rows[ARM].get("protocol") != PROTOCOL:
        raise RuntimeError("Candidate protocol salah")
    if any(row.get("test_images_accessed") is not False for row in rows.values()):
        raise RuntimeError("Test lock gagal")
    values = {name: row["metrics"] for name, row in rows.items()}
    references = ("D0DIRECT", "AF2LUMDIRECT")
    dominated_by = [name for name in references if _dominates(values[name], values[ARM])]
    dominates = [name for name in references if _dominates(values[ARM], values[name])]
    if dominated_by:
        decision, next_step = "DOMINATED_STOP", "STOP_WITHOUT_TEST_OR_EXTRA_SEEDS"
    elif dominates:
        decision, next_step = "PARETO_ADVANCE", "FREEZE_COMPONENT_ABLATION"
    else:
        decision, next_step = "PARETO_TRADEOFF", "REVIEW_CLASSWISE_BEFORE_ABLATION"
    payload = {
        "format": "coffee_detector.coffee_standard_j25.af2_luminance_safe.seed42_decision.v1",
        "protocol": PROTOCOL,
        "values": values,
        "deltas_vs_d0": {
            key: values[ARM][key] - values["D0DIRECT"][key] for key in METRICS
        },
        "deltas_vs_af2_luminance": {
            key: values[ARM][key] - values["AF2LUMDIRECT"][key] for key in METRICS
        },
        "dominated_by": dominated_by,
        "dominates": dominates,
        "decision": decision,
        "next": next_step,
        "claim_boundary": "final-package screen; component causality not isolated",
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
    parser.add_argument("--direct-root")
    parser.add_argument("--luminance-root")
    parser.add_argument("--decision-output")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    if args.decision_output:
        if not args.direct_root or not args.luminance_root or not args.output_root:
            parser.error("Decision mode requires direct, luminance, and candidate roots")
        build_decision(
            args.direct_root, args.luminance_root, args.output_root, args.decision_output
        )
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
