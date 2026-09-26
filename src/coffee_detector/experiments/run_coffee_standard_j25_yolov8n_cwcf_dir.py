"""Directional-Haar representation screen for the J25 YOLOv8n CWCF path.

The completed V8N_CWCF1 arm remains the matched baseline. DIR1 changes only
the wavelet representation: two rotation-neutral detail-energy channels become
six signed LH/HL/HH channels across the same two Haar levels.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    METRICS,
    NC,
    SEED,
    _epochs,
    _json,
    _yaml,
    validate_j25_development,
)
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf import (
    PROTOCOL as CWCF1_PROTOCOL,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import (
    _parameter_count,
    _sha256,
    _source_class_count,
    _state_fingerprint,
)
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)
from coffee_detector.j25_cwcf import (
    CWCFConfig,
    build_cwcf_model,
    build_j25_attribute_matrix,
    chromatic_wavelet_cue,
    haar_bands,
    haar_decompose,
    load_cwcf_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/V8N_CWCF_DIR1.yaml"
BASELINE_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/V8N_CWCF1.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolov8n-p3.yaml"
ARM = "V8N_CWCF_DIR1"
PROTOCOL = "coffee-standard-j25-yolov8n-cwcf-dir1-seed42-v1"
TARGET_CLASS = "Biji Hitam Pecah"
REPRESENTATION_KEYS = {"cue_channels", "wavelet_detail_mode"}


def _raw_predictions(model: torch.nn.Module, probe: torch.Tensor) -> dict:
    model.train()
    with torch.no_grad():
        value = model(probe)
    if "one2many" in value:
        value = value["one2many"]
    return value


def _normalized_cwcf(mapping: dict) -> dict:
    return CWCFConfig.from_mapping(mapping).to_dict()


def _cwcf_without_representation(mapping: dict) -> dict:
    normalized = _normalized_cwcf(mapping)
    return {
        key: value
        for key, value in normalized.items()
        if key not in REPRESENTATION_KEYS
    }


def _build_native(checkpoint: Path, seed: int):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("YOLOv8n pretrained source bukan nc=80")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DetectionModel(str(MODEL_YAML), ch=3, nc=NC, verbose=False)
        model.load(source)
    return model, source


def _build_directional_candidate(
    source: torch.nn.Module,
    seed: int,
    candidate_config: CWCFConfig,
    baseline_config: CWCFConfig,
    *,
    verbose: bool = False,
):
    """Build DIR1 while preserving CWCF1 auxiliary-head initialization.

    Changing adapter input width changes how many random numbers Conv2d
    construction consumes. The adapters are zeroed immediately, but without
    this copy the later attribute-head initialization would differ from CWCF1.
    Reconstructing the frozen CWCF1 seed-42 initialization and copying only its
    auxiliary attribute heads keeps the comparison isolated to cue
    representation/cue-adapter width.
    """

    candidate = build_cwcf_model(
        str(MODEL_YAML),
        nc=NC,
        source=source,
        seed=seed,
        config=candidate_config,
        verbose=verbose,
    )
    baseline_reference = build_cwcf_model(
        str(MODEL_YAML),
        nc=NC,
        source=source,
        seed=seed,
        config=baseline_config,
        verbose=False,
    )
    candidate.model[-1].attribute_heads.load_state_dict(
        copy.deepcopy(baseline_reference.model[-1].attribute_heads.state_dict()),
        strict=True,
    )
    return candidate, baseline_reference


def _module_state_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    lhs, rhs = left.state_dict(), right.state_dict()
    return list(lhs) == list(rhs) and all(
        torch.equal(lhs[key].cpu(), rhs[key].cpu()) for key in lhs
    )


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    cwcf1_result: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen V8N_CWCF_DIR1 pertama dikunci seed 42")

    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    baseline_result = _json(cwcf1_result, "V8N_CWCF1 result")
    if baseline_result.get("protocol") != CWCF1_PROTOCOL:
        raise RuntimeError("V8N_CWCF1 baseline protocol tidak sesuai")
    if baseline_result.get("seed") != seed:
        raise RuntimeError("V8N_CWCF1 baseline seed tidak sesuai")
    if baseline_result.get("test_images_accessed") is not False:
        raise RuntimeError("Baseline melanggar test lock")

    config = _yaml(CONFIG)
    baseline_cfg = _yaml(BASELINE_CONFIG)
    frozen = CWCFConfig.from_mapping(config["cwcf"])
    baseline_frozen = CWCFConfig.from_mapping(baseline_cfg["cwcf"])
    baseline_contract = baseline_result.get("run_contract", {})
    checkpoint_sha = _sha256(checkpoint)

    native, source = _build_native(checkpoint, seed)
    candidate, cwcf1_reference = _build_directional_candidate(
        source, seed, frozen, baseline_frozen
    )
    candidate_head = candidate.model[-1]
    baseline_head = cwcf1_reference.model[-1]

    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)
    native_raw = _raw_predictions(native, probe)
    candidate_raw = _raw_predictions(candidate, probe)
    baseline_raw = _raw_predictions(cwcf1_reference, probe)

    energy_cue = chromatic_wavelet_cue(
        probe, clip=baseline_frozen.cue_clip, detail_mode="energy"
    )
    directional_cue = chromatic_wavelet_cue(
        probe, clip=frozen.cue_clip, detail_mode="directional"
    )

    luminance = (
        0.2126 * probe[:, 0:1]
        + 0.7152 * probe[:, 1:2]
        + 0.0722 * probe[:, 2:3]
    )
    ll1, lh1, hl1, hh1 = haar_bands(luminance)
    ll1_ref, detail1 = haar_decompose(luminance)
    _, lh2, hl2, hh2 = haar_bands(ll1)
    _, detail2 = haar_decompose(ll1)
    reconstructed1 = torch.sqrt(
        torch.cat((lh1, hl1, hh1), dim=1).square().mean(dim=1, keepdim=True)
        + 1e-8
    )
    reconstructed2 = torch.sqrt(
        torch.cat((lh2, hl2, hh2), dim=1).square().mean(dim=1, keepdim=True)
        + 1e-8
    )

    matrix = build_j25_attribute_matrix()
    logits = torch.randn(3, matrix.shape[1], requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits, matrix[[7, 8, 12]])
    loss.backward()

    initial_boxes_exact = torch.equal(native_raw["boxes"], candidate_raw["boxes"])
    initial_scores_exact = torch.equal(native_raw["scores"], candidate_raw["scores"])
    baseline_boxes_exact = torch.equal(native_raw["boxes"], baseline_raw["boxes"])
    baseline_scores_exact = torch.equal(native_raw["scores"], baseline_raw["scores"])

    with torch.no_grad():
        candidate_head.adapters[0].affine.weight[0, 2, 0, 0] = 0.25
    active_raw = _raw_predictions(candidate, probe)
    active_boxes_exact = torch.equal(native_raw["boxes"], active_raw["boxes"])
    active_scores_change = not torch.equal(native_raw["scores"], active_raw["scores"])

    adapter_zero = all(
        torch.count_nonzero(adapter.affine.weight).item() == 0
        and torch.count_nonzero(adapter.affine.bias).item() == 0
        for adapter in _build_directional_candidate(
            source, seed, frozen, baseline_frozen
        )[0].model[-1].adapters
    )

    gates = {
        "baseline_protocol_exact": baseline_result.get("protocol") == CWCF1_PROTOCOL,
        "same_pretrained_checkpoint_as_cwcf1":
            checkpoint_sha == baseline_contract.get("pretrained_checkpoint_sha256"),
        "same_dataset_as_cwcf1": config.get("dataset") == baseline_cfg.get("dataset"),
        "same_training_schedule_as_cwcf1":
            config.get("train") == baseline_cfg.get("train")
            and baseline_contract.get("train") == config.get("train"),
        "same_sampler_as_cwcf1":
            config.get("sampler") == baseline_cfg.get("sampler") == "none",
        "only_cue_representation_config_changes":
            _cwcf_without_representation(config["cwcf"])
            == _cwcf_without_representation(baseline_cfg["cwcf"]),
        "baseline_energy_mode_four_channels":
            baseline_frozen.wavelet_detail_mode == "energy"
            and baseline_frozen.cue_channels == 4,
        "candidate_directional_mode_eight_channels":
            frozen.wavelet_detail_mode == "directional"
            and frozen.cue_channels == 8,
        "energy_cue_shape_exact": tuple(energy_cue.shape) == (1, 4, 64, 64),
        "directional_cue_shape_exact":
            tuple(directional_cue.shape) == (1, 8, 64, 64),
        "chroma_channels_unchanged":
            torch.equal(energy_cue[:, :2], directional_cue[:, :2]),
        "haar_ll_unchanged": torch.equal(ll1, ll1_ref),
        "haar_energy_reconstructs_level1":
            torch.allclose(detail1, reconstructed1, atol=0.0, rtol=0.0),
        "haar_energy_reconstructs_level2":
            torch.allclose(detail2, reconstructed2, atol=0.0, rtol=0.0),
        "native_base_head_state_matches_cwcf1":
            _module_state_equal(candidate_head.base_head, baseline_head.base_head),
        "attribute_head_initialization_matches_cwcf1":
            _module_state_equal(
                candidate_head.attribute_heads, baseline_head.attribute_heads
            ),
        "directional_adapters_zero_initialized": bool(adapter_zero),
        "cwcf1_reference_native_boxes_bitwise_exact": bool(baseline_boxes_exact),
        "cwcf1_reference_native_scores_bitwise_exact": bool(baseline_scores_exact),
        "candidate_initial_boxes_bitwise_exact": bool(initial_boxes_exact),
        "candidate_initial_scores_bitwise_exact": bool(initial_scores_exact),
        "active_directional_conditioning_preserves_boxes": bool(active_boxes_exact),
        "active_directional_conditioning_changes_scores": bool(active_scores_change),
        "attribute_loss_finite": bool(torch.isfinite(loss)),
        "attribute_gradients_finite_nonzero": bool(
            logits.grad is not None
            and torch.isfinite(logits.grad).all()
            and logits.grad.abs().sum() > 0
        ),
        "test_not_accessed": True,
    }

    payload = {
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf_dir1.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": checkpoint_sha,
        "baseline_result_sha256": _sha256(cwcf1_result),
        "baseline_protocol": baseline_result.get("protocol"),
        "config_sha256": _sha256(CONFIG),
        "baseline_config_sha256": _sha256(BASELINE_CONFIG),
        "native_parameters": _parameter_count(native),
        "cwcf1_reference_parameters": _parameter_count(cwcf1_reference),
        "candidate_parameters": _parameter_count(candidate),
        "added_parameters_vs_cwcf1":
            _parameter_count(candidate) - _parameter_count(cwcf1_reference),
        "candidate_state_sha256": _state_fingerprint(candidate),
        "cwcf": frozen.to_dict(),
        "gates": gates,
        "training_authorized": all(gates.values()),
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["decision"] != "PASS":
        raise RuntimeError(
            f"Static preflight gagal: {[k for k, v in gates.items() if not v]}"
        )
    return payload


def _trainer(
    *,
    pretrained: Path,
    seed: int,
    config: CWCFConfig,
    baseline_config: CWCFConfig,
):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class J25V8NCWCFDirectionalTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if bool(getattr(self.args, "resume", False)):
                model = build_cwcf_model(
                    str(MODEL_YAML),
                    nc=self.data["nc"],
                    source=None,
                    seed=seed,
                    config=config,
                    verbose=verbose,
                )
                if weights:
                    load_cwcf_weights(model, weights)
                return self.set_model_names_for_load(model)

            source = YOLO(str(pretrained)).model
            model, _ = _build_directional_candidate(
                source,
                seed,
                config,
                baseline_config,
                verbose=verbose,
            )
            return self.set_model_names_for_load(model)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    return J25V8NCWCFDirectionalTrainer


def run_arm(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
    pretrained_checkpoint: str | Path,
    cwcf1_result: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != SEED:
        raise RuntimeError("V8N_CWCF_DIR1 screening dikunci seed 42")
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training")

    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("V8N_CWCF_DIR1 hanya untuk J25 train-siblings v2")

    baseline = _json(cwcf1_result, "V8N_CWCF1 result")
    baseline_contract = baseline["run_contract"]
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static = run_static_preflight(
        checkpoint,
        cwcf1_result,
        destination / "static_preflight.json",
        seed=seed,
    )

    config = _yaml(CONFIG)
    baseline_cfg = _yaml(BASELINE_CONFIG)
    frozen = CWCFConfig.from_mapping(config["cwcf"])
    baseline_frozen = CWCFConfig.from_mapping(baseline_cfg["cwcf"])
    train_args = dict(config["train"])

    dataset_gates = {
        "same_source_archive_as_cwcf1":
            dataset["source_archive_sha256"]
            == baseline_contract.get("source_archive_sha256"),
        "same_development_contract_as_cwcf1":
            dataset["development_contract_sha256"]
            == baseline_contract.get("development_contract_sha256"),
        "same_provenance_summary_as_cwcf1":
            dataset["provenance_summary_sha256"]
            == baseline_contract.get("provenance_summary_sha256"),
    }
    if not all(dataset_gates.values()):
        raise RuntimeError(
            f"Dataset contract tidak matched: {[k for k,v in dataset_gates.items() if not v]}"
        )

    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = destination / "val_reports" / f"{ARM}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf_dir1.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "matched_baseline_protocol": CWCF1_PROTOCOL,
        "matched_baseline_result_sha256": _sha256(cwcf1_result),
        "source_archive_sha256": dataset["source_archive_sha256"],
        "development_contract_sha256": dataset["development_contract_sha256"],
        "provenance_summary_sha256": dataset["provenance_summary_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "config_sha256": static["config_sha256"],
        "baseline_config_sha256": static["baseline_config_sha256"],
        "train": train_args,
        "cwcf": frozen.to_dict(),
        "sampler": "none",
        "comparison_change":
            "energy(D1,D2)->signed directional(LH1,HL1,HH1,LH2,HL2,HH2)",
        "test_images_accessed": False,
    }

    if result_path.is_file():
        old = _json(result_path, "V8N_CWCF_DIR1 result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing V8N_CWCF_DIR1 result berbeda kontrak")
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
                args = {"resume": True, "device": device}
            else:
                if last.is_file():
                    raise RuntimeError("last.pt tidak resumable")
                model = YOLO(str(MODEL_YAML))
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
            model.train(
                trainer=_trainer(
                    pretrained=checkpoint,
                    seed=seed,
                    config=frozen,
                    baseline_config=baseline_frozen,
                ),
                **args,
            )
        training_executed = True

    if not _run_complete(run_dir, int(train_args["epochs"])) or not best.is_file():
        raise RuntimeError("V8N_CWCF_DIR1 belum selesai secara valid")

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
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf_dir1.arm_result.v1",
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
    baseline = _json(cwcf1_result, "V8N_CWCF1")
    candidate = _json(candidate_result, "V8N_CWCF_DIR1")

    if baseline.get("protocol") != CWCF1_PROTOCOL:
        raise RuntimeError("CWCF1 baseline protocol salah")
    if candidate.get("protocol") != PROTOCOL:
        raise RuntimeError("DIR1 candidate protocol salah")
    for label, payload in (
        ("V8N_CWCF1", baseline),
        ("V8N_CWCF_DIR1", candidate),
    ):
        if payload.get("seed") != SEED:
            raise RuntimeError(f"{label} bukan seed 42")
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"{label} melanggar test lock")

    values = {
        "V8N_CWCF1": {metric: float(baseline["metrics"][metric]) for metric in METRICS},
        "V8N_CWCF_DIR1": {
            metric: float(candidate["metrics"][metric]) for metric in METRICS
        },
    }
    delta = {
        metric: values["V8N_CWCF_DIR1"][metric] - values["V8N_CWCF1"][metric]
        for metric in METRICS
    }
    target_values = {
        "V8N_CWCF1": float(baseline["target_class_map50_95"]),
        "V8N_CWCF_DIR1": float(candidate["target_class_map50_95"]),
    }
    target_delta = target_values["V8N_CWCF_DIR1"] - target_values["V8N_CWCF1"]

    macro_pass = delta["macro_map50_95"] > 0.0
    tail_pass = delta["bottom3_class_map50_95"] > 0.0
    if macro_pass and tail_pass:
        decision = "PASS_DIR1_SCREEN"
        next_step = "RETAIN_DIRECTIONAL_REPRESENTATION"
    elif any(value > 0.0 for value in delta.values()) and any(
        value < 0.0 for value in delta.values()
    ):
        decision = "PARETO_TRADEOFF_DIR1"
        next_step = "REVIEW_DIRECTIONAL_REPRESENTATION"
    else:
        decision = "FAIL_DIR1_SCREEN"
        next_step = "RETAIN_V8N_CWCF1"

    payload = {
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf_dir1.seed42_decision.v1",
        "protocol": PROTOCOL,
        "seed": SEED,
        "values": values,
        "v8n_cwcf_dir1_minus_cwcf1": delta,
        "target_class": TARGET_CLASS,
        "target_values": target_values,
        "target_delta": target_delta,
        "criteria": {
            "macro_delta_positive": macro_pass,
            "bottom3_delta_positive": tail_pass,
            "worst_class_is_descriptive_not_a_promotion_gate": True,
            "target_class_is_descriptive_not_a_promotion_gate": True,
            "all_25_validation_classes_required": True,
            "test_must_remain_closed": True,
        },
        "decision": decision,
        "next_step": next_step,
        "claim_boundary": (
            "single-seed matched representation screen; only the CWCF wavelet "
            "detail representation changes from energy to directional bands"
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
    parser.add_argument("--cwcf1-result")
    parser.add_argument("--output-root")
    parser.add_argument("--candidate-result")
    parser.add_argument("--decision-output")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()

    if args.decision_output:
        if args.cwcf1_result is None or args.candidate_result is None:
            parser.error("Decision mode requires CWCF1 and DIR1 results")
        payload = build_decision(
            args.cwcf1_result,
            args.candidate_result,
            args.decision_output,
        )
        print(json.dumps(payload, indent=2), flush=True)
        return

    required = (
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.pretrained_checkpoint,
        args.cwcf1_result,
        args.output_root,
    )
    if any(value is None for value in required):
        parser.error("Training mode requires data, contracts, pretrained, CWCF1, and output")
    run_arm(
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.pretrained_checkpoint,
        args.cwcf1_result,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
