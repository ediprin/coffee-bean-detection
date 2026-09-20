"""Port the frozen J25 CWCF1 mechanism to the matched YOLOv8n detector.

The scientific question is narrow: under the exact V8N_MATCHED data/training
contract, does the same CWCF1 chromatic-wavelet classification mechanism add
value over native YOLOv8n? Locked test remains closed.
"""

from __future__ import annotations

import argparse
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
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_matched import (
    PROTOCOL as V8N_BASELINE_PROTOCOL,
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
    load_cwcf_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/V8N_CWCF1.yaml"
BASELINE_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/V8N_MATCHED.yaml"
ORIGINAL_CWCF_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/CWCF1.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolov8n-p3.yaml"
ARM = "V8N_CWCF1"
PROTOCOL = "coffee-standard-j25-yolov8n-cwcf1-seed42-v1"
TARGET_CLASS = "Biji Hitam Pecah"


def _raw_predictions(model: torch.nn.Module, probe: torch.Tensor) -> dict:
    model.train()
    with torch.no_grad():
        value = model(probe)
    if "one2many" in value:
        value = value["one2many"]
    return value


def _build_native(
    checkpoint: Path,
    seed: int,
    *,
    cfg: str | dict | Path = MODEL_YAML,
):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("YOLOv8n pretrained source bukan nc=80")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DetectionModel(str(cfg) if isinstance(cfg, Path) else cfg, ch=3, nc=NC, verbose=False)
        model.load(source)
    return model, source


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    matched_baseline_result: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen V8N_CWCF1 pertama dikunci seed 42")

    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    baseline_result = _json(matched_baseline_result, "V8N_MATCHED result")
    if baseline_result.get("protocol") != V8N_BASELINE_PROTOCOL:
        raise RuntimeError("V8N_MATCHED protocol tidak sesuai")
    if baseline_result.get("seed") != seed:
        raise RuntimeError("V8N_MATCHED seed tidak sesuai")
    if baseline_result.get("test_images_accessed") is not False:
        raise RuntimeError("Baseline melanggar test lock")

    config = _yaml(CONFIG)
    baseline_cfg = _yaml(BASELINE_CONFIG)
    original_cwcf = _yaml(ORIGINAL_CWCF_CONFIG)
    frozen = CWCFConfig.from_mapping(config["cwcf"])
    checkpoint_sha = _sha256(checkpoint)
    baseline_contract = baseline_result.get("run_contract", {})

    native, source = _build_native(checkpoint, seed, cfg=MODEL_YAML)
    source_yaml = getattr(source, "yaml", None)
    if not isinstance(source_yaml, dict):
        raise RuntimeError("YOLOv8n checkpoint tidak mengekspos source yaml")
    reference, _ = _build_native(checkpoint, seed, cfg=source_yaml)
    candidate = build_cwcf_model(
        str(MODEL_YAML), nc=NC, source=source, seed=seed, config=frozen
    )

    native_state = native.state_dict()
    reference_state = reference.state_dict()
    pinned_keys_exact = list(native_state) == list(reference_state)
    pinned_tensors_exact = pinned_keys_exact and all(
        torch.equal(native_state[key].cpu(), reference_state[key].cpu())
        for key in native_state
    )

    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)
    native_raw = _raw_predictions(native, probe)
    reference_raw = _raw_predictions(reference, probe)
    candidate_raw = _raw_predictions(candidate, probe)

    pinned_boxes_exact = torch.equal(native_raw["boxes"], reference_raw["boxes"])
    pinned_scores_exact = torch.equal(native_raw["scores"], reference_raw["scores"])
    initial_boxes_exact = torch.equal(native_raw["boxes"], candidate_raw["boxes"])
    initial_scores_exact = torch.equal(native_raw["scores"], candidate_raw["scores"])

    head = candidate.model[-1]
    if getattr(head, "end2end", True):
        raise RuntimeError("YOLOv8n CWCF port diharapkan non-end2end")
    with torch.no_grad():
        head.adapters[0].affine.weight[0, 0, 0, 0] = 0.25
    active_raw = _raw_predictions(candidate, probe)
    active_boxes_exact = torch.equal(native_raw["boxes"], active_raw["boxes"])
    active_scores_change = not torch.equal(native_raw["scores"], active_raw["scores"])

    matrix = build_j25_attribute_matrix()
    compositional_target_exact = bool(
        matrix[7, 5].item() == 1.0
        and matrix[7, 6].item() == 1.0
        and matrix[8, 5].item() == 1.0
        and matrix[8, 6].item() == 0.0
        and matrix[12, 5].item() == 0.0
        and matrix[12, 6].item() == 1.0
    )
    logits = torch.randn(3, matrix.shape[1], requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits, matrix[[7, 8, 12]])
    loss.backward()

    gates = {
        "baseline_protocol_exact": baseline_result.get("protocol") == V8N_BASELINE_PROTOCOL,
        "same_pretrained_checkpoint_as_v8n_baseline":
            checkpoint_sha == baseline_contract.get("pretrained_checkpoint_sha256"),
        "same_dataset_as_v8n_baseline":
            config.get("dataset") == baseline_cfg.get("dataset"),
        "same_training_schedule_as_v8n_baseline":
            config.get("train") == baseline_cfg.get("train"),
        "baseline_run_schedule_matches_candidate":
            baseline_contract.get("train") == config.get("train"),
        "baseline_run_dataset_matches_candidate":
            baseline_contract.get("source_archive_sha256") is not None
            and baseline_contract.get("development_contract_sha256") is not None
            and baseline_contract.get("provenance_summary_sha256") is not None,
        "same_cwcf_hyperparameters_as_original_cwcf1":
            config.get("cwcf") == original_cwcf.get("cwcf"),
        "sampler_is_none": config.get("sampler") == "none"
            and baseline_contract.get("sampler") == "none",
        "pinned_yaml_state_keys_match_checkpoint_yaml": pinned_keys_exact,
        "pinned_yaml_state_tensors_match_checkpoint_yaml": bool(pinned_tensors_exact),
        "pinned_yaml_boxes_bitwise_exact": bool(pinned_boxes_exact),
        "pinned_yaml_scores_bitwise_exact": bool(pinned_scores_exact),
        "initial_native_boxes_bitwise_exact": bool(initial_boxes_exact),
        "initial_native_scores_bitwise_exact": bool(initial_scores_exact),
        "candidate_is_non_end2end_yolov8": not bool(getattr(head, "end2end", True)),
        "active_conditioning_preserves_boxes": bool(active_boxes_exact),
        "active_conditioning_changes_scores": bool(active_scores_change),
        "black_broken_is_explicit_conjunction": compositional_target_exact,
        "attribute_loss_finite": bool(torch.isfinite(loss)),
        "attribute_gradients_finite_nonzero": bool(
            logits.grad is not None
            and torch.isfinite(logits.grad).all()
            and logits.grad.abs().sum() > 0
        ),
        "test_not_accessed": True,
    }

    payload = {
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf1.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": checkpoint_sha,
        "baseline_result_sha256": _sha256(matched_baseline_result),
        "baseline_protocol": baseline_result.get("protocol"),
        "native_state_sha256": _state_fingerprint(native),
        "reference_state_sha256": _state_fingerprint(reference),
        "config_sha256": _sha256(CONFIG),
        "baseline_config_sha256": _sha256(BASELINE_CONFIG),
        "original_cwcf_config_sha256": _sha256(ORIGINAL_CWCF_CONFIG),
        "native_parameters": _parameter_count(native),
        "candidate_parameters": _parameter_count(candidate),
        "added_parameters": _parameter_count(candidate) - _parameter_count(native),
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
            f"Static preflight gagal: {[key for key, value in gates.items() if not value]}"
        )
    return payload


def _trainer(*, pretrained: Path, seed: int, config: CWCFConfig):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class J25V8NCWCFTrainer(DetectionTrainer):
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
            model = build_cwcf_model(
                str(MODEL_YAML),
                nc=self.data["nc"],
                source=source,
                seed=seed,
                config=config,
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

    return J25V8NCWCFTrainer


def run_arm(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
    pretrained_checkpoint: str | Path,
    matched_baseline_result: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != SEED:
        raise RuntimeError("V8N_CWCF1 screening dikunci seed 42")
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training")

    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("V8N_CWCF1 hanya untuk J25 train-siblings v2")

    baseline = _json(matched_baseline_result, "V8N_MATCHED result")
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static = run_static_preflight(
        checkpoint,
        matched_baseline_result,
        destination / "static_preflight.json",
        seed=seed,
    )

    config = _yaml(CONFIG)
    frozen = CWCFConfig.from_mapping(config["cwcf"])
    train_args = dict(config["train"])
    baseline_contract = baseline["run_contract"]

    dataset_gates = {
        "same_source_archive_as_baseline":
            dataset["source_archive_sha256"] == baseline_contract.get("source_archive_sha256"),
        "same_development_contract_as_baseline":
            dataset["development_contract_sha256"]
            == baseline_contract.get("development_contract_sha256"),
        "same_provenance_summary_as_baseline":
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
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf1.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "matched_baseline_protocol": V8N_BASELINE_PROTOCOL,
        "matched_baseline_result_sha256": _sha256(matched_baseline_result),
        "source_archive_sha256": dataset["source_archive_sha256"],
        "development_contract_sha256": dataset["development_contract_sha256"],
        "provenance_summary_sha256": dataset["provenance_summary_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "config_sha256": static["config_sha256"],
        "baseline_config_sha256": static["baseline_config_sha256"],
        "original_cwcf_config_sha256": static["original_cwcf_config_sha256"],
        "train": train_args,
        "cwcf": frozen.to_dict(),
        "sampler": "none",
        "test_images_accessed": False,
    }

    if result_path.is_file():
        old = _json(result_path, "V8N_CWCF1 result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing V8N_CWCF1 result berbeda kontrak")
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
                trainer=_trainer(pretrained=checkpoint, seed=seed, config=frozen),
                **args,
            )
        training_executed = True

    if not _run_complete(run_dir, int(train_args["epochs"])) or not best.is_file():
        raise RuntimeError("V8N_CWCF1 belum selesai secara valid")

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
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf1.arm_result.v1",
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
    matched_baseline_result: str | Path,
    candidate_result: str | Path,
    output: str | Path,
) -> dict:
    baseline = _json(matched_baseline_result, "V8N_MATCHED")
    candidate = _json(candidate_result, "V8N_CWCF1")

    if baseline.get("protocol") != V8N_BASELINE_PROTOCOL:
        raise RuntimeError("Baseline protocol salah")
    if candidate.get("protocol") != PROTOCOL:
        raise RuntimeError("Candidate protocol salah")
    for label, payload in (("V8N_MATCHED", baseline), ("V8N_CWCF1", candidate)):
        if payload.get("seed") != SEED:
            raise RuntimeError(f"{label} bukan seed 42")
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"{label} melanggar test lock")

    values = {
        "V8N_MATCHED": {metric: float(baseline["metrics"][metric]) for metric in METRICS},
        "V8N_CWCF1": {metric: float(candidate["metrics"][metric]) for metric in METRICS},
    }
    delta = {
        metric: values["V8N_CWCF1"][metric] - values["V8N_MATCHED"][metric]
        for metric in METRICS
    }
    target_values = {
        "V8N_MATCHED": float(baseline["target_class_map50_95"]),
        "V8N_CWCF1": float(candidate["target_class_map50_95"]),
    }
    target_delta = target_values["V8N_CWCF1"] - target_values["V8N_MATCHED"]

    primary_pass = delta["macro_map50_95"] > 0.0
    tail_pass = delta["bottom3_class_map50_95"] > 0.0
    if primary_pass and tail_pass:
        decision = "PASS_CONFIRMATION_SCREEN"
        next_step = "PAIRED_MULTI_SEED_CONFIRMATION"
    elif any(value > 0.0 for value in delta.values()) and any(
        value < 0.0 for value in delta.values()
    ):
        decision = "PARETO_TRADEOFF_SCREEN"
        next_step = "REVIEW_BEFORE_CONFIRMATION"
    else:
        decision = "NO_ADVANTAGE_SCREEN"
        next_step = "RETAIN_V8N_MATCHED"

    payload = {
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf1.seed42_decision.v1",
        "protocol": PROTOCOL,
        "seed": SEED,
        "values": values,
        "v8n_cwcf1_minus_v8n": delta,
        "target_class": TARGET_CLASS,
        "target_values": target_values,
        "target_delta": target_delta,
        "criteria": {
            "macro_delta_positive": primary_pass,
            "bottom3_delta_positive": tail_pass,
            "worst_class_is_descriptive_not_a_promotion_gate": True,
            "target_class_is_descriptive_not_a_promotion_gate": True,
            "all_25_validation_classes_required": True,
            "test_must_remain_closed": True,
        },
        "decision": decision,
        "next_step": next_step,
        "claim_boundary": (
            "single-seed matched portability screen of the frozen CWCF1 mechanism "
            "on YOLOv8n; no locked-test or confirmation claim"
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
    parser.add_argument("--matched-baseline-result")
    parser.add_argument("--output-root")
    parser.add_argument("--candidate-result")
    parser.add_argument("--decision-output")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()

    if args.decision_output:
        if args.matched_baseline_result is None or args.candidate_result is None:
            parser.error("Decision mode requires baseline and candidate results")
        payload = build_decision(
            args.matched_baseline_result,
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
        args.matched_baseline_result,
        args.output_root,
    )
    if any(value is None for value in required):
        parser.error("Training mode requires data, contracts, pretrained, baseline, and output")
    run_arm(
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.pretrained_checkpoint,
        args.matched_baseline_result,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
