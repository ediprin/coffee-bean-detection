"""Learnable per-scale residual gates for J25 YOLOv8n CWCF.

SG1 keeps the complete V8N_CWCF1 mechanism and adds exactly three scalar
parameters, one for each P3/P4/P5 CWCF residual. Each gate is

    g_l = 2 * sigmoid(alpha_l)

with alpha_l initialized to zero, so g_l starts exactly at one.
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
    load_cwcf_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/V8N_CWCF_SG1.yaml"
BASELINE_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/V8N_CWCF1.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolov8n-p3.yaml"
ARM = "V8N_CWCF_SG1"
PROTOCOL = "coffee-standard-j25-yolov8n-cwcf-sg1-seed42-v1"
TARGET_CLASS = "Biji Hitam Pecah"
SCALE_NAMES = ("P3", "P4", "P5")


def _raw_predictions(model: torch.nn.Module, probe: torch.Tensor) -> dict:
    model.train()
    with torch.no_grad():
        value = model(probe)
    if "one2many" in value:
        value = value["one2many"]
    return value


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


def _non_gate_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if ".scale_gate_logits." not in key
    }


def _state_mapping_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return list(left) == list(right) and all(
        torch.equal(left[key].cpu(), right[key].cpu()) for key in left
    )


def _gate_snapshot_from_head(head: torch.nn.Module) -> dict:
    parameters = getattr(head, "scale_gate_logits", None)
    if parameters is None:
        raise RuntimeError("Checkpoint/model SG1 tidak memiliki scale gates")
    logits = [float(parameter.detach().cpu()) for parameter in parameters]
    gains = [
        float((2.0 * torch.sigmoid(parameter.detach())).cpu())
        for parameter in parameters
    ]
    if len(logits) != len(SCALE_NAMES):
        raise RuntimeError("Jumlah scale gate bukan tiga")
    return {
        "logits": dict(zip(SCALE_NAMES, logits)),
        "gains": dict(zip(SCALE_NAMES, gains)),
    }


def _checkpoint_gate_snapshot(checkpoint: str | Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(Path(checkpoint).expanduser().resolve())).model
    return _gate_snapshot_from_head(model.model[-1])


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    cwcf1_result: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen V8N_CWCF_SG1 pertama dikunci seed 42")

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
    candidate_mapping = dict(config["cwcf"])
    baseline_mapping = dict(baseline_cfg["cwcf"])
    candidate_frozen = CWCFConfig.from_mapping(candidate_mapping)
    baseline_frozen = CWCFConfig.from_mapping(baseline_mapping)
    baseline_contract = baseline_result.get("run_contract", {})
    checkpoint_sha = _sha256(checkpoint)

    candidate_without_gate = candidate_frozen.to_dict()
    baseline_normalized = baseline_frozen.to_dict()
    candidate_gate_enabled = candidate_without_gate.pop("learnable_scale_gates")
    baseline_gate_enabled = baseline_normalized.pop("learnable_scale_gates")

    native, source = _build_native(checkpoint, seed)
    candidate = build_cwcf_model(
        str(MODEL_YAML),
        nc=NC,
        source=source,
        seed=seed,
        config=candidate_frozen,
    )
    cwcf1_reference = build_cwcf_model(
        str(MODEL_YAML),
        nc=NC,
        source=source,
        seed=seed,
        config=baseline_frozen,
    )

    candidate_head = candidate.model[-1]
    baseline_head = cwcf1_reference.model[-1]
    non_gate_state_exact = _state_mapping_equal(
        _non_gate_state(candidate), _non_gate_state(cwcf1_reference)
    )
    added_parameters = _parameter_count(candidate) - _parameter_count(cwcf1_reference)
    gate_initial = _gate_snapshot_from_head(candidate_head)

    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)
    native_raw = _raw_predictions(native, probe)
    candidate_raw = _raw_predictions(candidate, probe)
    baseline_raw = _raw_predictions(cwcf1_reference, probe)

    initial_candidate_boxes_exact = torch.equal(
        candidate_raw["boxes"], baseline_raw["boxes"]
    )
    initial_candidate_scores_exact = torch.equal(
        candidate_raw["scores"], baseline_raw["scores"]
    )
    initial_native_boxes_exact = torch.equal(
        native_raw["boxes"], candidate_raw["boxes"]
    )
    initial_native_scores_exact = torch.equal(
        native_raw["scores"], candidate_raw["scores"]
    )

    # Activate the same nonzero residual at all scales. With alpha=0 -> g=1,
    # SG1 must remain functionally identical to CWCF1.
    with torch.no_grad():
        for index, (candidate_adapter, baseline_adapter) in enumerate(
            zip(candidate_head.adapters, baseline_head.adapters)
        ):
            value = 0.10 * (index + 1)
            candidate_adapter.affine.bias[0] = value
            baseline_adapter.affine.bias[0] = value
    candidate_gate1_raw = _raw_predictions(candidate, probe)
    baseline_active_raw = _raw_predictions(cwcf1_reference, probe)
    gate1_boxes_exact = torch.equal(
        candidate_gate1_raw["boxes"], baseline_active_raw["boxes"]
    )
    gate1_scores_exact = torch.equal(
        candidate_gate1_raw["scores"], baseline_active_raw["scores"]
    )

    # Change only P5 gate. Scores should change while boxes stay untouched.
    with torch.no_grad():
        candidate_head.scale_gate_logits[2].fill_(-2.0)
    gated_raw = _raw_predictions(candidate, probe)
    changed_gate = _gate_snapshot_from_head(candidate_head)
    gate_changes_scores = not torch.equal(
        candidate_gate1_raw["scores"], gated_raw["scores"]
    )
    gate_preserves_boxes = torch.equal(
        candidate_gate1_raw["boxes"], gated_raw["boxes"]
    )

    # With a nonzero residual, scale-gate gradients must be reachable.
    candidate.zero_grad(set_to_none=True)
    value = candidate(probe)
    if "one2many" in value:
        value = value["one2many"]
    value["scores"].mean().backward()
    gate_gradients = [
        parameter.grad
        for parameter in candidate_head.scale_gate_logits
    ]
    gate_gradients_finite_nonzero = bool(
        all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gate_gradients
        )
        and sum(float(gradient.abs().sum()) for gradient in gate_gradients) > 0.0
    )

    matrix = build_j25_attribute_matrix()
    logits = torch.randn(3, matrix.shape[1], requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(logits, matrix[[7, 8, 12]])
    loss.backward()

    gates = {
        "baseline_protocol_exact": baseline_result.get("protocol") == CWCF1_PROTOCOL,
        "same_pretrained_checkpoint_as_cwcf1":
            checkpoint_sha == baseline_contract.get("pretrained_checkpoint_sha256"),
        "same_dataset_as_cwcf1":
            config.get("dataset") == baseline_cfg.get("dataset"),
        "same_training_schedule_as_cwcf1":
            config.get("train") == baseline_cfg.get("train")
            and baseline_contract.get("train") == config.get("train"),
        "same_sampler_as_cwcf1":
            config.get("sampler") == baseline_cfg.get("sampler") == "none",
        "only_scale_gate_flag_changes":
            candidate_without_gate == baseline_normalized
            and candidate_gate_enabled is True
            and baseline_gate_enabled is False,
        "non_gate_state_bitwise_equals_cwcf1": bool(non_gate_state_exact),
        "exactly_three_parameters_added": added_parameters == 3,
        "three_scale_gates_present":
            len(candidate_head.scale_gate_logits) == 3,
        "gate_logits_initialize_zero":
            all(value == 0.0 for value in gate_initial["logits"].values()),
        "gate_gains_initialize_one":
            all(value == 1.0 for value in gate_initial["gains"].values()),
        "initial_candidate_boxes_equal_cwcf1": bool(initial_candidate_boxes_exact),
        "initial_candidate_scores_equal_cwcf1": bool(initial_candidate_scores_exact),
        "initial_candidate_boxes_equal_native": bool(initial_native_boxes_exact),
        "initial_candidate_scores_equal_native": bool(initial_native_scores_exact),
        "gate_one_active_boxes_equal_cwcf1": bool(gate1_boxes_exact),
        "gate_one_active_scores_equal_cwcf1": bool(gate1_scores_exact),
        "p5_gate_perturbation_changes_scores": bool(gate_changes_scores),
        "p5_gate_perturbation_preserves_boxes": bool(gate_preserves_boxes),
        "p5_gate_gain_moves_below_one":
            changed_gate["gains"]["P5"] < 1.0,
        "scale_gate_gradients_finite_nonzero":
            gate_gradients_finite_nonzero,
        "attribute_loss_finite": bool(torch.isfinite(loss)),
        "attribute_gradients_finite_nonzero": bool(
            logits.grad is not None
            and torch.isfinite(logits.grad).all()
            and logits.grad.abs().sum() > 0
        ),
        "test_not_accessed": True,
    }

    payload = {
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf_sg1.static.v1",
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
        "added_parameters_vs_cwcf1": added_parameters,
        "candidate_initial_state_sha256": _state_fingerprint(candidate),
        "initial_scale_gates": gate_initial,
        "probe_scale_gates": changed_gate,
        "cwcf": candidate_frozen.to_dict(),
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


def _trainer(*, pretrained: Path, seed: int, config: CWCFConfig):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class J25V8NCWCFScaleGateTrainer(DetectionTrainer):
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

    return J25V8NCWCFScaleGateTrainer


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
        raise RuntimeError("V8N_CWCF_SG1 screening dikunci seed 42")
    if not authorize_training:
        raise RuntimeError("Training memerlukan --authorize-training")

    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("V8N_CWCF_SG1 hanya untuk J25 train-siblings v2")

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
    frozen = CWCFConfig.from_mapping(config["cwcf"])
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
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf_sg1.arm_contract.v1",
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
            "add three bounded learnable scale gates g=2*sigmoid(alpha), alpha_init=0",
        "test_images_accessed": False,
    }

    if result_path.is_file():
        old = _json(result_path, "V8N_CWCF_SG1 result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing V8N_CWCF_SG1 result berbeda kontrak")
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
        raise RuntimeError("V8N_CWCF_SG1 belum selesai secara valid")

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
    learned_scale_gates = _checkpoint_gate_snapshot(best)

    result = {
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf_sg1.arm_result.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "metrics": {metric: float(metrics[metric]) for metric in METRICS},
        "map50_95_by_class": metrics["map50_95_by_class"],
        "target_class": TARGET_CLASS,
        "target_class_map50_95": float(metrics["map50_95_by_class"][TARGET_CLASS]),
        "learned_scale_gates": learned_scale_gates,
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
    candidate = _json(candidate_result, "V8N_CWCF_SG1")

    if baseline.get("protocol") != CWCF1_PROTOCOL:
        raise RuntimeError("CWCF1 baseline protocol salah")
    if candidate.get("protocol") != PROTOCOL:
        raise RuntimeError("SG1 candidate protocol salah")
    for label, payload in (
        ("V8N_CWCF1", baseline),
        ("V8N_CWCF_SG1", candidate),
    ):
        if payload.get("seed") != SEED:
            raise RuntimeError(f"{label} bukan seed 42")
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"{label} melanggar test lock")

    values = {
        "V8N_CWCF1": {metric: float(baseline["metrics"][metric]) for metric in METRICS},
        "V8N_CWCF_SG1": {
            metric: float(candidate["metrics"][metric]) for metric in METRICS
        },
    }
    delta = {
        metric: values["V8N_CWCF_SG1"][metric] - values["V8N_CWCF1"][metric]
        for metric in METRICS
    }
    target_values = {
        "V8N_CWCF1": float(baseline["target_class_map50_95"]),
        "V8N_CWCF_SG1": float(candidate["target_class_map50_95"]),
    }
    target_delta = target_values["V8N_CWCF_SG1"] - target_values["V8N_CWCF1"]

    macro_pass = delta["macro_map50_95"] > 0.0
    tail_pass = delta["bottom3_class_map50_95"] > 0.0
    if macro_pass and tail_pass:
        decision = "PASS_SG1_SCREEN"
        next_step = "RETAIN_LEARNED_SCALE_GATES"
    elif any(value > 0.0 for value in delta.values()) and any(
        value < 0.0 for value in delta.values()
    ):
        decision = "PARETO_TRADEOFF_SG1"
        next_step = "REVIEW_LEARNED_SCALE_GATES"
    else:
        decision = "FAIL_SG1_SCREEN"
        next_step = "RETAIN_V8N_CWCF1"

    payload = {
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf_sg1.seed42_decision.v1",
        "protocol": PROTOCOL,
        "seed": SEED,
        "values": values,
        "v8n_cwcf_sg1_minus_cwcf1": delta,
        "target_class": TARGET_CLASS,
        "target_values": target_values,
        "target_delta": target_delta,
        "learned_scale_gates": candidate.get("learned_scale_gates"),
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
            "single-seed matched scale-gating screen; CWCF cue, loss, detector, "
            "data, initialization, and training schedule remain matched"
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
            parser.error("Decision mode requires CWCF1 and SG1 results")
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
