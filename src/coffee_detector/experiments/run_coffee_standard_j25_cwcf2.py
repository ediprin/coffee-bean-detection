"""Fresh J25 CWCF2 explicit-composition screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    METRICS, MODEL_YAML, NC, SEED, _epochs, _json, _yaml, validate_j25_development,
)
from coffee_detector.experiments.run_coffee_standard_j25_cwcf import (
    DIRECT_PROTOCOL,
    PROTOCOL as CWCF1_PROTOCOL,
    SAFEAUG_PROTOCOL,
    _build_native,
    _raw_predictions,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import (
    OFFICIAL_YOLO26N_SHA256, _parameter_count, _require_official_pretrained,
    _sha256,
)
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state, _epoch_sequence, _exclusive_training_lock, _run_complete,
)
from coffee_detector.j25_cwcf import (
    CWCFConfig,
    attribute_compatibility_logits,
    balanced_attribute_bce,
    build_cwcf_model,
    build_j25_attribute_matrix,
    load_cwcf_weights,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/CWCF2.yaml"
CWCF1_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/CWCF1.yaml"
SAFEAUG_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/SAFEAUG0.yaml"
ARM = "CWCF2"
PROTOCOL = "coffee-standard-j25-cwcf2-seed42-v1"


def run_static_preflight(
    pretrained_checkpoint: str | Path, output: str | Path, *, seed: int = SEED
) -> dict:
    if seed != SEED:
        raise ValueError("Screen pertama dikunci seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    config, cwcf1, safeaug = _yaml(CONFIG), _yaml(CWCF1_CONFIG), _yaml(SAFEAUG_CONFIG)
    if config["model"] != safeaug["model"] or config["train"] != safeaug["train"]:
        raise RuntimeError("CWCF2 dan SAFEAUG0 tidak matched")
    frozen = CWCFConfig.from_mapping(config["cwcf"])
    if not frozen.explicit_composition or not frozen.class_balanced_attributes:
        raise RuntimeError("CWCF2 wajib explicit dan class-balanced")
    native, source = _build_native(checkpoint, seed)
    control = build_cwcf_model(
        str(MODEL_YAML), nc=NC, source=source, seed=seed,
        config=CWCFConfig.from_mapping(cwcf1["cwcf"]),
    )
    candidate = build_cwcf_model(
        str(MODEL_YAML), nc=NC, source=source, seed=seed, config=frozen
    )
    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)
    native_raw = _raw_predictions(native, probe)
    control_raw = _raw_predictions(control, probe)
    candidate_raw = _raw_predictions(candidate, probe)
    initial_boxes_exact = torch.equal(control_raw["boxes"], candidate_raw["boxes"])
    initial_scores_exact = torch.equal(control_raw["scores"], candidate_raw["scores"])
    native_scores_exact = torch.equal(native_raw["scores"], candidate_raw["scores"])
    head = candidate.model[-1]
    with torch.no_grad():
        head.composition_gates[0].fill_(1.0)
    active = _raw_predictions(candidate, probe)
    active_boxes_exact = torch.equal(control_raw["boxes"], active["boxes"])
    active_scores_change = not torch.equal(control_raw["scores"], active["scores"])

    matrix = build_j25_attribute_matrix()
    attribute = torch.full((1, matrix.shape[1], 1, 1), -5.0, requires_grad=True)
    attribute.data[:, 5:7] = 5.0
    compatibility = attribute_compatibility_logits(attribute, matrix)
    conjunction_order = bool(
        compatibility[0, 7] > compatibility[0, 8]
        and compatibility[0, 7] > compatibility[0, 12]
    )
    compatibility[0, 7].backward()
    logits = torch.zeros(4, matrix.shape[1], requires_grad=True)
    labels = torch.tensor([7, 7, 7, 12])
    auxiliary = balanced_attribute_bce(logits, matrix[labels], labels)
    auxiliary.backward()
    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "same_model_yaml_as_safeaug0": config["model"] == safeaug["model"],
        "same_50_epoch_schedule_as_safeaug0": config["train"] == safeaug["train"],
        "same_wavelet_chroma_core_as_cwcf1": all(
            config["cwcf"][key] == cwcf1["cwcf"][key]
            for key in ("cue_channels", "attribute_gain", "wavelet_levels", "cue_clip")
        ),
        "sampler_is_none": config.get("sampler") == "none",
        "initial_boxes_bitwise_exact_to_cwcf1": bool(initial_boxes_exact),
        "initial_scores_bitwise_exact_to_cwcf1": bool(initial_scores_exact),
        "initial_scores_bitwise_exact_to_native": bool(native_scores_exact),
        "active_composition_preserves_boxes": bool(active_boxes_exact),
        "active_composition_changes_scores": bool(active_scores_change),
        "black_broken_compatibility_exceeds_primitives": conjunction_order,
        "composition_gradients_finite_nonzero": bool(
            attribute.grad is not None and torch.isfinite(attribute.grad).all()
            and attribute.grad.abs().sum() > 0
        ),
        "balanced_attribute_gradients_finite_nonzero": bool(
            logits.grad is not None and torch.isfinite(logits.grad).all()
            and logits.grad.abs().sum() > 0
        ),
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.coffee_standard_j25.cwcf2.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "config_sha256": _sha256(CONFIG),
        "cwcf1_config_sha256": _sha256(CWCF1_CONFIG),
        "safeaug0_config_sha256": _sha256(SAFEAUG_CONFIG),
        "native_parameters": _parameter_count(native),
        "cwcf1_parameters": _parameter_count(control),
        "candidate_parameters": _parameter_count(candidate),
        "added_vs_native": _parameter_count(candidate) - _parameter_count(native),
        "added_vs_cwcf1": _parameter_count(candidate) - _parameter_count(control),
        "cwcf": frozen.to_dict(),
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


def _trainer(*, pretrained: Path, seed: int, config: CWCFConfig):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer

    class J25CWCF2Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if bool(getattr(self.args, "resume", False)):
                model = build_cwcf_model(
                    str(MODEL_YAML), nc=self.data["nc"], source=None, seed=seed,
                    config=config, verbose=verbose,
                )
                if weights:
                    load_cwcf_weights(model, weights)
                return self.set_model_names_for_load(model)
            source = YOLO(str(pretrained)).model
            model = build_cwcf_model(
                str(MODEL_YAML), nc=self.data["nc"], source=source, seed=seed,
                config=config, verbose=verbose,
            )
            return self.set_model_names_for_load(model)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    return J25CWCF2Trainer


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
        raise RuntimeError("CWCF2 hanya untuk J25 train-siblings v2")
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static = run_static_preflight(checkpoint, destination / "static_preflight.json", seed=seed)
    config = _yaml(CONFIG)
    frozen = CWCFConfig.from_mapping(config["cwcf"])
    train_args = dict(config["train"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = destination / "val_reports" / f"{ARM}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.coffee_standard_j25.cwcf2.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "source_archive_sha256": dataset["source_archive_sha256"],
        "development_contract_sha256": dataset["development_contract_sha256"],
        "provenance_summary_sha256": dataset["provenance_summary_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "config_sha256": static["config_sha256"],
        "cwcf": frozen.to_dict(),
        "train": train_args,
        "sampler": "none",
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _json(result_path, "CWCF2 result")
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
        ) as lease:
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
            # The Drive-visible heartbeat detects lock replacement.  Checking the
            # shared lease at batch boundaries prevents a second Colab runtime
            # from silently continuing to write this run after taking the lock.
            model.add_callback("on_train_batch_start", lambda _trainer: lease.assert_owned())
            model.add_callback("on_val_batch_start", lambda _validator: lease.assert_owned())
            model.train(
                trainer=_trainer(pretrained=checkpoint, seed=seed, config=frozen), **args
            )
        training_executed = True
    if not _run_complete(run_dir, int(train_args["epochs"])) or not best.is_file():
        raise RuntimeError("Run belum selesai secara valid")
    evaluation = evaluate(
        best, root, destination / "val_reports" / f"{ARM}_seed{seed}_val.json",
        split="val", device=device,
    )
    metrics = evaluation["metrics"]
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.coffee_standard_j25.cwcf2.arm_result.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "metrics": {metric: float(metrics[metric]) for metric in METRICS},
        "map50_95_by_class": metrics["map50_95_by_class"],
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


def evaluate_quarantined_run(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
) -> dict:
    """Evaluate a loadable checkpoint from an interleaved run without legitimizing it.

    This is intentionally diagnostic-only.  It answers whether a clean rerun is
    worth the compute, while preserving the concurrent-writer failure in evidence.
    """

    if seed != SEED:
        raise RuntimeError("Diagnostic CWCF2 dikunci seed 42")
    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    destination = Path(output_root).expanduser().resolve()
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    best = run_dir / "weights/best.pt"
    contract_path = run_dir / "run_contract.json"
    results_path = run_dir / "results.csv"
    if not best.is_file() or not contract_path.is_file() or not results_path.is_file():
        raise FileNotFoundError("Artefak run CWCF2 terkontaminasi tidak lengkap")
    contract = _json(contract_path, "CWCF2 run contract")
    if (
        contract.get("protocol") != PROTOCOL
        or contract.get("seed") != seed
        or contract.get("test_images_accessed") is not False
        or contract.get("source_archive_sha256") != dataset["source_archive_sha256"]
        or contract.get("development_contract_sha256")
        != dataset["development_contract_sha256"]
        or contract.get("provenance_summary_sha256")
        != dataset["provenance_summary_sha256"]
    ):
        raise RuntimeError("Kontrak run terkontaminasi tidak cocok dengan dataset J25")
    sequence = _epoch_sequence(results_path)
    expected = list(range(sequence[0], sequence[0] + len(sequence))) if sequence else []
    if sequence == expected:
        raise RuntimeError("Run monotonik; jalur quarantine tidak boleh digunakan")

    report_path = (
        destination / "quarantine_reports" / f"{ARM}_seed{seed}_quarantined_val.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation = evaluate(best, root, report_path, split="val", device=device)
    metrics = evaluation["metrics"]
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.coffee_standard_j25.cwcf2.quarantined_diagnostic.v1",
        "protocol": PROTOCOL,
        "status": "QUARANTINED_CONCURRENT_WRITER_DIAGNOSTIC_ONLY",
        "valid_for_claims": False,
        "seed": seed,
        "metrics": {metric: float(metrics[metric]) for metric in METRICS},
        "map50_95_by_class": metrics["map50_95_by_class"],
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "observed_epoch_sequence": sequence,
        "training_executed_this_call": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "run_contract": contract,
        "next": "CLEAN_SINGLE_WRITER_RERUN_ONLY_IF_DIAGNOSTIC_IS_PROMISING",
    }
    output = (
        destination / "quarantine_reports" / f"{ARM}_seed{seed}_quarantined_result.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_decision(
    direct_result: str | Path,
    safeaug_result: str | Path,
    cwcf1_result: str | Path,
    cwcf2_result: str | Path,
    output: str | Path,
) -> dict:
    inputs = {
        "D0DIRECT": (_json(direct_result, "D0DIRECT"), DIRECT_PROTOCOL),
        "SAFEAUG0": (_json(safeaug_result, "SAFEAUG0"), SAFEAUG_PROTOCOL),
        "CWCF1": (_json(cwcf1_result, "CWCF1"), CWCF1_PROTOCOL),
        "CWCF2": (_json(cwcf2_result, "CWCF2"), PROTOCOL),
    }
    for name, (payload, protocol) in inputs.items():
        if payload.get("protocol") != protocol or payload.get("test_images_accessed") is not False:
            raise RuntimeError(f"Kontrak reference salah: {name}")
        if payload.get("valid_for_claims") is False:
            raise RuntimeError(f"Artefak quarantine tidak boleh masuk decision: {name}")
    values = {name: payload["metrics"] for name, (payload, _) in inputs.items()}
    deltas = {
        reference: {
            metric: float(values["CWCF2"][metric]) - float(values[reference][metric])
            for metric in METRICS
        }
        for reference in ("D0DIRECT", "SAFEAUG0", "CWCF1")
    }
    versus_cwcf1 = deltas["CWCF1"]
    if all(value > 0 for value in versus_cwcf1.values()):
        status = "DOMINATES_CWCF1"
    elif any(value > 0 for value in versus_cwcf1.values()) and any(
        value < 0 for value in versus_cwcf1.values()
    ):
        status = "PARETO_TRADEOFF_VS_CWCF1"
    else:
        status = "NO_ADVANTAGE_VS_CWCF1"
    result = {
        "format": "coffee_detector.coffee_standard_j25.cwcf2.seed42_decision.v1",
        "protocol": PROTOCOL,
        "values": values,
        "cwcf2_minus": deltas,
        "status": status,
        "claim_boundary": "single-seed matched screening; no test or confirmation claim",
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


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
    parser.add_argument("--quarantine-evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.quarantine_evaluate_only:
        evaluate_quarantined_run(
            args.data_root, args.development_contract, args.provenance_summary,
            args.output_root, seed=args.seed, device=args.device,
        )
    else:
        run_arm(
            args.data_root, args.development_contract, args.provenance_summary,
            args.pretrained_checkpoint, args.output_root, seed=args.seed,
            device=args.device, authorize_training=args.authorize_training,
        )


if __name__ == "__main__":
    main()
