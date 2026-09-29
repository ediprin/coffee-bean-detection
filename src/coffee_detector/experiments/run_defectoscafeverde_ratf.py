"""Fresh single-arm RATF1 screen against the matched Defectos D0DIRECT run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coffee_detector.data.prepare_defectoscafeverde_grouped import NAMES
from coffee_detector.defectos_ratf import (
    RATFConfig,
    build_ratf_model,
    load_ratf_weights,
    ratf_texture_cue,
)
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_defectoscafeverde_af2_direct import (
    METRICS,
    MODEL_YAML,
    NATIVE_CONFIG,
    NC,
    SEED,
    _completed_epochs,
    _load_json,
    _load_yaml,
    validate_development_dataset,
)
from coffee_detector.experiments.run_defectoscafeverde_cwcf_direct import (
    PROTOCOL as D0_REFERENCE_PROTOCOL,
    _build_native,
    _raw,
    _repair_completed_resume_boundary_csv,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import (
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
CANDIDATE_CONFIG = REPO_ROOT / "configs/defectoscafeverde/RATF1.yaml"
ARM = "RATF1"
PROTOCOL = "defectoscafeverde-grouped-ratf1-seed42-v1"
HARD_CLASSES = ("agrio", "concha", "negro", "oreja", "partido")


def _build_candidate(
    checkpoint: Path,
    seed: int,
    config: RATFConfig,
    *,
    verbose: bool = False,
):
    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("Official pretrained source harus memiliki 80 kelas")
    return build_ratf_model(
        str(MODEL_YAML),
        nc=NC,
        source=source,
        seed=seed,
        config=config,
        verbose=verbose,
    )


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen pertama dikunci seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    native_yaml = _load_yaml(NATIVE_CONFIG)
    candidate_yaml = _load_yaml(CANDIDATE_CONFIG)
    config = RATFConfig.from_mapping(candidate_yaml["ratf"])
    native, _ = _build_native(checkpoint, seed)
    candidate = _build_candidate(checkpoint, seed, config)
    native_head, candidate_head = native.model[-1], candidate.model[-1]
    common_backbone = all(
        torch.equal(left.detach().cpu(), right.detach().cpu())
        for native_module, candidate_module in zip(native.model[:-1], candidate.model[:-1])
        for left, right in zip(
            native_module.state_dict().values(), candidate_module.state_dict().values()
        )
    )
    native_head_state = native_head.state_dict()
    candidate_native_state = candidate_head.base_head.state_dict()
    head_keys_equal = tuple(native_head_state) == tuple(candidate_native_state)
    head_tensors_equal = head_keys_equal and all(
        torch.equal(native_head_state[key], candidate_native_state[key])
        for key in native_head_state
    )

    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)
    cue_first = ratf_texture_cue(probe, config)
    cue_second = ratf_texture_cue(probe, config)
    native_raw, candidate_raw = _raw(native, probe), _raw(candidate, probe)
    initial_boxes_exact = torch.equal(native_raw["boxes"], candidate_raw["boxes"])
    initial_scores_exact = torch.equal(native_raw["scores"], candidate_raw["scores"])

    adapter = candidate_head.adapters[0]
    feature = torch.randn(2, adapter.channels, 8, 8)
    small_cue = torch.randn(2, config.cue_channels, 32, 32)
    novel, reference = adapter.decompose(feature, small_cue)
    dot = (novel * reference).sum(dim=1).abs().max()
    scale = (novel.norm(dim=1) * reference.norm(dim=1)).clamp_min(1e-8).max()
    orthogonality_relative_error = float((dot / scale).detach())

    candidate.zero_grad(set_to_none=True)
    candidate.train()
    gradient_output = candidate(probe.clone())
    gradient_raw = (
        gradient_output["one2many"]
        if "one2many" in gradient_output
        else gradient_output
    )
    gradient_raw["scores"].sum().backward()
    score_gradient = candidate_head.adapters[0].class_residual.weight.grad
    gradient_live = bool(
        score_gradient is not None
        and torch.isfinite(score_gradient).all()
        and score_gradient.abs().sum() > 0
    )
    candidate.zero_grad(set_to_none=True)

    with torch.no_grad():
        candidate_head.adapters[0].class_residual.weight[0, 0, 0, 0] = 0.25
    active_raw = _raw(candidate, probe)
    active_boxes_exact = torch.equal(native_raw["boxes"], active_raw["boxes"])
    active_scores_change = not torch.equal(native_raw["scores"], active_raw["scores"])
    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "target_class_count_12": getattr(native_head, "nc", None) == NC,
        "same_model_yaml": native_yaml["model"] == candidate_yaml["model"],
        "same_50_epoch_schedule": (
            native_yaml["train"] == candidate_yaml["train"]
            and native_yaml["train"]["epochs"] == 50
        ),
        "fresh_official_pretrained_start": candidate_yaml["weights"] == "yolo26n.pt",
        "common_backbone_tensors_exact": bool(common_backbone),
        "native_head_schema_exact": bool(head_keys_equal),
        "native_head_tensors_exact": bool(head_tensors_equal),
        "cue_shape_exact": tuple(cue_first.shape) == (1, 3, 64, 64),
        "cue_finite_and_live": bool(
            torch.isfinite(cue_first).all() and cue_first.abs().sum() > 0
        ),
        "cue_deterministic": bool(torch.equal(cue_first, cue_second)),
        "texture_projection_orthogonal": orthogonality_relative_error < 1e-4,
        "initial_boxes_bitwise_exact": bool(initial_boxes_exact),
        "initial_scores_bitwise_exact": bool(initial_scores_exact),
        "classification_residual_gradient_finite_nonzero": gradient_live,
        "active_texture_preserves_boxes": bool(active_boxes_exact),
        "active_texture_changes_scores": bool(active_scores_change),
        "test_not_accessed": True,
    }
    native, _ = _build_native(checkpoint, seed)
    candidate = _build_candidate(checkpoint, seed, config)
    payload = {
        "format": "coffee_detector.defectoscafeverde.ratf.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "native_initial_state_sha256": _state_fingerprint(native),
        "candidate_initial_state_sha256": _state_fingerprint(candidate),
        "native_config_sha256": _sha256(NATIVE_CONFIG),
        "candidate_config_sha256": _sha256(CANDIDATE_CONFIG),
        "native_parameters": _parameter_count(native),
        "candidate_parameters": _parameter_count(candidate),
        "candidate_added_parameters": _parameter_count(candidate) - _parameter_count(native),
        "orthogonality_relative_error": orthogonality_relative_error,
        "ratf": config.to_dict(),
        "hard_classes": list(HARD_CLASSES),
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


def _trainer(
    *, checkpoint: Path, seed: int, config: RATFConfig, expected_fingerprint: str
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class RATFTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if bool(getattr(self.args, "resume", False)):
                model = build_ratf_model(
                    str(MODEL_YAML), nc=self.data["nc"], source=None,
                    seed=seed, config=config, verbose=verbose,
                )
                if weights:
                    load_ratf_weights(model, weights)
                return self.set_model_names_for_load(model)
            model = _build_candidate(checkpoint, seed, config, verbose=verbose)
            if _state_fingerprint(model) != expected_fingerprint:
                raise RuntimeError("Initial state RATF1 tidak cocok dengan static preflight")
            return self.set_model_names_for_load(model)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    return RATFTrainer


def run_arm(
    data_root: str | Path,
    grouped_audit: str | Path,
    pretrained_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != SEED or not authorize_training:
        raise RuntimeError("Seed-42 dan --authorize-training wajib")
    root = Path(data_root).expanduser().resolve()
    dataset = validate_development_dataset(root, grouped_audit)
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static_path = destination / "static_preflight_RATF1.json"
    static = run_static_preflight(checkpoint, static_path, seed=seed)
    static["dataset_audit_sha256"] = _sha256(grouped_audit)
    static["dataset_contract"] = dataset
    static_path.write_text(json.dumps(static, indent=2) + "\n", encoding="utf-8")
    train_args = dict(_load_yaml(NATIVE_CONFIG)["train"])
    config = RATFConfig.from_mapping(_load_yaml(CANDIDATE_CONFIG)["ratf"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = destination / "val_reports" / f"{ARM}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.defectoscafeverde.ratf.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "dataset_audit_sha256": static["dataset_audit_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "initial_state_sha256": static["candidate_initial_state_sha256"],
        "native_config_sha256": static["native_config_sha256"],
        "candidate_config_sha256": static["candidate_config_sha256"],
        "train": train_args,
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _load_json(result_path, "RATF1 result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing RATF1 result berbeda kontrak")
        return old
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _load_json(contract_path, "Run contract") != contract:
        if _completed_epochs(run_dir / "results.csv") > 0 or any(
            path.is_file() for path in (run_dir / "weights/best.pt", run_dir / "weights/last.pt")
        ):
            raise RuntimeError("Run directory RATF1 berbeda kontrak")
        contract_path.unlink()
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    csv_repair = _repair_completed_resume_boundary_csv(run_dir)
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
                    raise RuntimeError("last.pt RATF1 tidak resumable")
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
                    checkpoint=checkpoint,
                    seed=seed,
                    config=config,
                    expected_fingerprint=static["candidate_initial_state_sha256"],
                ),
                **args,
            )
        training_executed = True
    post_training_repair = _repair_completed_resume_boundary_csv(run_dir)
    if (
        post_training_repair.get("status") == "repaired_adjacent_duplicate_epochs"
        or csv_repair.get("status") != "repaired_adjacent_duplicate_epochs"
    ):
        csv_repair = post_training_repair
    if not _run_complete(run_dir, int(train_args["epochs"])) or not best.is_file():
        raise RuntimeError("Run RATF1 belum selesai secara valid")
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
        "format": "coffee_detector.defectoscafeverde.ratf.arm_result.v1",
        "protocol": PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "metrics": {metric: float(metrics[metric]) for metric in METRICS},
        "map50_95_by_class": metrics["map50_95_by_class"],
        "hard_class_map50_95": {
            name: float(metrics["map50_95_by_class"][name]) for name in HARD_CLASSES
        },
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "completed_epochs": _completed_epochs(run_dir / "results.csv"),
        "maximum_epochs": int(train_args["epochs"]),
        "training_executed_this_call": training_executed,
        "results_csv_repair": csv_repair,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_decision(
    d0_result: str | Path,
    candidate_result: str | Path,
    output: str | Path,
) -> dict:
    native = _load_json(d0_result, "D0DIRECT result")
    candidate = _load_json(candidate_result, "RATF1 result")
    if (
        native.get("protocol") != D0_REFERENCE_PROTOCOL
        or native.get("arm") != "D0DIRECT"
        or native.get("test_images_accessed") is not False
    ):
        raise RuntimeError("D0DIRECT reference tidak valid")
    if (
        candidate.get("protocol") != PROTOCOL
        or candidate.get("arm") != ARM
        or candidate.get("test_images_accessed") is not False
    ):
        raise RuntimeError("RATF1 result tidak valid")
    native_contract, candidate_contract = native["run_contract"], candidate["run_contract"]
    paired = {
        "dataset_audit_sha256_equal": native_contract["dataset_audit_sha256"]
        == candidate_contract["dataset_audit_sha256"],
        "pretrained_checkpoint_sha256_equal": native_contract["pretrained_checkpoint_sha256"]
        == candidate_contract["pretrained_checkpoint_sha256"],
        "native_config_sha256_equal": native_contract["native_config_sha256"]
        == candidate_contract["native_config_sha256"],
        "training_schedule_equal": native_contract["train"] == candidate_contract["train"],
        "same_seed": native.get("seed") == candidate.get("seed") == SEED,
        "test_not_opened": True,
    }
    if not all(paired.values()):
        raise RuntimeError(f"Paired contract gagal: {[key for key, value in paired.items() if not value]}")
    deltas = {
        metric: float(candidate["metrics"][metric]) - float(native["metrics"][metric])
        for metric in METRICS
    }
    overall_route = (
        deltas["macro_map50_95"] > 0
        and deltas["bottom3_class_map50_95"] >= 0
        and deltas["worst_class_map50_95"] >= 0
    )
    lower_tail_route = (
        deltas["macro_map50_95"] >= -0.001
        and deltas["bottom3_class_map50_95"] > 0
        and deltas["worst_class_map50_95"] > 0
    )
    native_classwise = native.get("map50_95_by_class", {})
    candidate_classwise = candidate.get("map50_95_by_class", {})
    hard_class_deltas = {
        name: float(candidate_classwise[name]) - float(native_classwise[name])
        for name in HARD_CLASSES
    }
    payload = {
        "format": "coffee_detector.defectoscafeverde.ratf.seed42_decision.v1",
        "protocol": PROTOCOL,
        "values": {"D0DIRECT": native["metrics"], ARM: candidate["metrics"]},
        "deltas": deltas,
        "hard_class_deltas": hard_class_deltas,
        "paired": paired,
        "screen": {
            "overall_pareto_route": overall_route,
            "lower_tail_pareto_route": lower_tail_route,
            "decision": (
                "PROMOTE_TO_PAIRED_CONFIRMATION"
                if overall_route or lower_tail_route
                else "STOP_AFTER_SEED42"
            ),
        },
        "evaluation_split": "val",
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-audit", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_arm(
        args.data_root,
        args.grouped_audit,
        args.pretrained_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
