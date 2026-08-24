from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch
import yaml

from coffee_detector.afab.model import AFABDetectionModel, load_afab_weights
from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer
from coffee_detector.analysis.faruq_v3_diagnostics import run_faruq_v3_diagnostics
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
NATIVE_CONFIG = REPO_ROOT / "configs/coffee_fg/D0_yolo26n_p3.yaml"
AF2_CONFIG = REPO_ROOT / "configs/afab/AF2_yolo26n_chaotic_amplitude.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
ARMS = ("D0DIRECT", "AF2DIRECT")
SEED = 42
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
EXPECTED_AF2 = {
    "mode": "af2",
    "patch_size": 32,
    "overlap": 0.50,
    "radius_ratio": 0.05,
    "gamma": 0.10,
    "angular_bins": 360,
    "chunk_size": 128,
    "eps": 1.0e-8,
}

# Prospectively frozen seed-42 promotion thresholds.
MIN_RAW_PROPOSAL_DELTA = -0.005
ROUTE_A_MIN_MACRO = 0.005
ROUTE_A_MIN_BOTTOM3 = -0.005
ROUTE_A_MIN_WORST = -0.005
ROUTE_B_MIN_MACRO = -0.002
ROUTE_B_MIN_BOTTOM3 = 0.010
ROUTE_B_MIN_WORST = 0.010


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config bukan mapping: {path}")
    return payload


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _state_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _source_class_count(source: torch.nn.Module) -> int | None:
    nc = getattr(source, "nc", None)
    if nc is not None:
        return int(nc)
    names = getattr(source, "names", None)
    if isinstance(names, (dict, list, tuple)):
        return len(names)
    return None


def _build_initialized_detector(
    *,
    use_af2: bool,
    pretrained_checkpoint: Path,
    af2_config: AFABConfig,
    seed: int,
    verbose: bool = False,
) -> tuple[torch.nn.Module, dict | None]:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    source = YOLO(str(pretrained_checkpoint)).model
    source_nc = _source_class_count(source)
    if source_nc != 80:
        raise RuntimeError(
            "Direct protocol hanya menerima pretrained 80-class source; "
            f"diterima nc={source_nc}. Jangan gunakan D0/D0FT/coffee checkpoint."
        )

    # The target 21-class head contains tensors that cannot be copied from the
    # 80-class source. Forking RNG makes those fresh tensors exactly matched
    # between native and AF2 without perturbing the trainer's augmentation RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if use_af2:
            model = AFABDetectionModel(
                str(MODEL_YAML),
                ch=3,
                nc=21,
                verbose=verbose,
                afab=af2_config,
            )
            transfer = load_afab_weights(model, source)
        else:
            model = DetectionModel(
                str(MODEL_YAML), ch=3, nc=21, verbose=verbose
            )
            model.load(source)
            transfer = None
    return model, transfer


def run_direct_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if seed != SEED:
        raise ValueError("Seed-42 screen dikunci pada seed 42")

    native_cfg = _load_yaml(NATIVE_CONFIG)
    af2_cfg = _load_yaml(AF2_CONFIG)
    if Path(native_cfg.get("model", "")) != Path(af2_cfg.get("model", "")):
        raise RuntimeError("Native dan AF2 tidak memakai model YAML yang sama")
    if native_cfg.get("train") != af2_cfg.get("train"):
        raise RuntimeError("Native dan AF2 tidak schedule-matched")
    if af2_cfg.get("afab") != EXPECTED_AF2:
        raise RuntimeError(
            f"AF2 config berubah dari frozen protocol: {af2_cfg.get('afab')}"
        )
    if int(native_cfg["train"].get("epochs", 0)) != 50:
        raise RuntimeError("Direct screen harus mempertahankan maximum 50 epochs")

    frozen = AFABConfig.from_mapping(af2_cfg["afab"])
    native, native_transfer = _build_initialized_detector(
        use_af2=False,
        pretrained_checkpoint=checkpoint,
        af2_config=frozen,
        seed=seed,
    )
    candidate, candidate_transfer = _build_initialized_detector(
        use_af2=True,
        pretrained_checkpoint=checkpoint,
        af2_config=frozen,
        seed=seed,
    )
    native_state = native.state_dict()
    candidate_state = candidate.state_dict()
    same_keys = list(native_state) == list(candidate_state)
    exact_tensors = bool(
        same_keys
        and all(
            torch.equal(native_state[key].detach().cpu(), candidate_state[key].detach().cpu())
            for key in native_state
        )
    )
    native_params = _parameter_count(native)
    candidate_params = _parameter_count(candidate)
    frontend = AFABInputEnhancer(frozen)
    frontend_params = _parameter_count(frontend)
    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64, dtype=torch.float32).reshape(1, 3, 64, 64)
    with torch.inference_mode():
        enhanced = frontend(probe)
    transform_max_abs = float((enhanced - probe).abs().max().item())
    finite = bool(torch.isfinite(enhanced).all().item())
    shape_preserved = tuple(enhanced.shape) == tuple(probe.shape)
    common_fingerprint = _state_fingerprint(native)

    gates = {
        "same_model_yaml": Path(native_cfg["model"]) == Path(af2_cfg["model"]),
        "training_schedule_exact": native_cfg["train"] == af2_cfg["train"],
        "af2_config_frozen": af2_cfg["afab"] == EXPECTED_AF2,
        "detector_state_keys_exact": same_keys,
        "detector_state_tensors_exact": exact_tensors,
        "detector_parameter_count_exact": native_params == candidate_params,
        "af2_learned_parameters_zero": frontend_params == 0,
        "af2_probe_shape_preserved": shape_preserved,
        "af2_probe_finite": finite,
        "af2_probe_live": transform_max_abs > 0.0,
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.af2_direct.static_preflight.v1",
        "protocol": "faruq-v3-af2-direct-from-pretrained-seed42-v1",
        "seed": seed,
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_images_accessed": False,
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": _sha256(checkpoint),
        "pretrained_source_nc": 80,
        "native_config": str(NATIVE_CONFIG),
        "native_config_sha256": _sha256(NATIVE_CONFIG),
        "af2_config": str(AF2_CONFIG),
        "af2_config_sha256": _sha256(AF2_CONFIG),
        "common_initialized_detector_state_sha256": common_fingerprint,
        "native_parameter_count": native_params,
        "candidate_parameter_count": candidate_params,
        "af2_learned_parameter_count": frontend_params,
        "af2_probe_max_abs_change": transform_max_abs,
        "candidate_weight_transfer": candidate_transfer,
        "native_weight_transfer": native_transfer,
        "gates": gates,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if decision != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"AF2 direct static preflight FAIL: {failed}")
    return payload


def _make_direct_trainer(
    *,
    use_af2: bool,
    af2_config: AFABConfig,
    pretrained_checkpoint: Path,
    seed: int,
    expected_initial_fingerprint: str,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class DirectTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            resumed = bool(getattr(self.args, "resume", False))
            if resumed:
                # Resume reconstructs the same model class and then loads the
                # run-local checkpoint supplied by Ultralytics.
                if use_af2:
                    model = AFABDetectionModel(
                        str(MODEL_YAML),
                        nc=self.data["nc"],
                        ch=self.data["channels"],
                        verbose=verbose,
                        afab=af2_config,
                    )
                    if weights:
                        load_afab_weights(model, weights)
                else:
                    from ultralytics.nn.tasks import DetectionModel

                    model = DetectionModel(
                        str(MODEL_YAML),
                        nc=self.data["nc"],
                        ch=self.data["channels"],
                        verbose=verbose,
                    )
                    if weights:
                        model.load(weights)
                return self.set_model_names_for_load(model)

            model, _ = _build_initialized_detector(
                use_af2=use_af2,
                pretrained_checkpoint=pretrained_checkpoint,
                af2_config=af2_config,
                seed=seed,
                verbose=verbose,
            )
            fingerprint = _state_fingerprint(model)
            if fingerprint != expected_initial_fingerprint:
                raise RuntimeError(
                    "Initial detector state berbeda dari paired static preflight: "
                    f"{fingerprint} != {expected_initial_fingerprint}"
                )
            print(
                "DIRECT INITIALIZATION MATCH:", fingerprint,
                "| AF2=", use_af2,
                flush=True,
            )
            return self.set_model_names_for_load(model)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    DirectTrainer.__name__ = "AF2DirectTrainer" if use_af2 else "NativeDirectTrainer"
    return DirectTrainer


def _completed_epochs(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    with results_csv.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _arm_result_path(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / "val_reports" / f"{arm}_seed{seed}_result.json"


def _load_complete_arm_result(path: Path, contract: dict) -> dict | None:
    if not path.is_file():
        return None
    payload = _load_json(path, "Arm result")
    if payload.get("run_contract") != contract:
        raise RuntimeError(f"Result lama memiliki kontrak berbeda: {path}")
    if payload.get("evaluation_split") != "val" or payload.get("test_images_accessed") is not False:
        raise RuntimeError(f"Result lama tidak mempertahankan validation/test lock: {path}")
    return payload


def _run_arm(
    arm: str,
    data_root: Path,
    grouped_summary: Path,
    pretrained_checkpoint: Path,
    static_preflight: dict,
    output_root: Path,
    *,
    seed: int,
    device: str,
) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    use_af2 = arm == "AF2DIRECT"
    native_cfg = _load_yaml(NATIVE_CONFIG)
    af2_cfg = _load_yaml(AF2_CONFIG)
    frozen = AFABConfig.from_mapping(af2_cfg["afab"])
    train_args = dict(native_cfg["train"])
    max_epochs = int(train_args["epochs"])

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    result_path = _arm_result_path(output_root, arm, seed)
    run_dir = output_root / arm / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.af2_direct.arm_contract.v1",
        "arm": arm,
        "seed": seed,
        "pretrained_checkpoint_sha256": static_preflight["pretrained_checkpoint_sha256"],
        "common_initialized_detector_state_sha256": static_preflight[
            "common_initialized_detector_state_sha256"
        ],
        "native_config_sha256": static_preflight["native_config_sha256"],
        "af2_config_sha256": static_preflight["af2_config_sha256"],
        "train": train_args,
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        previous = _load_json(contract_path, "Run contract")
        if previous != contract:
            raise RuntimeError(f"Run directory memiliki kontrak berbeda: {run_dir}")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    old_result = _load_complete_arm_result(result_path, contract)
    if old_result is not None:
        print(f"REUSE COMPLETE {arm} seed {seed}", flush=True)
        return old_result

    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    trainer = _make_direct_trainer(
        use_af2=use_af2,
        af2_config=frozen,
        pretrained_checkpoint=pretrained_checkpoint,
        seed=seed,
        expected_initial_fingerprint=static_preflight[
            "common_initialized_detector_state_sha256"
        ],
    )
    training_executed = False
    if not _run_complete(run_dir, max_epochs):
        from ultralytics import YOLO

        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root, lock_name=f"{arm}_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {arm} seed {seed} dari epoch {epoch + 1}", flush=True)
                model = YOLO(str(last))
                args = {"resume": True, "device": device}
            else:
                if last.is_file() and not resumable:
                    raise RuntimeError(
                        f"{arm} memiliki last.pt non-resumable tetapi belum dinilai complete"
                    )
                print(
                    f"START {arm} seed {seed} langsung dari official pretrained",
                    flush=True,
                )
                model = YOLO(str(MODEL_YAML))
                args = dict(train_args)
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root / arm),
                    name=f"{arm}_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=False,
                    verbose=False,
                    device=device,
                )
            model.train(trainer=trainer, **args)
        training_executed = True

    if not _run_complete(run_dir, max_epochs):
        raise RuntimeError(f"Training {arm} belum complete/early-stopped secara valid: {run_dir}")
    if not best.is_file():
        raise FileNotFoundError(best)

    eval_path = reports / f"{arm}_seed{seed}_val.json"
    evaluation = evaluate(best, data_root, eval_path, split="val", device=device)
    if evaluation["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    diagnostic_path = reports / f"{arm}_seed{seed}_diagnostic.json"
    diagnostic = run_faruq_v3_diagnostics(
        best,
        data_root,
        diagnostic_path,
        split="val",
        device=device,
    )
    result = {
        "format": "coffee_detector.af2_direct.arm_result.v1",
        "protocol": "faruq-v3-af2-direct-from-pretrained-seed42-v1",
        "arm": arm,
        "seed": seed,
        "metrics": {name: float(evaluation["metrics"][name]) for name in METRICS},
        "diagnostic": {
            "raw_top500_proposal_accessibility": float(
                diagnostic["raw_candidate_sensitivity"]["500"]["proposal_accessibility"]
            ),
            "localization_conditioned_top1": float(
                diagnostic["global"]["localization_conditioned_class_accuracy"]
            ),
            "correct_decision_recall": float(
                diagnostic["global"]["correct_class"]
                / max(int(diagnostic["global"]["targets"]), 1)
            ),
        },
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "completed_epochs": _completed_epochs(run_dir / "results.csv"),
        "maximum_epochs": max_epochs,
        "training_executed_this_call": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("DONE:", arm, seed, result["metrics"], result["diagnostic"], flush=True)
    return result


def _screen_decision(deltas: dict[str, float]) -> dict:
    localization_safe = deltas["raw_top500_proposal_accessibility"] >= MIN_RAW_PROPOSAL_DELTA
    route_a = bool(
        deltas["macro_map50_95"] >= ROUTE_A_MIN_MACRO
        and deltas["bottom3_class_map50_95"] >= ROUTE_A_MIN_BOTTOM3
        and deltas["worst_class_map50_95"] >= ROUTE_A_MIN_WORST
        and localization_safe
    )
    route_b = bool(
        deltas["macro_map50_95"] >= ROUTE_B_MIN_MACRO
        and deltas["bottom3_class_map50_95"] >= ROUTE_B_MIN_BOTTOM3
        and deltas["worst_class_map50_95"] >= ROUTE_B_MIN_WORST
        and localization_safe
    )
    return {
        "localization_safe": localization_safe,
        "route_a_direct_overall_gain": route_a,
        "route_b_lower_tail_pareto": route_b,
        "decision": "PROMOTE_TO_3_SEED" if route_a or route_b else "DO_NOT_PROMOTE",
    }


def build_pair_summary(control: dict, candidate: dict, output: str | Path) -> dict:
    if control.get("arm") != "D0DIRECT" or candidate.get("arm") != "AF2DIRECT":
        raise ValueError("Pair harus D0DIRECT vs AF2DIRECT")
    if control.get("seed") != candidate.get("seed"):
        raise ValueError("Pair seed tidak cocok")
    if control.get("test_images_accessed") is not False or candidate.get("test_images_accessed") is not False:
        raise RuntimeError("Test lock tidak valid")

    deltas = {
        name: float(candidate["metrics"][name] - control["metrics"][name])
        for name in METRICS
    }
    for name in (
        "raw_top500_proposal_accessibility",
        "localization_conditioned_top1",
        "correct_decision_recall",
    ):
        deltas[name] = float(candidate["diagnostic"][name] - control["diagnostic"][name])
    screen = _screen_decision(deltas)
    payload = {
        "format": "coffee_detector.af2_direct.pair_summary.v1",
        "protocol": "faruq-v3-af2-direct-from-pretrained-seed42-v1",
        "seed": control["seed"],
        "control": control,
        "candidate": candidate,
        "deltas_af2direct_minus_d0direct": deltas,
        "screen": screen,
        "thresholds": {
            "minimum_raw_top500_proposal_delta": MIN_RAW_PROPOSAL_DELTA,
            "route_a": {
                "minimum_macro_delta": ROUTE_A_MIN_MACRO,
                "minimum_bottom3_delta": ROUTE_A_MIN_BOTTOM3,
                "minimum_worst_delta": ROUTE_A_MIN_WORST,
            },
            "route_b": {
                "minimum_macro_delta": ROUTE_B_MIN_MACRO,
                "minimum_bottom3_delta": ROUTE_B_MIN_BOTTOM3,
                "minimum_worst_delta": ROUTE_B_MIN_WORST,
            },
        },
        "training_executed": bool(
            control.get("training_executed_this_call")
            or candidate.get("training_executed_this_call")
        ),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "claim_limit": (
            "seed-42 promotion screen only; PASS authorizes a prospective three-seed "
            "confirmation and is not a final superiority claim"
        ),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def run_faruq_v3_af2_direct(
    data_root: str | Path,
    grouped_summary: str | Path,
    pretrained_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if seed != SEED:
        raise ValueError("Direct screen pertama dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    root = Path(data_root).expanduser().resolve()
    grouped = Path(grouped_summary).expanduser().resolve()
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    load_faruq_grouped_summary(grouped, root)
    audit_path = destination / "val_reports/dataset_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(root, audit_path, near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Dataset audit gagal")

    static_path = destination / "direct_static_preflight.json"
    if static_path.is_file():
        static = _load_json(static_path, "Static preflight")
        expected_sha = _sha256(checkpoint)
        if static.get("pretrained_checkpoint_sha256") != expected_sha:
            raise RuntimeError("Pretrained checkpoint berubah dari static preflight")
        if static.get("decision") != "PASS" or not static.get("training_authorized"):
            raise RuntimeError("Static preflight lama bukan PASS")
    else:
        static = run_direct_static_preflight(checkpoint, static_path, seed=seed)

    control = _run_arm(
        "D0DIRECT", root, grouped, checkpoint, static, destination, seed=seed, device=device
    )
    candidate = _run_arm(
        "AF2DIRECT", root, grouped, checkpoint, static, destination, seed=seed, device=device
    )
    summary = build_pair_summary(
        control,
        candidate,
        destination / "af2_direct_seed42_summary.json",
    )
    print(json.dumps({
        "deltas": summary["deltas_af2direct_minus_d0direct"],
        "screen": summary["screen"],
        "test_images_accessed": summary["test_images_accessed"],
    }, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched native-vs-AF2 direct training from official YOLO26n pretrained"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_af2_direct(
        args.data_root,
        args.grouped_summary,
        args.pretrained_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
