from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from coffee_detector.analysis.faruq_v3_diagnostics import run_faruq_v3_diagnostics
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.dlrbc.model import (
    DLRBCConfig,
    DLRBCDetectionModel,
    DLRBCDetectHead,
    load_dlrbc_weights,
)
from coffee_detector.dlrbc.trainer import make_fresh_trainer
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
CONFIGS = {
    "B0_FRESH": REPO_ROOT / "configs/dlrbc/B0_FRESH.yaml",
    "LRLIN_FRESH": REPO_ROOT / "configs/dlrbc/LRLIN_FRESH.yaml",
    "DLRBC_FRESH": REPO_ROOT / "configs/dlrbc/DLRBC_FRESH.yaml",
}
ARMS = tuple(CONFIGS)
SEED = 42
OFFICIAL_YOLO26N_SHA256 = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)
EXPECTED_TRAIN = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "workers": 2,
    "patience": 15,
    "optimizer": "auto",
    "pretrained": True,
    "cache": False,
    "close_mosaic": 10,
    "max_det": 500,
}
EXPECTED_COMMON_DLRBC = {
    "rank": 8,
    "projection_ratio": 0.5,
    "minimum_projection": 16,
    "residual_scale": 0.1,
    "signed_sqrt": True,
    "eps": 1.0e-6,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config bukan mapping: {path}")
    return payload


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _require_official_pretrained(path: str | Path) -> tuple[Path, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = _sha256(source)
    if digest != OFFICIAL_YOLO26N_SHA256:
        raise RuntimeError(
            "Fresh source bukan official yolo26n.pt yang dibekukan: "
            f"sha256={digest}, expected={OFFICIAL_YOLO26N_SHA256}"
        )
    return source, digest


def _tensor_digest(entries: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(entries):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _module_digest(module: torch.nn.Module) -> str:
    return _tensor_digest(list(module.state_dict().items()))


def _parameter_count(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _base_detector_entries(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    entries: list[tuple[str, torch.Tensor]] = []
    for index, layer in enumerate(model.model):
        target = layer.base_head if isinstance(layer, DLRBCDetectHead) else layer
        entries.extend(
            (f"{index}.{key}", value) for key, value in target.state_dict().items()
        )
    return entries


def _adapter_entries(model: torch.nn.Module) -> list[tuple[str, torch.Tensor]]:
    head = model.model[-1]
    if not isinstance(head, DLRBCDetectHead):
        return []
    return [
        *[(f"one2many.{k}", v) for k, v in head.one2many_residuals.state_dict().items()],
        *[(f"one2one.{k}", v) for k, v in head.one2one_residuals.state_dict().items()],
    ]


def _source_class_count(source: torch.nn.Module) -> int | None:
    nc = getattr(source, "nc", None)
    if nc is not None:
        return int(nc)
    names = getattr(source, "names", None)
    if isinstance(names, (dict, list, tuple)):
        return len(names)
    return None


def _arm_config(arm: str) -> tuple[dict[str, Any], DLRBCConfig | None]:
    if arm not in CONFIGS:
        raise ValueError(f"Arm tidak dikenal: {arm}")
    payload = _load_yaml(CONFIGS[arm])
    config = DLRBCConfig.from_mapping(payload["dlrbc"]) if "dlrbc" in payload else None
    return payload, config


def build_fresh_detector(
    *,
    arm: str,
    pretrained_checkpoint: str | Path,
    seed: int,
    verbose: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build from official COCO weights; never consume a Coffee checkpoint."""

    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    checkpoint, checkpoint_sha = _require_official_pretrained(pretrained_checkpoint)
    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("Official source harus memiliki nc=80")
    _, config = _arm_config(arm)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if config is None:
            model = DetectionModel(str(MODEL_YAML), ch=3, nc=21, verbose=verbose)
            model.load(source)
            transfer = {"resume": 0, "kind": "native_shape_compatible"}
        else:
            model = DLRBCDetectionModel(
                str(MODEL_YAML), ch=3, nc=21, verbose=verbose, dlrbc=config
            )
            transfer = load_dlrbc_weights(model, source)
    metadata = {
        "arm": arm,
        "seed": int(seed),
        "official_pretrained_sha256": checkpoint_sha,
        "full_state_sha256": _module_digest(model),
        "base_detector_state_sha256": _tensor_digest(_base_detector_entries(model)),
        "adapter_state_sha256": (
            _tensor_digest(_adapter_entries(model)) if config is not None else None
        ),
        "parameter_count": _parameter_count(model),
        "transfer": transfer,
    }
    return model, metadata


def _raw_predictions(model: torch.nn.Module, probe: torch.Tensor) -> dict[str, dict]:
    model.eval()
    with torch.inference_mode():
        output = model(probe)
    if not isinstance(output, tuple) or not isinstance(output[1], dict):
        raise TypeError("Model tidak mengekspos raw dual-head predictions")
    raw = output[1]
    if not {"one2many", "one2one"} <= set(raw):
        raise KeyError("Raw output kehilangan dual head")
    return raw


def run_fresh_static_audit(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
    device: str = "cpu",
) -> dict[str, Any]:
    checkpoint, checkpoint_sha = _require_official_pretrained(pretrained_checkpoint)
    if seed != SEED:
        raise ValueError("Static screen pertama dikunci pada seed 42")
    configs = {arm: _load_yaml(path) for arm, path in CONFIGS.items()}
    _, linear_config = _arm_config("LRLIN_FRESH")
    _, quadratic_config = _arm_config("DLRBC_FRESH")
    assert linear_config is not None and quadratic_config is not None

    common_linear = linear_config.to_dict()
    common_quadratic = quadratic_config.to_dict()
    linear_mode = common_linear.pop("mode")
    quadratic_mode = common_quadratic.pop("mode")
    expected_common = EXPECTED_COMMON_DLRBC

    models: dict[str, torch.nn.Module] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        model, info = build_fresh_detector(
            arm=arm, pretrained_checkpoint=checkpoint, seed=seed, verbose=False
        )
        models[arm] = model
        metadata[arm] = info

    linear = models["LRLIN_FRESH"]
    quadratic = models["DLRBC_FRESH"]
    linear_head = linear.model[-1]
    quadratic_head = quadratic.model[-1]
    if not isinstance(linear_head, DLRBCDetectHead) or not isinstance(
        quadratic_head, DLRBCDetectHead
    ):
        raise TypeError("Static audit tidak memperoleh DLRBCDetectHead")
    widths = tuple(int(value) for value in quadratic_head.class_tower_channels)
    projections = tuple(
        int(module.projection_channels) for module in quadratic_head.one2many_residuals
    )

    torch_device = torch.device(device)
    for model in models.values():
        model.to(torch_device)
    probe = torch.linspace(
        0.0, 1.0, 3 * 64 * 64, dtype=torch.float32, device=torch_device
    ).reshape(1, 3, 64, 64)
    native_raw = _raw_predictions(models["B0_FRESH"], probe)
    linear_raw = _raw_predictions(linear, probe)
    quadratic_raw = _raw_predictions(quadratic, probe)
    branches = ("one2many", "one2one")
    boxes_linear_equal = all(
        torch.equal(native_raw[name]["boxes"], linear_raw[name]["boxes"])
        for name in branches
    )
    boxes_quadratic_equal = all(
        torch.equal(native_raw[name]["boxes"], quadratic_raw[name]["boxes"])
        for name in branches
    )
    linear_live = any(
        not torch.equal(native_raw[name]["scores"], linear_raw[name]["scores"])
        for name in branches
    )
    quadratic_live = any(
        not torch.equal(native_raw[name]["scores"], quadratic_raw[name]["scores"])
        for name in branches
    )
    functional_difference = any(
        not torch.equal(linear_raw[name]["scores"], quadratic_raw[name]["scores"])
        for name in branches
    )

    gradient_probe = torch.randn(
        2, widths[0], 8, 8, dtype=torch.float32, device=torch_device, requires_grad=True
    )
    gradient_module = quadratic_head.one2many_residuals[0]
    gradient_loss = gradient_module(gradient_probe).square().mean()
    gradient_loss.backward()
    finite_input_gradient = bool(
        gradient_probe.grad is not None
        and torch.isfinite(gradient_probe.grad).all().item()
        and gradient_probe.grad.abs().sum().item() > 0.0
    )
    finite_parameter_gradients = bool(
        all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in gradient_module.parameters()
        )
    )
    separate_dual_adapters = all(
        left.projection.weight.data_ptr() != right.projection.weight.data_ptr()
        for left, right in zip(
            quadratic_head.one2many_residuals,
            quadratic_head.one2one_residuals,
        )
    )

    schedule_match = all(payload.get("train") == EXPECTED_TRAIN for payload in configs.values())
    model_match = all(Path(payload.get("model", "")) == Path("configs/coffee_fg/models/yolo26n-p3.yaml") for payload in configs.values())
    config_hashes = {arm: _sha256(path) for arm, path in CONFIGS.items()}
    gates = {
        "official_pretrained_sha256_exact": checkpoint_sha == OFFICIAL_YOLO26N_SHA256,
        "same_model_yaml": model_match,
        "same_fresh_training_schedule": schedule_match,
        "no_coffee_parent_checkpoint": True,
        "linear_and_quadratic_config_matched": common_linear == common_quadratic == expected_common,
        "arm_modes_exact": linear_mode == "linear" and quadratic_mode == "quadratic",
        "base_detector_initial_state_exact": len({m["base_detector_state_sha256"] for m in metadata.values()}) == 1,
        "matched_adapter_initial_state_exact": metadata["LRLIN_FRESH"]["adapter_state_sha256"] == metadata["DLRBC_FRESH"]["adapter_state_sha256"],
        "matched_candidate_parameter_count": metadata["LRLIN_FRESH"]["parameter_count"] == metadata["DLRBC_FRESH"]["parameter_count"],
        "candidate_adds_parameters": metadata["DLRBC_FRESH"]["parameter_count"] > metadata["B0_FRESH"]["parameter_count"],
        "rank_even_and_projection_reduces": quadratic_config.rank % 2 == 0 and all(quadratic_config.rank <= m < c for m, c in zip(projections, widths)),
        "expected_head_width_and_projection": widths == (64, 64, 64) and projections == (32, 32, 32),
        "one2many_and_one2one_adapters_separate": separate_dual_adapters,
        "linear_residual_live": linear_live,
        "quadratic_residual_live": quadratic_live,
        "linear_quadratic_functionally_different": functional_difference,
        "linear_preserves_native_boxes": boxes_linear_equal,
        "quadratic_preserves_native_boxes": boxes_quadratic_equal,
        "finite_nonzero_input_gradient": finite_input_gradient,
        "finite_parameter_gradients": finite_parameter_gradients,
        "test_not_accessed": True,
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    payload = {
        "format": "coffee_detector.dlrbc_fresh.static_audit.v1",
        "protocol": "faruq-v3-dlrbc-fresh-seed42-v1",
        "seed": seed,
        "decision": decision,
        "training_authorized": decision == "PASS",
        "test_images_accessed": False,
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_checkpoint_sha256": checkpoint_sha,
        "expected_official_pretrained_sha256": OFFICIAL_YOLO26N_SHA256,
        "model_yaml": str(MODEL_YAML),
        "config_sha256": config_hashes,
        "arm_initial_state_sha256": {
            arm: metadata[arm]["full_state_sha256"] for arm in ARMS
        },
        "base_detector_initial_state_sha256": metadata["B0_FRESH"]["base_detector_state_sha256"],
        "adapter_initial_state_sha256": metadata["DLRBC_FRESH"]["adapter_state_sha256"],
        "parameter_count": {arm: metadata[arm]["parameter_count"] for arm in ARMS},
        "class_tower_channels": widths,
        "projection_channels": projections,
        "rank": quadratic_config.rank,
        "initialization": "official_yolo26n_shape_compatible_plus_fresh_head_orthogonal_xavier",
        "gates": gates,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if decision != "PASS":
        failed = [name for name, value in gates.items() if not value]
        raise RuntimeError(f"DLRBC fresh static audit FAIL: {failed}")
    return payload


def _completed_epochs(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    with results_csv.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _result_path(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / "val_reports" / f"{arm}_seed{seed}_result.json"


def run_faruq_v3_dlrbc_fresh_arm(
    arm: str,
    data_root: str | Path,
    grouped_summary: str | Path,
    pretrained_checkpoint: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
    authorize_training: bool = False,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"Arm tidak dikenal: {arm}")
    if seed != SEED:
        raise ValueError("Screen pertama dikunci pada seed 42")
    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi")
    root = Path(data_root).expanduser().resolve()
    grouped = Path(grouped_summary).expanduser().resolve()
    checkpoint, checkpoint_sha = _require_official_pretrained(pretrained_checkpoint)
    static_path = Path(static_audit).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if (root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    load_faruq_grouped_summary(grouped, root)
    audit_path = destination / "val_reports" / f"dataset_audit_{arm}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(root, audit_path, near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Dataset audit gagal")

    if static_path.is_file():
        static = _load_json(static_path, "Static audit")
    else:
        static = run_fresh_static_audit(
            checkpoint, static_path, seed=seed, device="cpu"
        )
    if static.get("decision") != "PASS" or not static.get("training_authorized"):
        raise RuntimeError("Static audit bukan PASS")
    if static.get("pretrained_checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("Official source berubah dari static audit")
    if static.get("test_images_accessed") is not False:
        raise RuntimeError("Static audit melanggar test lock")

    config_payload, dlrbc_config = _arm_config(arm)
    train_args = dict(config_payload["train"])
    max_epochs = int(train_args["epochs"])
    run_dir = destination / arm / f"{arm}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = _result_path(destination, arm, seed)
    contract = {
        "format": "coffee_detector.dlrbc_fresh.arm_contract.v1",
        "protocol": "faruq-v3-dlrbc-fresh-seed42-v1",
        "arm": arm,
        "seed": seed,
        "official_pretrained_sha256": checkpoint_sha,
        "config_sha256": _sha256(CONFIGS[arm]),
        "initial_state_sha256": static["arm_initial_state_sha256"][arm],
        "train": train_args,
        "fresh_optimizer": True,
        "coffee_parent_checkpoint": None,
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        previous = _load_json(contract_path, "Run contract")
        if previous != contract:
            raise RuntimeError(f"Run directory memiliki kontrak berbeda: {run_dir}")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    if result_path.is_file():
        previous = _load_json(result_path, "Arm result")
        if previous.get("run_contract") != contract:
            raise RuntimeError("Result lama memiliki kontrak berbeda")
        if previous.get("test_images_accessed") is not False:
            raise RuntimeError("Result lama melanggar test lock")
        print(f"REUSE COMPLETE {arm} seed {seed}", flush=True)
        return previous

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    trainer = make_fresh_trainer(
        arm=arm,
        model_yaml=MODEL_YAML,
        pretrained_checkpoint=checkpoint,
        seed=seed,
        expected_initial_fingerprint=static["arm_initial_state_sha256"][arm],
        build_fresh=build_fresh_detector,
        config=dlrbc_config,
    )
    training_executed = False
    if not _run_complete(run_dir, max_epochs):
        from ultralytics import YOLO

        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            destination, lock_name=f"{arm}_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME SAME RUN {arm} seed {seed} dari epoch {epoch + 1}", flush=True)
                model = YOLO(str(last))
                args = {"resume": True, "device": device}
            else:
                if last.is_file() and not resumable:
                    raise RuntimeError(
                        f"{arm} memiliki last.pt non-resumable tetapi belum complete"
                    )
                print(
                    f"START FRESH {arm} seed {seed} dari official yolo26n.pt",
                    flush=True,
                )
                model = YOLO(str(MODEL_YAML))
                args = dict(train_args)
                args.update(
                    data=str(root / "data.yaml"),
                    project=str(destination / arm),
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
        raise RuntimeError(f"Training {arm} belum complete/early-stopped valid: {run_dir}")
    if not best.is_file():
        raise FileNotFoundError(best)
    reports = destination / "val_reports"
    evaluation = evaluate(
        best, root, reports / f"{arm}_seed{seed}_val.json", split="val", device=device
    )
    if evaluation["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    diagnostic = run_faruq_v3_diagnostics(
        best,
        root,
        reports / f"{arm}_seed{seed}_diagnostic.json",
        split="val",
        device=device,
    )
    result = {
        "format": "coffee_detector.dlrbc_fresh.arm_result.v1",
        "protocol": "faruq-v3-dlrbc-fresh-seed42-v1",
        "arm": arm,
        "seed": seed,
        "metrics": {name: float(evaluation["metrics"][name]) for name in METRICS},
        "diagnostic": {
            "raw_top500_proposal_accessibility": float(
                diagnostic["raw_candidate_sensitivity"]["500"]["proposal_accessibility"]
            ),
            "conditional_top1_accuracy": float(
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
        "fresh_optimizer": True,
        "coffee_parent_checkpoint": None,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "run_contract": contract,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("DONE:", arm, seed, result["metrics"], result["diagnostic"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one parallel-safe fresh DLRBC seed-42 arm"
    )
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_faruq_v3_dlrbc_fresh_arm(
        args.arm,
        args.data_root,
        args.grouped_summary,
        args.pretrained_checkpoint,
        args.static_audit,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
