"""Fresh matched J25 screen isolating RGB-channel AF2 from luminance AF2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from coffee_detector.af2_luminance import (
    AF2LuminanceDetectionModel,
    AF2LuminanceInputEnhancer,
    load_af2_luminance_weights,
    rec709_luminance,
)
from coffee_detector.afab import AFABConfig, AFABInputEnhancer, minmax_spatial
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    AF2_CONFIG_V2,
    METRICS,
    MODEL_YAML,
    NATIVE_CONFIG_V2,
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
CONFIG = REPO_ROOT / "configs/coffee_standard_j25/AF2LUMDIRECT_TRAIN_SIBLINGS.yaml"
ARM = "AF2LUMDIRECT"
PROTOCOL = "coffee-standard-j25-af2-luminance-isolation-seed42-v1"
REFERENCE_PROTOCOL = "coffee-standard-j25-train-siblings-af2-direct-seed42-v2"
NC = 25


def _build_detector(pretrained_checkpoint: Path, af2: AFABConfig, seed: int):
    from ultralytics import YOLO

    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("Official pretrained source bukan nc=80")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AF2LuminanceDetectionModel(
            str(MODEL_YAML), ch=3, nc=NC, verbose=False, afab=af2
        )
        transfer = load_af2_luminance_weights(model, source)
    return model, transfer


def _isoluminant_pair() -> tuple[torch.Tensor, torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij"
    )
    texture = 0.04 * torch.sin(12.0 * xx + 7.0 * yy)
    base = torch.stack(
        (0.48 + texture, 0.52 - texture / 2.0, 0.44 + texture / 3.0), dim=0
    ).unsqueeze(0)
    delta = 0.05 * torch.sin(9.0 * xx - 5.0 * yy)
    changed = base.clone()
    changed[:, 0] += delta
    changed[:, 1] -= delta * (0.2126 / 0.7152)
    return base, changed


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen pertama dikunci seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    native_cfg, rgb_cfg, lum_cfg = (
        _yaml(NATIVE_CONFIG_V2),
        _yaml(AF2_CONFIG_V2),
        _yaml(CONFIG),
    )
    if not (
        native_cfg["model"] == rgb_cfg["model"] == lum_cfg["model"]
        and native_cfg["train"] == rgb_cfg["train"] == lum_cfg["train"]
    ):
        raise RuntimeError("D0/AF2-RGB/AF2-LUM tidak matched")
    if rgb_cfg.get("afab") != EXPECTED_AF2 or lum_cfg.get("afab") != EXPECTED_AF2:
        raise RuntimeError("Operator AF2 berubah")
    if lum_cfg.get("frontend") != "luminance_shared":
        raise RuntimeError("Frontend luminance tidak dibekukan")
    af2 = AFABConfig.from_mapping(lum_cfg["afab"])
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

    first, second = _isoluminant_pair()
    luminance = AF2LuminanceInputEnhancer(af2)
    legacy = AFABInputEnhancer(af2)
    with torch.inference_mode():
        luminance_difference = float(
            (rec709_luminance(first) - rec709_luminance(second)).abs().max()
        )
        gate_first = luminance.shared_gate(first)
        gate_second = luminance.shared_gate(second)
        shared_gate_difference = float((gate_first - gate_second).abs().max())
        rgb_gate_first = minmax_spatial(legacy.recover(first), eps=af2.eps)
        rgb_gate_second = minmax_spatial(legacy.recover(second), eps=af2.eps)
        rgb_gate_difference = float((rgb_gate_first - rgb_gate_second).abs().max())
        enhanced_probe = luminance(first)
    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "same_model_yaml": native_cfg["model"] == lum_cfg["model"],
        "same_50_epoch_schedule": native_cfg["train"] == lum_cfg["train"]
        and native_cfg["train"]["epochs"] == 50,
        "same_legacy_af2_hyperparameters": rgb_cfg["afab"] == lum_cfg["afab"],
        "detector_state_keys_exact": same_keys,
        "detector_state_tensors_exact": bool(same_tensors),
        "detector_parameter_count_exact": _parameter_count(native)
        == _parameter_count(candidate),
        "frontend_trainable_parameters_zero": _parameter_count(luminance) == 0,
        "isoluminant_probe_exact": luminance_difference <= 1e-6,
        "luminance_gate_invariant_to_isoluminant_chroma": shared_gate_difference <= 1e-4,
        "legacy_rgb_gate_sensitive_to_isoluminant_chroma": rgb_gate_difference > 1e-3,
        "frontend_output_finite": bool(torch.isfinite(enhanced_probe).all()),
        "frontend_output_active": float((enhanced_probe - first).abs().max()) > 0,
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.coffee_standard_j25.af2_luminance.static.v1",
        "protocol": PROTOCOL,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "common_initialized_detector_state_sha256": _state_fingerprint(native),
        "config_sha256": _sha256(CONFIG),
        "reference_native_config_sha256": _sha256(NATIVE_CONFIG_V2),
        "reference_rgb_config_sha256": _sha256(AF2_CONFIG_V2),
        "native_parameters": _parameter_count(native),
        "candidate_parameters": _parameter_count(candidate),
        "weight_transfer": transfer,
        "probe_differences": {
            "luminance": luminance_difference,
            "shared_luminance_gate": shared_gate_difference,
            "legacy_rgb_gate": rgb_gate_difference,
        },
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


def _trainer(*, pretrained: Path, af2: AFABConfig, seed: int, fingerprint: str):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class J25AF2LuminanceTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if bool(getattr(self.args, "resume", False)):
                model = AF2LuminanceDetectionModel(
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

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    return J25AF2LuminanceTrainer


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
    if dataset["data_format"] != "coffee_detector.coffee_standard_j25_source_split.train_siblings.v2":
        raise RuntimeError("AF2-LUM hanya untuk J25 train-siblings v2")
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static = run_static_preflight(checkpoint, destination / "static_preflight.json", seed=seed)
    config = _yaml(CONFIG)
    af2 = AFABConfig.from_mapping(config["afab"])
    train_args = dict(config["train"])
    run_dir = destination / ARM / f"{ARM}_seed{seed}"
    result_path = destination / "val_reports" / f"{ARM}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.coffee_standard_j25.af2_luminance.arm_contract.v1",
        "protocol": PROTOCOL,
        "reference_protocol": REFERENCE_PROTOCOL,
        "arm": ARM,
        "seed": seed,
        "source_archive_sha256": dataset["source_archive_sha256"],
        "development_contract_sha256": dataset["development_contract_sha256"],
        "provenance_summary_sha256": dataset["provenance_summary_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "common_initialized_detector_state_sha256": static["common_initialized_detector_state_sha256"],
        "config_sha256": static["config_sha256"],
        "train": train_args,
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _json(result_path, "AF2LUM result")
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
                    pretrained=checkpoint, af2=af2, seed=seed,
                    fingerprint=static["common_initialized_detector_state_sha256"],
                ),
                **args,
            )
        training_executed = True
    if not _run_complete(run_dir, int(train_args["epochs"])) or not best.is_file():
        raise RuntimeError("Run belum selesai secara valid")
    evaluation = evaluate(
        best, root, destination / "val_reports" / f"{ARM}_seed{seed}_val.json",
        split="val", device=device,
    )
    if evaluation["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.coffee_standard_j25.af2_luminance.arm_result.v1",
        "protocol": PROTOCOL,
        "reference_protocol": REFERENCE_PROTOCOL,
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


def build_decision(
    reference_root: str | Path,
    candidate_root: str | Path,
    output: str | Path,
) -> dict:
    reference_root = Path(reference_root).expanduser().resolve()
    candidate_root = Path(candidate_root).expanduser().resolve()
    rows = {
        "D0DIRECT": _json(reference_root / "val_reports/D0DIRECT_seed42_result.json", "D0"),
        "AF2DIRECT": _json(reference_root / "val_reports/AF2DIRECT_seed42_result.json", "AF2 RGB"),
        ARM: _json(candidate_root / f"val_reports/{ARM}_seed42_result.json", "AF2 LUM"),
    }
    if any(row.get("test_images_accessed") is not False for row in rows.values()):
        raise RuntimeError("Test lock gagal")
    if any(rows[name].get("protocol") != REFERENCE_PROTOCOL for name in ("D0DIRECT", "AF2DIRECT")):
        raise RuntimeError("Reference protocol tidak matched")
    if rows[ARM].get("reference_protocol") != REFERENCE_PROTOCOL:
        raise RuntimeError("Candidate reference protocol salah")
    values = {name: row["metrics"] for name, row in rows.items()}
    lum_minus_rgb = {
        metric: values[ARM][metric] - values["AF2DIRECT"][metric] for metric in METRICS
    }
    lum_minus_native = {
        metric: values[ARM][metric] - values["D0DIRECT"][metric] for metric in METRICS
    }
    overall = (
        lum_minus_rgb["macro_map50_95"] >= 0.002
        and lum_minus_rgb["bottom3_class_map50_95"] >= 0
        and lum_minus_rgb["worst_class_map50_95"] >= -0.01
    )
    tail = (
        lum_minus_rgb["macro_map50_95"] >= -0.002
        and lum_minus_rgb["bottom3_class_map50_95"] >= 0.01
        and lum_minus_rgb["worst_class_map50_95"] >= 0.01
    )
    rgb_better = (
        lum_minus_rgb["macro_map50_95"] <= -0.002
        and lum_minus_rgb["bottom3_class_map50_95"] <= 0
    )
    if overall or tail:
        interpretation = "SUPPORTS_RGB_CHANNEL_INTERFERENCE"
    elif rgb_better:
        interpretation = "RGB_CHANNEL_SPECTRA_BENEFICIAL"
    else:
        interpretation = "CHANNEL_COLOR_CAUSE_NOT_ISOLATED"
    payload = {
        "format": "coffee_detector.coffee_standard_j25.af2_luminance.seed42_decision.v1",
        "protocol": PROTOCOL,
        "values": values,
        "luminance_minus_rgb": lum_minus_rgb,
        "luminance_minus_native": lum_minus_native,
        "criteria": {
            "luminance_overall_route": overall,
            "luminance_lower_tail_route": tail,
            "rgb_better_route": rgb_better,
        },
        "interpretation": interpretation,
        "next": "STOP_AFTER_DIAGNOSTIC_SEED42",
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
    parser.add_argument("--reference-root")
    parser.add_argument("--decision-output")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    if args.decision_output:
        if not args.reference_root or not args.output_root:
            parser.error("Decision mode requires reference-root and output-root")
        build_decision(args.reference_root, args.output_root, args.decision_output)
    else:
        required = (
            args.data_root, args.development_contract, args.provenance_summary,
            args.pretrained_checkpoint, args.output_root,
        )
        if any(value is None for value in required):
            parser.error("Training mode requires data, contracts, pretrained, and output")
        run_arm(
            args.data_root, args.development_contract, args.provenance_summary,
            args.pretrained_checkpoint, args.output_root, seed=args.seed,
            device=args.device, authorize_training=args.authorize_training,
        )


if __name__ == "__main__":
    main()
