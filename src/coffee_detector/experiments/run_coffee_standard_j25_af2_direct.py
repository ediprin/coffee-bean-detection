"""Matched native-vs-AF2 screen on the frozen J25 source-level split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from coffee_detector.afab.model import AFABDetectionModel, load_afab_weights
from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer
from coffee_detector.data.prepare_coffee_standard_j25_source_split import (
    FORMAT as DATA_FORMAT,
    TRAIN_SIBLINGS_FORMAT,
)
from coffee_detector.evaluate import evaluate
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
NATIVE_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/D0DIRECT.yaml"
AF2_CONFIG = REPO_ROOT / "configs/coffee_standard_j25/AF2DIRECT.yaml"
NATIVE_CONFIG_V2 = REPO_ROOT / "configs/coffee_standard_j25/D0DIRECT_TRAIN_SIBLINGS.yaml"
AF2_CONFIG_V2 = REPO_ROOT / "configs/coffee_standard_j25/AF2DIRECT_TRAIN_SIBLINGS.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
ARMS = ("D0DIRECT", "AF2DIRECT")
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")
SEED = 42
NC = 25


def _json(path: str | Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise RuntimeError(path)
    return value


def validate_j25_development(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
) -> dict:
    root = Path(data_root).expanduser().resolve()
    contract = _json(development_contract, "Development contract")
    provenance = _json(provenance_summary, "Thesis provenance")
    data = _yaml(root / "data.yaml")
    names = data.get("names", {})
    class_count = len(names) if isinstance(names, (dict, list)) else 0
    data_format = contract.get("format")
    expected_train_images = 695 if data_format == TRAIN_SIBLINGS_FORMAT else 315
    gates = {
        "development_contract_pass": data_format in {DATA_FORMAT, TRAIN_SIBLINGS_FORMAT}
        and contract.get("decision") == "PASS",
        "provenance_pass": provenance.get("decision")
        == "PASS_THESIS_LINEAGE_WITH_ONE_ANNOTATION_DISCREPANCY",
        "same_author_archive_sha256": contract.get("source_archive_sha256")
        == provenance.get("source_archive_sha256"),
        "exact_train_images": contract.get("images", {}).get("train") == expected_train_images,
        "exact_validation_images": contract.get("images", {}).get("val") == 68,
        "exact_25_class_ontology": class_count == NC,
        "test_directory_absent": not (root / "test").exists(),
        "test_yaml_key_absent": "test" not in data,
        "test_not_extracted": contract.get("test_images_extracted") is False,
        "provenance_model_test_false": provenance.get("test_images_accessed_by_model") is False,
    }
    if not all(gates.values()):
        raise RuntimeError(f"J25 development gate gagal: {[k for k,v in gates.items() if not v]}")
    return {
        "gates": gates,
        "source_archive_sha256": contract["source_archive_sha256"],
        "development_contract_sha256": _sha256(development_contract),
        "provenance_summary_sha256": _sha256(provenance_summary),
        "data_format": data_format,
    }


def _build_detector(
    *, use_af2: bool, pretrained_checkpoint: Path, af2: AFABConfig, seed: int
) -> tuple[torch.nn.Module, dict | None]:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("Official pretrained source bukan nc=80")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if use_af2:
            model = AFABDetectionModel(str(MODEL_YAML), ch=3, nc=NC, verbose=False, afab=af2)
            transfer = load_afab_weights(model, source)
        else:
            model = DetectionModel(str(MODEL_YAML), ch=3, nc=NC, verbose=False)
            model.load(source)
            transfer = None
    return model, transfer


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
    protocol: str = "coffee-standard-j25-source-split-af2-direct-seed42-v1",
    native_config: str | Path = NATIVE_CONFIG,
    af2_config: str | Path = AF2_CONFIG,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen pertama dikunci seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    native_config, af2_config = Path(native_config), Path(af2_config)
    native_cfg, af2_cfg = _yaml(native_config), _yaml(af2_config)
    if native_cfg["model"] != af2_cfg["model"] or native_cfg["train"] != af2_cfg["train"]:
        raise RuntimeError("Native/AF2 tidak matched")
    if af2_cfg.get("afab") != EXPECTED_AF2:
        raise RuntimeError("AF2 config berubah")
    af2 = AFABConfig.from_mapping(af2_cfg["afab"])
    native, _ = _build_detector(
        use_af2=False, pretrained_checkpoint=checkpoint, af2=af2, seed=seed
    )
    candidate, transfer = _build_detector(
        use_af2=True, pretrained_checkpoint=checkpoint, af2=af2, seed=seed
    )
    native_state, candidate_state = native.state_dict(), candidate.state_dict()
    same_keys = list(native_state) == list(candidate_state)
    same_tensors = same_keys and all(
        torch.equal(native_state[key].cpu(), candidate_state[key].cpu()) for key in native_state
    )
    probe = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    frontend = AFABInputEnhancer(af2)
    with torch.inference_mode():
        enhanced = frontend(probe)
    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "target_class_count_25": getattr(native.model[-1], "nc", None) == NC,
        "same_model_yaml": native_cfg["model"] == af2_cfg["model"],
        "same_50_epoch_schedule": native_cfg["train"] == af2_cfg["train"]
        and native_cfg["train"]["epochs"] == 50,
        "detector_state_keys_exact": same_keys,
        "detector_state_tensors_exact": bool(same_tensors),
        "detector_parameter_count_exact": _parameter_count(native) == _parameter_count(candidate),
        "af2_trainable_parameters_zero": _parameter_count(frontend) == 0,
        "af2_probe_finite": bool(torch.isfinite(enhanced).all()),
        "af2_probe_live": float((enhanced - probe).abs().max()) > 0,
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.coffee_standard_j25.af2_direct.static.v1",
        "protocol": protocol,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "common_initialized_detector_state_sha256": _state_fingerprint(native),
        "native_config_sha256": _sha256(native_config),
        "af2_config_sha256": _sha256(af2_config),
        "native_parameters": _parameter_count(native),
        "candidate_parameters": _parameter_count(candidate),
        "candidate_weight_transfer": transfer,
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
    *, use_af2: bool, pretrained: Path, af2: AFABConfig, seed: int, fingerprint: str
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class J25DirectTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if bool(getattr(self.args, "resume", False)):
                if use_af2:
                    model = AFABDetectionModel(
                        str(MODEL_YAML), nc=self.data["nc"], ch=self.data["channels"],
                        verbose=verbose, afab=af2,
                    )
                    if weights:
                        load_afab_weights(model, weights)
                else:
                    from ultralytics.nn.tasks import DetectionModel
                    model = DetectionModel(
                        str(MODEL_YAML), nc=self.data["nc"], ch=self.data["channels"], verbose=verbose
                    )
                    if weights:
                        model.load(weights)
                return self.set_model_names_for_load(model)
            model, _ = _build_detector(
                use_af2=use_af2, pretrained_checkpoint=pretrained, af2=af2, seed=seed
            )
            if _state_fingerprint(model) != fingerprint:
                raise RuntimeError("Initial state berbeda dari static preflight")
            return self.set_model_names_for_load(model)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    return J25DirectTrainer


def _epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def run_arm(
    arm: str,
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
    if arm not in ARMS or seed != SEED or not authorize_training:
        raise RuntimeError("Arm, seed-42, dan authorization wajib valid")
    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    protocol = (
        "coffee-standard-j25-train-siblings-af2-direct-seed42-v2"
        if dataset["data_format"] == TRAIN_SIBLINGS_FORMAT
        else "coffee-standard-j25-source-split-af2-direct-seed42-v1"
    )
    native_config, af2_config = (
        (NATIVE_CONFIG_V2, AF2_CONFIG_V2)
        if dataset["data_format"] == TRAIN_SIBLINGS_FORMAT
        else (NATIVE_CONFIG, AF2_CONFIG)
    )
    static_path = destination / f"static_preflight_{arm}.json"
    static = run_static_preflight(
        checkpoint,
        static_path,
        seed=seed,
        protocol=protocol,
        native_config=native_config,
        af2_config=af2_config,
    )
    use_af2 = arm == "AF2DIRECT"
    native_cfg, af2_cfg = _yaml(native_config), _yaml(af2_config)
    af2 = AFABConfig.from_mapping(af2_cfg["afab"])
    train_args = dict(native_cfg["train"])
    run_dir = destination / arm / f"{arm}_seed{seed}"
    result_path = destination / "val_reports" / f"{arm}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.coffee_standard_j25.af2_direct.arm_contract.v1",
        "protocol": static["protocol"],
        "arm": arm,
        "seed": seed,
        "source_archive_sha256": dataset["source_archive_sha256"],
        "development_contract_sha256": dataset["development_contract_sha256"],
        "provenance_summary_sha256": dataset["provenance_summary_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "common_initialized_detector_state_sha256": static["common_initialized_detector_state_sha256"],
        "native_config_sha256": static["native_config_sha256"],
        "af2_config_sha256": static["af2_config_sha256"],
        "train": train_args,
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _json(result_path, "Arm result")
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
        with _exclusive_training_lock(destination, lock_name=f"{arm}_seed{seed}.training.lock"):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                if last.is_file():
                    raise RuntimeError("last.pt tidak resumable")
                model, args = YOLO(str(MODEL_YAML)), dict(train_args)
                args.update(
                    data=str(root / "data.yaml"), project=str(destination / arm),
                    name=f"{arm}_seed{seed}", exist_ok=True, seed=seed,
                    deterministic=True, plots=False, verbose=False, device=device,
                )
            model.train(
                trainer=_trainer(
                    use_af2=use_af2, pretrained=checkpoint, af2=af2, seed=seed,
                    fingerprint=static["common_initialized_detector_state_sha256"],
                ),
                **args,
            )
        training_executed = True
    if not _run_complete(run_dir, int(train_args["epochs"])) or not best.is_file():
        raise RuntimeError("Run belum selesai secara valid")
    evaluation = evaluate(
        best, root, destination / "val_reports" / f"{arm}_seed{seed}_val.json",
        split="val", device=device,
    )
    if evaluation["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.coffee_standard_j25.af2_direct.arm_result.v1",
        "protocol": static["protocol"],
        "arm": arm,
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


def build_decision(output_root: str | Path, output: str | Path) -> dict:
    root = Path(output_root).expanduser().resolve()
    values = {
        arm: _json(root / "val_reports" / f"{arm}_seed42_result.json", arm)
        for arm in ARMS
    }
    if any(row.get("test_images_accessed") is not False for row in values.values()):
        raise RuntimeError("Test lock gagal")
    protocols = {row.get("protocol") for row in values.values()}
    if len(protocols) != 1:
        raise RuntimeError(f"Arm protocols differ: {sorted(protocols)}")
    deltas = {
        metric: values["AF2DIRECT"]["metrics"][metric] - values["D0DIRECT"]["metrics"][metric]
        for metric in METRICS
    }
    overall = deltas["macro_map50_95"] >= 0.005 and deltas["bottom3_class_map50_95"] >= 0 and deltas["worst_class_map50_95"] >= -0.01
    tail = deltas["macro_map50_95"] >= -0.002 and deltas["bottom3_class_map50_95"] >= 0.01 and deltas["worst_class_map50_95"] >= 0.01
    payload = {
        "format": "coffee_detector.coffee_standard_j25.af2_direct.seed42_decision.v1",
        "protocol": protocols.pop(),
        "values": {arm: row["metrics"] for arm, row in values.items()},
        "deltas": deltas,
        "criteria": {"overall_route": overall, "lower_tail_route": tail},
        "decision": "PROMOTE_TO_PAIRED_3_SEED" if overall or tail else "STOP_AFTER_SEED42",
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--data-root")
    parser.add_argument("--development-contract")
    parser.add_argument("--provenance-summary")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--decision-output")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    if args.decision_output:
        build_decision(args.output_root, args.decision_output)
    else:
        required = (args.arm, args.data_root, args.development_contract, args.provenance_summary, args.pretrained_checkpoint)
        if any(value is None for value in required):
            parser.error("Arm mode requires arm, data, contracts, provenance, and pretrained")
        run_arm(
            args.arm, args.data_root, args.development_contract, args.provenance_summary,
            args.pretrained_checkpoint, args.output_root, seed=args.seed,
            device=args.device, authorize_training=args.authorize_training,
        )


if __name__ == "__main__":
    main()
