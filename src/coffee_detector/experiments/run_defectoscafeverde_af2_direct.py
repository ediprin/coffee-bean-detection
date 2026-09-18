"""Matched seed-42 native-vs-AF2 screen on grouped DefectosCafeVerde."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from coffee_detector.afab.model import AFABDetectionModel, load_afab_weights
from coffee_detector.afab.operator import AFABConfig, AFABInputEnhancer
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
NATIVE_CONFIG = REPO_ROOT / "configs/defectoscafeverde/D0DIRECT.yaml"
AF2_CONFIG = REPO_ROOT / "configs/defectoscafeverde/AF2DIRECT.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
ARMS = ("D0DIRECT", "AF2DIRECT")
SEED = 42
NC = 12
EXPECTED_IMAGES = {"train": 2827, "val": 808}
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


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


def _build_detector(
    *,
    use_af2: bool,
    pretrained_checkpoint: Path,
    af2_config: AFABConfig,
    seed: int,
    verbose: bool = False,
) -> tuple[torch.nn.Module, dict | None]:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("Official pretrained source harus memiliki 80 kelas")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if use_af2:
            model = AFABDetectionModel(
                str(MODEL_YAML), ch=3, nc=NC, verbose=verbose, afab=af2_config
            )
            transfer = load_afab_weights(model, source)
        else:
            model = DetectionModel(str(MODEL_YAML), ch=3, nc=NC, verbose=verbose)
            model.load(source)
            transfer = None
    return model, transfer


def validate_development_dataset(data_root: str | Path, grouped_audit: str | Path) -> dict:
    root = Path(data_root).expanduser().resolve()
    if (root / "test").exists():
        raise RuntimeError("TEST TEREXPOSE — development root hanya boleh train/val")
    data = _load_yaml(root / "data.yaml")
    if "test" in data:
        raise RuntimeError("data.yaml development tidak boleh memiliki test")
    names = data.get("names", {})
    if isinstance(names, dict):
        class_count = len(names)
    elif isinstance(names, list):
        class_count = len(names)
    else:
        class_count = 0
    counts = {}
    for split, expected in EXPECTED_IMAGES.items():
        images = [
            path
            for path in (root / split / "images").glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        labels = list((root / split / "labels").glob("*.txt"))
        if len(images) != expected or len(labels) != expected:
            raise RuntimeError(
                f"Kontrak {split} gagal: images={len(images)}, labels={len(labels)}, expected={expected}"
            )
        counts[split] = len(images)
    audit = _load_json(grouped_audit, "Grouped dataset audit")
    gates = {
        "grouped_audit_pass": audit.get("decision") == "PASS_GROUPED_DATASET_GATE",
        "grouped_audit_training_false": audit.get("training_authorized") is False,
        "grouped_audit_test_false": audit.get("test_accessed") is False,
        "development_root_has_no_test": not (root / "test").exists(),
        "development_yaml_has_no_test": "test" not in data,
        "exact_train_images": counts.get("train") == EXPECTED_IMAGES["train"],
        "exact_val_images": counts.get("val") == EXPECTED_IMAGES["val"],
        "exact_12_class_ontology": class_count == NC,
    }
    if not all(gates.values()):
        raise RuntimeError(f"Development dataset gate gagal: {[k for k,v in gates.items() if not v]}")
    return {"gates": gates, "images": counts, "classes": class_count}


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen pertama dikunci pada seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    native_cfg, af2_cfg = _load_yaml(NATIVE_CONFIG), _load_yaml(AF2_CONFIG)
    if native_cfg["train"] != af2_cfg["train"]:
        raise RuntimeError("Schedule native dan AF2 tidak identik")
    if native_cfg["model"] != af2_cfg["model"]:
        raise RuntimeError("Model YAML native dan AF2 tidak identik")
    if af2_cfg["afab"] != EXPECTED_AF2:
        raise RuntimeError("Konfigurasi AF2 berubah")
    af2 = AFABConfig.from_mapping(af2_cfg["afab"])
    native, _ = _build_detector(
        use_af2=False,
        pretrained_checkpoint=checkpoint,
        af2_config=af2,
        seed=seed,
    )
    candidate, transfer = _build_detector(
        use_af2=True,
        pretrained_checkpoint=checkpoint,
        af2_config=af2,
        seed=seed,
    )
    native_state, candidate_state = native.state_dict(), candidate.state_dict()
    same_keys = list(native_state) == list(candidate_state)
    same_tensors = same_keys and all(
        torch.equal(native_state[key].cpu(), candidate_state[key].cpu()) for key in native_state
    )
    frontend = AFABInputEnhancer(af2)
    probe = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    with torch.inference_mode():
        enhanced = frontend(probe)
    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "target_class_count_12": getattr(native.model[-1], "nc", None) == NC,
        "same_model_yaml": native_cfg["model"] == af2_cfg["model"],
        "same_50_epoch_schedule": native_cfg["train"] == af2_cfg["train"] and native_cfg["train"]["epochs"] == 50,
        "detector_state_keys_exact": same_keys,
        "detector_state_tensors_exact": bool(same_tensors),
        "detector_parameter_count_exact": _parameter_count(native) == _parameter_count(candidate),
        "af2_learned_parameters_zero": _parameter_count(frontend) == 0,
        "af2_probe_finite": bool(torch.isfinite(enhanced).all()),
        "af2_probe_live": float((enhanced - probe).abs().max()) > 0,
        "test_not_accessed": True,
    }
    payload = {
        "format": "coffee_detector.defectoscafeverde.af2_direct.static.v1",
        "protocol": "defectoscafeverde-grouped-af2-direct-seed42-v1",
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "seed": seed,
        "pretrained_checkpoint_sha256": pretrained_sha,
        "common_initialized_detector_state_sha256": _state_fingerprint(native),
        "native_config_sha256": _sha256(NATIVE_CONFIG),
        "af2_config_sha256": _sha256(AF2_CONFIG),
        "native_parameters": _parameter_count(native),
        "candidate_parameters": _parameter_count(candidate),
        "candidate_weight_transfer": transfer,
        "gates": gates,
        "training_authorized": True,
        "test_images_accessed": False,
    }
    payload["training_authorized"] = payload["decision"] == "PASS"
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["decision"] != "PASS":
        raise RuntimeError(f"Static preflight gagal: {gates}")
    return payload


def _make_trainer(
    *,
    use_af2: bool,
    af2_config: AFABConfig,
    pretrained_checkpoint: Path,
    seed: int,
    expected_fingerprint: str,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class DirectTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if bool(getattr(self.args, "resume", False)):
                if use_af2:
                    model = AFABDetectionModel(
                        str(MODEL_YAML), nc=self.data["nc"], ch=self.data["channels"],
                        verbose=verbose, afab=af2_config,
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
                use_af2=use_af2,
                pretrained_checkpoint=pretrained_checkpoint,
                af2_config=af2_config,
                seed=seed,
                verbose=verbose,
            )
            fingerprint = _state_fingerprint(model)
            if fingerprint != expected_fingerprint:
                raise RuntimeError("Initial detector state tidak cocok dengan static preflight")
            return self.set_model_names_for_load(model)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer
            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(self.best, updates={"train_results": last.get("train_results")})

    DirectTrainer.__name__ = "DefectosAF2Trainer" if use_af2 else "DefectosNativeTrainer"
    return DirectTrainer


def _completed_epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _run_arm(
    arm: str,
    data_root: Path,
    checkpoint: Path,
    static: dict,
    output_root: Path,
    *,
    seed: int,
    device: str,
) -> dict:
    use_af2 = arm == "AF2DIRECT"
    native_cfg, af2_cfg = _load_yaml(NATIVE_CONFIG), _load_yaml(AF2_CONFIG)
    af2 = AFABConfig.from_mapping(af2_cfg["afab"])
    train_args = dict(native_cfg["train"])
    run_dir = output_root / arm / f"{arm}_seed{seed}"
    result_path = output_root / "val_reports" / f"{arm}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.defectoscafeverde.af2_direct.arm_contract.v1",
        "arm": arm,
        "seed": seed,
        "dataset_audit_sha256": static["dataset_audit_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "common_initialized_detector_state_sha256": static["common_initialized_detector_state_sha256"],
        "train": train_args,
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _load_json(result_path, "Arm result")
        if old.get("run_contract") != contract:
            raise RuntimeError(f"Result lama berbeda kontrak: {result_path}")
        return old
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _load_json(contract_path, "Run contract") != contract:
        raise RuntimeError(f"Run directory berbeda kontrak: {run_dir}")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    max_epochs = int(train_args["epochs"])
    trainer = _make_trainer(
        use_af2=use_af2,
        af2_config=af2,
        pretrained_checkpoint=checkpoint,
        seed=seed,
        expected_fingerprint=static["common_initialized_detector_state_sha256"],
    )
    if not _run_complete(run_dir, max_epochs):
        from ultralytics import YOLO
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(output_root, lock_name=f"{arm}_seed{seed}.training.lock"):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                if last.is_file():
                    raise RuntimeError(f"last.pt {arm} tidak resumable")
                model = YOLO(str(MODEL_YAML))
                args = dict(train_args)
                args.update(
                    data=str(data_root / "data.yaml"), project=str(output_root / arm),
                    name=f"{arm}_seed{seed}", exist_ok=True, seed=seed,
                    deterministic=True, plots=False, verbose=False, device=device,
                )
            model.train(trainer=trainer, **args)
        training_executed = True
    if not _run_complete(run_dir, max_epochs) or not best.is_file():
        raise RuntimeError(f"Run belum complete: {run_dir}")
    evaluation = evaluate(
        best, data_root, output_root / "val_reports" / f"{arm}_seed{seed}_val.json",
        split="val", device=device,
    )
    if evaluation["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.defectoscafeverde.af2_direct.arm_result.v1",
        "arm": arm,
        "seed": seed,
        "metrics": {name: float(evaluation["metrics"][name]) for name in METRICS},
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
    return result


def screen_decision(deltas: dict[str, float]) -> dict:
    overall = (
        deltas["macro_map50_95"] >= 0.005
        and deltas["bottom3_class_map50_95"] >= 0.0
        and deltas["worst_class_map50_95"] >= -0.01
    )
    tail = (
        deltas["macro_map50_95"] >= -0.002
        and deltas["bottom3_class_map50_95"] >= 0.01
        and deltas["worst_class_map50_95"] >= 0.01
    )
    return {
        "overall_route": overall,
        "lower_tail_route": tail,
        "decision": "PROMOTE_TO_PAIRED_3_SEED" if overall or tail else "STOP_AFTER_SEED42",
    }


def run_screen(
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
    dataset_contract = validate_development_dataset(root, grouped_audit)
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static_path = destination / "static_preflight.json"
    static = run_static_preflight(checkpoint, static_path, seed=seed)
    static["dataset_audit_sha256"] = _sha256(grouped_audit)
    static["dataset_contract"] = dataset_contract
    static_path.write_text(json.dumps(static, indent=2) + "\n", encoding="utf-8")
    control = _run_arm("D0DIRECT", root, checkpoint, static, destination, seed=seed, device=device)
    candidate = _run_arm("AF2DIRECT", root, checkpoint, static, destination, seed=seed, device=device)
    deltas = {name: candidate["metrics"][name] - control["metrics"][name] for name in METRICS}
    payload = {
        "format": "coffee_detector.defectoscafeverde.af2_direct.seed42_decision.v1",
        "values": {"D0DIRECT": control["metrics"], "AF2DIRECT": candidate["metrics"]},
        "deltas": deltas,
        "screen": screen_decision(deltas),
        "training_executed": True,
        "evaluation_split": "val",
        "test_opened": False,
    }
    summary = destination / "defectoscafeverde_af2_direct_seed42_summary.json"
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
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
    run_screen(
        args.data_root, args.grouped_audit, args.pretrained_checkpoint, args.output_root,
        seed=args.seed, device=args.device, authorize_training=args.authorize_training,
    )


if __name__ == "__main__":
    main()
