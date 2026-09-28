"""Matched fresh D0DIRECT/DCWCF1 arms on grouped DefectosCafeVerde."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from coffee_detector.defectos_cwcf import (
    DEFECTOS_ATTRIBUTE_NAMES,
    build_defectos_attribute_matrix,
    build_defectos_cwcf_model,
    load_defectos_cwcf_weights,
)
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_defectoscafeverde_af2_direct import (
    METRICS,
    MODEL_YAML,
    NC,
    NATIVE_CONFIG,
    SEED,
    _completed_epochs,
    _load_json,
    _load_yaml,
    screen_decision,
    validate_development_dataset,
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
from coffee_detector.j25_cwcf import CWCFConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_CONFIG = REPO_ROOT / "configs/defectoscafeverde/DCWCF1.yaml"
ARMS = ("D0DIRECT", "DCWCF1")
PROTOCOL = "defectoscafeverde-grouped-dcwcf-direct-seed42-v1"


def _repair_completed_resume_boundary_csv(run_dir: Path) -> dict:
    """Repair only adjacent duplicate epochs after a fully stripped run.

    An interrupted Drive-backed resume can persist the CSV row immediately
    before the matching checkpoint. Resuming then writes that boundary epoch
    once more. This is distinct from interleaved writers: after collapsing
    adjacent equal epochs, the sequence must be gap-free and monotonic.
    """

    csv_path = run_dir / "results.csv"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    if not csv_path.is_file() or not best.is_file() or not last.is_file():
        return {"status": "not_applicable"}
    checkpoint_epoch, resumable = _checkpoint_state(last)
    if checkpoint_epoch != -1 or resumable:
        return {"status": "not_completed_stripped"}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    sequence = [int(float(row["epoch"])) for row in rows]
    if not sequence:
        return {"status": "empty"}
    expected = list(range(sequence[0], sequence[0] + len(sequence)))
    if sequence == expected:
        return {"status": "clean", "epochs": len(sequence)}

    repaired: list[dict] = []
    duplicate_epochs: list[int] = []
    for row, epoch in zip(rows, sequence):
        if not repaired:
            repaired.append(row)
            continue
        previous = int(float(repaired[-1]["epoch"]))
        if epoch == previous:
            repaired[-1] = row
            duplicate_epochs.append(epoch)
        elif epoch == previous + 1:
            repaired.append(row)
        else:
            raise RuntimeError(
                "results.csv bukan duplikat batas resume yang aman: "
                f"{sequence}"
            )
    repaired_sequence = [int(float(row["epoch"])) for row in repaired]
    repaired_expected = list(
        range(repaired_sequence[0], repaired_sequence[0] + len(repaired_sequence))
    )
    if not duplicate_epochs or repaired_sequence != repaired_expected:
        raise RuntimeError("results.csv gagal memenuhi kontrak repair batas resume")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = run_dir / f"results.raw-resume-boundary-{stamp}.csv"
    shutil.copy2(csv_path, backup)
    temporary = run_dir / "results.csv.repairing"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repaired)
    temporary.replace(csv_path)
    report = {
        "format": "coffee_detector.results_csv.resume_boundary_repair.v1",
        "status": "repaired_adjacent_duplicate_epochs",
        "raw_backup": str(backup),
        "raw_rows": len(rows),
        "repaired_rows": len(repaired),
        "duplicate_epochs": duplicate_epochs,
        "repaired_sequence_start": repaired_sequence[0],
        "repaired_sequence_end": repaired_sequence[-1],
        "best_checkpoint_sha256": _sha256(best),
        "last_checkpoint_sha256": _sha256(last),
        "test_images_accessed": False,
    }
    (run_dir / "results_csv_repair.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _build_native(checkpoint: Path, seed: int, *, verbose: bool = False):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("Official pretrained source harus memiliki 80 kelas")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DetectionModel(
            str(MODEL_YAML), ch=3, nc=NC, verbose=verbose
        )
        model.load(source)
    return model, source


def _build_candidate(checkpoint: Path, seed: int, config: CWCFConfig, *, verbose=False):
    from ultralytics import YOLO

    source = YOLO(str(checkpoint)).model
    if _source_class_count(source) != 80:
        raise RuntimeError("Official pretrained source harus memiliki 80 kelas")
    return build_defectos_cwcf_model(
        str(MODEL_YAML), nc=NC, source=source, seed=seed,
        config=config, verbose=verbose,
    )


def _raw(model: torch.nn.Module, probe: torch.Tensor) -> dict:
    model.train()
    with torch.no_grad():
        value = model(probe)
    return value["one2many"] if "one2many" in value else value


def run_static_preflight(
    pretrained_checkpoint: str | Path,
    output: str | Path,
    *,
    seed: int = SEED,
) -> dict:
    if seed != SEED:
        raise ValueError("Screen pertama dikunci seed 42")
    checkpoint, pretrained_sha = _require_official_pretrained(pretrained_checkpoint)
    native_yaml, candidate_yaml = _load_yaml(NATIVE_CONFIG), _load_yaml(CANDIDATE_CONFIG)
    if native_yaml["model"] != candidate_yaml["model"]:
        raise RuntimeError("Model YAML D0DIRECT dan DCWCF1 berbeda")
    if native_yaml["train"] != candidate_yaml["train"]:
        raise RuntimeError("Jadwal D0DIRECT dan DCWCF1 berbeda")
    config = CWCFConfig.from_mapping(candidate_yaml["cwcf"])
    native, _ = _build_native(checkpoint, seed)
    candidate = _build_candidate(checkpoint, seed, config)
    native_head = native.model[-1]
    candidate_head = candidate.model[-1]
    common_backbone = all(
        torch.equal(left.detach().cpu(), right.detach().cpu())
        for native_module, candidate_module in zip(
            native.model[:-1], candidate.model[:-1]
        )
        for left, right in zip(
            native_module.state_dict().values(), candidate_module.state_dict().values()
        )
    )
    head_keys_equal = tuple(native_head.state_dict()) == tuple(
        candidate_head.base_head.state_dict()
    )
    head_tensors_equal = head_keys_equal and all(
        torch.equal(native_head.state_dict()[key], candidate_head.base_head.state_dict()[key])
        for key in native_head.state_dict()
    )
    probe = torch.linspace(0.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)
    native_raw, candidate_raw = _raw(native, probe), _raw(candidate, probe)
    initial_boxes_exact = torch.equal(native_raw["boxes"], candidate_raw["boxes"])
    initial_scores_exact = torch.equal(native_raw["scores"], candidate_raw["scores"])
    with torch.no_grad():
        candidate_head.adapters[0].affine.weight[0, 0, 0, 0] = 0.25
    active_raw = _raw(candidate, probe)
    active_boxes_exact = torch.equal(native_raw["boxes"], active_raw["boxes"])
    active_scores_change = not torch.equal(native_raw["scores"], active_raw["scores"])

    matrix = build_defectos_attribute_matrix()
    logits = torch.randn(NC, len(DEFECTOS_ATTRIBUTE_NAMES), requires_grad=True)
    auxiliary = F.binary_cross_entropy_with_logits(logits, matrix)
    auxiliary.backward()
    gates = {
        "official_pretrained_sha256_exact": pretrained_sha == OFFICIAL_YOLO26N_SHA256,
        "target_class_count_12": getattr(native_head, "nc", None) == NC,
        "same_model_yaml": native_yaml["model"] == candidate_yaml["model"],
        "same_50_epoch_schedule": (
            native_yaml["train"] == candidate_yaml["train"]
            and native_yaml["train"]["epochs"] == 50
        ),
        "common_backbone_tensors_exact": bool(common_backbone),
        "native_head_schema_exact": bool(head_keys_equal),
        "native_head_tensors_exact": bool(head_tensors_equal),
        "initial_boxes_bitwise_exact": bool(initial_boxes_exact),
        "initial_scores_bitwise_exact": bool(initial_scores_exact),
        "active_conditioning_preserves_boxes": bool(active_boxes_exact),
        "active_conditioning_changes_scores": bool(active_scores_change),
        "attribute_matrix_shape_12_by_15": tuple(matrix.shape) == (12, 15),
        "attribute_codes_unique": int(torch.unique(matrix, dim=0).shape[0]) == NC,
        "paper_confusion_families_exact": bool(
            matrix[[0, 5, 7], 0].eq(1).all()
            and matrix[[1, 6], 1].eq(1).all()
            and matrix[[2, 8, 9], 2].eq(1).all()
        ),
        "attribute_loss_finite": bool(torch.isfinite(auxiliary)),
        "attribute_gradients_finite_nonzero": bool(
            logits.grad is not None
            and torch.isfinite(logits.grad).all()
            and logits.grad.abs().sum() > 0
        ),
        "test_not_accessed": True,
    }
    # Raw train-mode probes update BatchNorm buffers even under no_grad.  Rebuild
    # both endpoints so the fingerprints describe the pristine trainer state,
    # not the diagnostic-forward state.
    native, _ = _build_native(checkpoint, seed)
    candidate = _build_candidate(checkpoint, seed, config)
    payload = {
        "format": "coffee_detector.defectoscafeverde.dcwcf.static.v1",
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
        "cwcf": config.to_dict(),
        "attribute_names": list(DEFECTOS_ATTRIBUTE_NAMES),
        "attribute_matrix": matrix.int().tolist(),
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
    *,
    arm: str,
    checkpoint: Path,
    seed: int,
    config: CWCFConfig,
    expected_fingerprint: str,
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class DefectosCWCFTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if bool(getattr(self.args, "resume", False)):
                if arm == "DCWCF1":
                    model = build_defectos_cwcf_model(
                        str(MODEL_YAML), nc=self.data["nc"], source=None,
                        seed=seed, config=config, verbose=verbose,
                    )
                    if weights:
                        load_defectos_cwcf_weights(model, weights)
                else:
                    from ultralytics.nn.tasks import DetectionModel
                    model = DetectionModel(
                        str(MODEL_YAML), ch=self.data["channels"],
                        nc=self.data["nc"], verbose=verbose,
                    )
                    if weights:
                        model.load(weights)
                return self.set_model_names_for_load(model)
            model = (
                _build_candidate(checkpoint, seed, config, verbose=verbose)
                if arm == "DCWCF1"
                else _build_native(checkpoint, seed, verbose=verbose)[0]
            )
            if _state_fingerprint(model) != expected_fingerprint:
                raise RuntimeError("Initial state tidak cocok dengan static preflight")
            return self.set_model_names_for_load(model)

        def final_eval(self):
            from ultralytics.utils.torch_utils import strip_optimizer

            last = strip_optimizer(self.last) if self.last.exists() else {}
            if self.best.exists():
                strip_optimizer(
                    self.best, updates={"train_results": last.get("train_results")}
                )

    return DefectosCWCFTrainer


def run_arm(
    arm: str,
    data_root: str | Path,
    grouped_audit: str | Path,
    pretrained_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = SEED,
    device: str = "0",
    authorize_training: bool = False,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Arm harus salah satu {ARMS}")
    if seed != SEED or not authorize_training:
        raise RuntimeError("Seed-42 dan --authorize-training wajib")
    root = Path(data_root).expanduser().resolve()
    dataset = validate_development_dataset(root, grouped_audit)
    checkpoint, _ = _require_official_pretrained(pretrained_checkpoint)
    destination = Path(output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    static_path = destination / f"static_preflight_{arm}.json"
    static = run_static_preflight(checkpoint, static_path, seed=seed)
    static["dataset_audit_sha256"] = _sha256(grouped_audit)
    static["dataset_contract"] = dataset
    static_path.write_text(json.dumps(static, indent=2) + "\n", encoding="utf-8")
    train_args = dict(_load_yaml(NATIVE_CONFIG)["train"])
    config = CWCFConfig.from_mapping(_load_yaml(CANDIDATE_CONFIG)["cwcf"])
    run_dir = destination / arm / f"{arm}_seed{seed}"
    result_path = destination / "val_reports" / f"{arm}_seed{seed}_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    expected_fingerprint = static[
        "candidate_initial_state_sha256" if arm == "DCWCF1" else "native_initial_state_sha256"
    ]
    contract = {
        "format": "coffee_detector.defectoscafeverde.dcwcf.arm_contract.v1",
        "protocol": PROTOCOL,
        "arm": arm,
        "seed": seed,
        "dataset_audit_sha256": static["dataset_audit_sha256"],
        "pretrained_checkpoint_sha256": static["pretrained_checkpoint_sha256"],
        "initial_state_sha256": expected_fingerprint,
        "native_config_sha256": static["native_config_sha256"],
        "candidate_config_sha256": static["candidate_config_sha256"],
        "train": train_args,
        "test_images_accessed": False,
    }
    if result_path.is_file():
        old = _load_json(result_path, f"{arm} result")
        if old.get("run_contract") != contract:
            raise RuntimeError("Existing result berbeda kontrak")
        return old
    run_dir.mkdir(parents=True, exist_ok=True)
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file() and _load_json(contract_path, "Run contract") != contract:
        results_csv = run_dir / "results.csv"
        completed = _completed_epochs(results_csv) if results_csv.is_file() else 0
        learned_artifacts = (
            run_dir / "weights/best.pt",
            run_dir / "weights/last.pt",
        )
        if completed > 0 or any(path.is_file() for path in learned_artifacts):
            raise RuntimeError("Run directory berbeda kontrak")
        # A fail-fast setup can leave only the old contract.  It contains no
        # learned state and is safe to replace after a protocol-code repair.
        contract_path.unlink()
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    training_executed = False
    csv_repair = _repair_completed_resume_boundary_csv(run_dir)
    if not _run_complete(run_dir, int(train_args["epochs"])):
        from ultralytics import YOLO

        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            destination, lock_name=f"{arm}_seed{seed}.training.lock"
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                model, args = YOLO(str(last)), {"resume": True, "device": device}
            else:
                if last.is_file():
                    raise RuntimeError("last.pt tidak resumable")
                model, args = YOLO(str(MODEL_YAML)), dict(train_args)
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
            model.train(
                trainer=_trainer(
                    arm=arm, checkpoint=checkpoint, seed=seed, config=config,
                    expected_fingerprint=expected_fingerprint,
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
        raise RuntimeError("Run belum selesai secara valid")
    evaluation = evaluate(
        best,
        root,
        destination / "val_reports" / f"{arm}_seed{seed}_val.json",
        split="val",
        device=device,
    )
    metrics = evaluation["metrics"]
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    result = {
        "format": "coffee_detector.defectoscafeverde.dcwcf.arm_result.v1",
        "protocol": PROTOCOL,
        "arm": arm,
        "seed": seed,
        "metrics": {metric: float(metrics[metric]) for metric in METRICS},
        "map50_95_by_class": metrics["map50_95_by_class"],
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
    native_result: str | Path,
    candidate_result: str | Path,
    output: str | Path,
) -> dict:
    native = _load_json(native_result, "D0DIRECT result")
    candidate = _load_json(candidate_result, "DCWCF1 result")
    for payload, arm in ((native, "D0DIRECT"), (candidate, "DCWCF1")):
        if payload.get("protocol") != PROTOCOL or payload.get("arm") != arm:
            raise RuntimeError(f"Kontrak result salah: {arm}")
        if payload.get("test_images_accessed") is not False:
            raise RuntimeError("Test lock gagal")
    deltas = {
        metric: float(candidate["metrics"][metric]) - float(native["metrics"][metric])
        for metric in METRICS
    }
    screen = screen_decision(deltas)
    payload = {
        "format": "coffee_detector.defectoscafeverde.dcwcf.seed42_decision.v1",
        "protocol": PROTOCOL,
        "values": {"D0DIRECT": native["metrics"], "DCWCF1": candidate["metrics"]},
        "deltas": deltas,
        "screen": screen,
        "training_executed": True,
        "evaluation_split": "val",
        "test_opened": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-audit", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    run_arm(
        args.arm,
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
