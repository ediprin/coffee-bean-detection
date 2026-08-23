from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.af2_iso import frozen_arm_config
from coffee_detector.af2_orient_cmc0 import (
    make_af2_orient_cmc0_trainer,
    static_af2_orient_cmc0_audit,
)
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_af2_iso_arm import (
    _checkpoint_seed,
    _latency,
)
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.experiments.run_faruq_v3_stb_capacity_control import (
    _checkpoint_state,
    _exclusive_training_lock,
    _run_complete,
)
from coffee_detector.stb import STBConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/af2_orient_cmc0/AF2_ORIENT_CMC0_yolo26n.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
ARM = "AF2_ORIENT_CMC0"
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _metric_triplet(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def _load_af2_orient_parent(path: str | Path, d0_checkpoint: Path, seed: int) -> dict:
    payload = _load_json(path, "AF2_ORIENT parent result")
    if payload.get("arm") != "AF2_ORIENT":
        raise RuntimeError("Parent result bukan AF2_ORIENT")
    if int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"AF2_ORIENT parent bukan seed {seed}")
    if payload.get("evaluation_split") != "val":
        raise RuntimeError("AF2_ORIENT parent bukan validation result")
    if payload.get("test_images_accessed") is not False:
        raise RuntimeError("AF2_ORIENT parent melanggar test lock")
    if payload.get("initial_d0_checkpoint_sha256") != _sha256(d0_checkpoint):
        raise RuntimeError("AF2_ORIENT parent dan candidate tidak berasal dari D0 yang sama")
    if payload.get("operator") != frozen_arm_config("AF2_ORIENT").to_dict():
        raise RuntimeError("AF2_ORIENT parent operator drift")
    return payload


def _decision(candidate: dict, parent: dict) -> dict:
    delta = {name: candidate[name] - parent[name] for name in METRICS}
    superiority = {
        "macro_gain_at_least_0_20_point": delta["macro_map50_95"] >= 0.002,
        "bottom3_drop_no_more_than_0_50_point": delta["bottom3_class_map50_95"] >= -0.005,
        "worst_drop_no_more_than_1_00_point": delta["worst_class_map50_95"] >= -0.010,
    }
    tail_pareto = {
        "macro_drop_no_more_than_0_10_point": delta["macro_map50_95"] >= -0.001,
        "bottom3_gain_at_least_0_50_point": delta["bottom3_class_map50_95"] >= 0.005,
        "worst_gain_at_least_1_00_point": delta["worst_class_map50_95"] >= 0.010,
    }
    superiority_pass = all(superiority.values())
    tail_pareto_pass = all(tail_pareto.values())
    return {
        "deltas": delta,
        "superiority": superiority,
        "tail_pareto": tail_pareto,
        "decision": "PASS" if superiority_pass or tail_pareto_pass else "FAIL",
        "route": "SUPERIORITY" if superiority_pass else ("TAIL_PARETO" if tail_pareto_pass else "NONE"),
    }


def run_af2_orient_cmc0(
    data_root: str | Path,
    grouped_summary: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    af2_orient_result: str | Path | None = None,
    seed: int = 42,
    device: str = "0",
    authorize_training: bool = False,
    latency_iterations: int = 50,
) -> dict:
    if stage not in {"static", "train"}:
        raise ValueError("stage harus static atau train")
    if seed != 42:
        raise ValueError("Screening AF2_ORIENT+CMC0 dikunci seed 42")

    data_root = Path(data_root).expanduser().resolve()
    grouped_summary = Path(grouped_summary).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not d0_checkpoint.is_file():
        raise FileNotFoundError(d0_checkpoint)
    if _checkpoint_seed(d0_checkpoint) != seed:
        raise RuntimeError(f"Checkpoint D0 tidak cocok dengan seed {seed}")

    if stage == "static":
        return static_af2_orient_cmc0_audit(
            MODEL_YAML,
            d0_checkpoint,
            output_root / "static_audit.json",
            nc=21,
        )

    if not authorize_training:
        raise RuntimeError("Training belum diotorisasi; tambahkan --authorize-training")
    if af2_orient_result is None:
        raise RuntimeError("Training memerlukan --af2-orient-result sebagai matched parent")
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos split test")

    load_faruq_grouped_summary(grouped_summary, data_root)
    static = _load_json(output_root / "static_audit.json", "Static audit")
    if static.get("decision") != "PASS":
        raise RuntimeError("Static audit belum PASS")
    if static.get("test_images_accessed") is not False:
        raise RuntimeError("Static audit melanggar test lock")

    parent_payload = _load_af2_orient_parent(af2_orient_result, d0_checkpoint, seed)
    parent_metrics = _metric_triplet(parent_payload)

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = reports / "dataset_audit.json"
    audit = audit_dataset(data_root, dataset_audit, near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    if payload.get("code") != ARM:
        raise RuntimeError("Config code drift")
    if Path(payload.get("model", "")).as_posix() != "configs/coffee_fg/models/yolo26n-p3.yaml":
        raise RuntimeError("Model YAML drift")

    frozen_af2 = frozen_arm_config("AF2_ORIENT")
    raw_af2 = dict(payload.get("af2_iso") or {})
    if int(raw_af2.get("stride", frozen_af2.stride)) != frozen_af2.stride:
        raise RuntimeError("AF2_ORIENT stride drift")
    if int(raw_af2.get("radial_bands", frozen_af2.radial_bands)) != frozen_af2.radial_bands:
        raise RuntimeError("AF2_ORIENT radial_bands drift")
    raw_af2.pop("stride", None)
    raw_af2.pop("radial_bands", None)
    raw_af2.setdefault("arm", "AF2_ORIENT")
    raw_af2.setdefault("overlap", 0.50)
    raw_af2.setdefault("chunk_size", 128)
    raw_af2.setdefault("eps", 1.0e-8)
    raw_af2.setdefault("radial_boundaries", [])
    if raw_af2 != frozen_af2.to_dict():
        raise RuntimeError(f"AF2_ORIENT config drift: {raw_af2!r} != {frozen_af2.to_dict()!r}")

    frozen_cmc0 = STBConfig.from_mapping(payload.get("stb"))
    if frozen_cmc0 != STBConfig():
        raise RuntimeError("CMC0 config drift")

    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / f"{ARM}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"

    contract = {
        "format": "coffee_detector.af2_orient_cmc0.run_contract.v2",
        "arm": ARM,
        "seed": seed,
        "epochs": epochs,
        "config_sha256": _sha256(CONFIG),
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "af2_orient_parent_result_sha256": _sha256(af2_orient_result),
        "af2_operator": frozen_af2.to_dict(),
        "cmc0": frozen_cmc0.to_dict(),
        "evaluation_split": "val",
        "test_images_accessed": False,
    }
    contract_path = run_dir / "run_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(f"Run directory memiliki kontrak berbeda: {run_dir}")
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    training_executed = False
    if not _run_complete(run_dir, epochs):
        from ultralytics import YOLO

        trainer = make_af2_orient_cmc0_trainer(
            af2_config=frozen_af2,
            stb_config=frozen_cmc0,
            d0_checkpoint=d0_checkpoint,
        )
        epoch, resumable = _checkpoint_state(last)
        with _exclusive_training_lock(
            output_root,
            lock_name=f"{ARM}_seed{seed}.training.lock",
        ):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {ARM} dari epoch {epoch + 1}/{epochs}", flush=True)
                model = YOLO(str(last))
                args = {"resume": True, "device": device}
            else:
                print(f"START {ARM} seed {seed} dari matched D0 seed {seed}", flush=True)
                model = YOLO(str(REPO_ROOT / payload["model"]))
                args = dict(payload["train"])
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root),
                    name=f"{ARM}_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=False,
                    verbose=False,
                    device=device,
                )
            model.train(trainer=trainer, **args)
        training_executed = True

    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run belum lengkap: {run_dir}")

    report_path = reports / f"{ARM}_seed{seed}_val.json"
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    if len(report["metrics"].get("map50_95_by_class", {})) != 21:
        raise RuntimeError("Validation tidak memuat seluruh 21 kelas")

    candidate_metrics = _metric_triplet(report)
    screen = _decision(candidate_metrics, parent_metrics)
    latency = _latency(best, device, iterations=latency_iterations)

    result = {
        "format": "coffee_detector.af2_orient_cmc0.seed42_result.v2",
        "arm": ARM,
        "seed": seed,
        "candidate": candidate_metrics,
        "parent": {"AF2_ORIENT": parent_metrics},
        "comparison": screen,
        "metrics": report["metrics"],
        "latency": latency,
        "checkpoint": str(best),
        "checkpoint_sha256": _sha256(best),
        "initial_d0_checkpoint": str(d0_checkpoint),
        "initial_d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "af2_orient_parent_result": str(Path(af2_orient_result).expanduser().resolve()),
        "af2_orient_parent_result_sha256": _sha256(af2_orient_result),
        "af2_operator": frozen_af2.to_dict(),
        "cmc0": frozen_cmc0.to_dict(),
        "training_executed_this_call": training_executed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "decision": screen["decision"],
        "next_action": "AUTHORIZE_PAIRED_MULTI_SEED" if screen["decision"] == "PASS" else "STOP_NO_TUNING",
        "scientific_status": "exploratory composition screen; neither AF2_ORIENT nor the STB-vs-CMC0 causal claim is independently confirmed",
    }
    result_path = reports / f"{ARM}_seed{seed}_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 AF2_ORIENT + CMC0 seed42 screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--af2-orient-result")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("static", "train"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--latency-iterations", type=int, default=50)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_af2_orient_cmc0(
        args.data_root,
        args.grouped_summary,
        args.d0_checkpoint,
        args.output_root,
        stage=args.stage,
        af2_orient_result=args.af2_orient_result,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
        latency_iterations=args.latency_iterations,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
