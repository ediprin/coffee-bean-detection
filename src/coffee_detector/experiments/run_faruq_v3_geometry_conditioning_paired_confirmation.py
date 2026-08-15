"""Paired three-seed confirmation for GEO-C0 vs GEO1 on Faruq-v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.dataset import discover_layout
from coffee_detector.evaluate import evaluate
from coffee_detector.geometry_conditioning.audit import static_geometry_conditioning_audit
from coffee_detector.geometry_conditioning.model import GeometryConditioningConfig
from coffee_detector.geometry_conditioning.trainer import make_geometry_conditioning_trainer
from coffee_detector.experiments.run_faruq_v3_geometry_conditioning_screen import (
    METRICS,
    MODEL_YAML,
    CONFIG,
    _control_validity,
    _metrics,
    _size_mean,
)

SEED42 = 42
NEW_SEEDS = (123, 2026)
ALL_SEEDS = (SEED42, *NEW_SEEDS)
PROTOCOL = "faruq-v3-geometry-conditioning-paired-confirmation-v1"
SCREEN_PROTOCOL = "faruq-v3-geometry-conditioning-screening-v1"


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _epoch_sequence(path: Path) -> list[int]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [int(float(row["epoch"])) for row in rows]


def _epochs(path: Path) -> int:
    sequence = _epoch_sequence(path)
    if not sequence:
        return 0
    expected = list(range(sequence[0], sequence[0] + len(sequence)))
    if sequence != expected:
        raise RuntimeError(f"results.csv tidak monotonik: {sequence}")
    return sequence[-1] + 1


def _checkpoint_state(path: Path) -> tuple[int | None, bool]:
    if not path.is_file():
        return None, False
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None, False
    epoch = payload.get("epoch")
    return int(epoch) if epoch is not None else None, payload.get("optimizer") is not None


def _run_complete(run_dir: Path, epochs: int) -> bool:
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    if not best.is_file() or not last.is_file():
        return False
    epoch, resumable = _checkpoint_state(last)
    completed = _epochs(run_dir / "results.csv")
    if completed >= epochs:
        return (epoch == -1 and not resumable) or (epoch is not None and epoch + 1 >= epochs)
    return epoch == -1 and not resumable


@contextmanager
def _training_lock(output_root: Path, arm: str, seed: int, stale_seconds: int = 300):
    """Seed-specific Drive heartbeat lock; unlike the seed42 screening helper."""
    lock = output_root / f"{arm}_seed{seed}.training.lock"
    token = uuid.uuid4().hex
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, json.dumps({"token": token, "pid": os.getpid()}).encode())
            os.close(descriptor)
            break
        except FileExistsError:
            age = time.time() - lock.stat().st_mtime
            if age <= stale_seconds:
                raise RuntimeError(f"{arm} seed{seed} sedang ditulis runtime lain ({age:.0f}s)")
            lock.unlink(missing_ok=True)
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(30):
            try:
                lock.touch()
            except OSError:
                return

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=2)
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("token") == token:
            lock.unlink(missing_ok=True)


def _paired_artifacts(paired_root: Path, seed: int) -> dict[str, Path]:
    return {
        "d0_checkpoint": paired_root / "D0_base" / f"D0_seed{seed}" / "weights" / "best.pt",
        "d0ft_checkpoint": paired_root / "D0FT" / f"D0FT_seed{seed}" / "weights" / "best.pt",
        "d0ft_manifest": paired_root / "D0FT" / f"D0FT_seed{seed}" / "experiment_manifest.json",
        "d0ft_report": paired_root / "D0FT" / "val_reports" / f"D0FT_seed{seed}_val.json",
    }


def _validate_seed42_summary(path: str | Path) -> dict:
    payload = _load_json(path, "Seed42 GEO screening summary")
    if (
        payload.get("protocol") != SCREEN_PROTOCOL
        or int(payload.get("seed", -1)) != SEED42
        or payload.get("evaluation_split") != "val"
        or payload.get("test_images_accessed") is not False
        or payload.get("test_opened") is not False
        or payload.get("decision") != "RETAIN"
        or payload.get("geometry_retain_gate", {}).get("decision") != "PASS"
        or payload.get("control_validity", {}).get("decision") != "PASS"
    ):
        raise RuntimeError("Seed42 GEO summary bukan RETAIN validation-only yang kompatibel")
    for arm in ("D0FT", "GEO-C0", "GEO1"):
        row = payload.get("results", {}).get(arm)
        if not isinstance(row, dict):
            raise RuntimeError(f"Seed42 summary kehilangan arm {arm}")
        _metrics(row)
    return payload


def _validate_reused_pair(paired_root: Path, seed: int) -> dict:
    paths = _paired_artifacts(paired_root, seed)
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Artifact paired {label} seed{seed} tidak ditemukan: {path}")
    manifest = _load_json(paths["d0ft_manifest"], f"D0FT manifest seed{seed}")
    d0_hash = _sha256(paths["d0_checkpoint"])
    if manifest.get("weights_override_sha256") != d0_hash:
        raise RuntimeError(f"D0FT seed{seed} tidak berasal dari exact D0 checkpoint")
    report = _load_json(paths["d0ft_report"], f"D0FT val report seed{seed}")
    if report.get("split") != "val":
        raise RuntimeError(f"D0FT seed{seed} report bukan validation")
    _metrics(report)
    return {
        "d0_checkpoint": str(paths["d0_checkpoint"]),
        "d0_checkpoint_sha256": d0_hash,
        "d0ft_checkpoint": str(paths["d0ft_checkpoint"]),
        "d0ft_manifest": str(paths["d0ft_manifest"]),
        "d0ft_report": str(paths["d0ft_report"]),
    }


def _train_arm(
    *,
    arm: str,
    signal_mode: str,
    seed: int,
    d0_checkpoint: Path,
    data_root: Path,
    output_root: Path,
    config: GeometryConditioningConfig,
    train_args: dict,
    device: str | None,
) -> tuple[Path, bool]:
    from ultralytics import YOLO

    epochs = int(train_args["epochs"])
    run_dir = output_root / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    executed = False
    if not _run_complete(run_dir, epochs):
        trainer = make_geometry_conditioning_trainer(
            config, d0_checkpoint=d0_checkpoint, signal_mode=signal_mode
        )
        epoch, resumable = _checkpoint_state(last)
        with _training_lock(output_root, arm, seed):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {arm} seed{seed} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True}
            else:
                print(f"START {arm} seed{seed} dari exact paired D0", flush=True)
                model = YOLO(str(MODEL_YAML))
                args = dict(train_args)
                args.update(
                    data=str(data_root / "data.yaml"),
                    project=str(output_root),
                    name=f"{arm}_seed{seed}",
                    exist_ok=True,
                    seed=seed,
                    deterministic=True,
                    plots=True,
                    verbose=True,
                )
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        executed = True
    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run {arm} seed{seed} belum lengkap: {run_dir}")
    return best, executed


def _seed_record(seed: int, d0ft_report: dict, control_report: dict, geo_report: dict) -> dict:
    d0ft = _metrics(d0ft_report)
    control = _metrics(control_report)
    geo = _metrics(geo_report)
    c_size, c_by_class = _size_mean(control_report)
    g_size, g_by_class = _size_mean(geo_report)
    control["size_class_mean_map50_95"] = c_size
    control["size_map50_95_by_class"] = c_by_class
    geo["size_class_mean_map50_95"] = g_size
    geo["size_map50_95_by_class"] = g_by_class
    deltas = {metric: geo[metric] - control[metric] for metric in METRICS}
    deltas["size_class_mean_map50_95"] = g_size - c_size
    return {
        "seed": seed,
        "results": {"D0FT": d0ft, "GEO-C0": control, "GEO1": geo},
        "control_validity": _control_validity(control, d0ft),
        "geo_minus_geoc0": deltas,
    }


def _aggregate(per_seed: dict[str, dict]) -> tuple[dict, dict]:
    metric_names = (*METRICS, "size_class_mean_map50_95")
    aggregate = {}
    for metric in metric_names:
        deltas = [float(per_seed[str(seed)]["geo_minus_geoc0"][metric]) for seed in ALL_SEEDS]
        aggregate[metric] = {
            "delta_mean": sum(deltas) / len(deltas),
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "improved_seeds": sum(value > 0.0 for value in deltas),
            "per_seed_deltas": {str(seed): deltas[index] for index, seed in enumerate(ALL_SEEDS)},
        }
    tail_best = max(
        aggregate["bottom3_class_map50_95"]["delta_mean"],
        aggregate["worst_class_map50_95"]["delta_mean"],
    )
    control_all_pass = all(
        per_seed[str(seed)]["control_validity"]["decision"] == "PASS" for seed in ALL_SEEDS
    )
    criteria = {
        "macro_mean_gain_at_least_0_2_point": aggregate["macro_map50_95"]["delta_mean"] >= 0.002,
        "macro_improved_at_least_2_of_3": aggregate["macro_map50_95"]["improved_seeds"] >= 2,
        "bottom3_mean_not_lower": aggregate["bottom3_class_map50_95"]["delta_mean"] >= 0.0,
        "bottom3_improved_at_least_2_of_3": aggregate["bottom3_class_map50_95"]["improved_seeds"] >= 2,
        "worst_mean_not_lower": aggregate["worst_class_map50_95"]["delta_mean"] >= 0.0,
        "worst_improved_at_least_2_of_3": aggregate["worst_class_map50_95"]["improved_seeds"] >= 2,
        "size_mean_gain_at_least_0_5_point": aggregate["size_class_mean_map50_95"]["delta_mean"] >= 0.005,
        "size_mean_improved_at_least_2_of_3": aggregate["size_class_mean_map50_95"]["improved_seeds"] >= 2,
        "at_least_one_tail_mean_gain_at_least_0_5_point": tail_best >= 0.005,
        "all_three_geoc0_validity_pass": control_all_pass,
    }
    return aggregate, criteria


def run_confirmation(
    data_root: str | Path,
    seed42_summary: str | Path,
    paired_control_root: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if stage not in {"static", "train"}:
        raise ValueError("stage harus static/train")
    data_root = Path(data_root).expanduser().resolve()
    paired_root = Path(paired_control_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    layout = discover_layout(data_root)
    seed42 = _validate_seed42_summary(seed42_summary)

    static_root = output_root / "static_audits"
    reports_root = output_root / "val_reports"
    static_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    if stage == "static":
        provenance = {}
        gates = {
            "seed42_retain_valid": True,
            "development_has_no_test": not (data_root / "test").exists(),
        }
        for seed in NEW_SEEDS:
            reused = _validate_reused_pair(paired_root, seed)
            provenance[str(seed)] = reused
            static_path = static_root / f"GEO_seed{seed}_static.json"
            static = static_geometry_conditioning_audit(
                MODEL_YAML,
                reused["d0_checkpoint"],
                layout.names,
                static_path,
                nc=len(layout.names),
            )
            gates[f"seed{seed}_static_pass"] = static.get("decision") == "PASS"
        payload = {
            "protocol": f"{PROTOCOL}-static",
            "seeds": list(ALL_SEEDS),
            "evaluation_split": "val",
            "test_images_accessed": False,
            "test_opened": False,
            "seed42_summary": str(Path(seed42_summary).expanduser().resolve()),
            "reused_pair_provenance": provenance,
            "gates": gates,
            "decision": "PASS" if all(gates.values()) else "FAIL",
        }
        path = output_root / "static_preflight.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["summary"] = str(path)
        return payload

    if not authorize_training:
        raise RuntimeError("Paired GEO confirmation belum diotorisasi")
    preflight = _load_json(output_root / "static_preflight.json", "Static preflight")
    if preflight.get("decision") != "PASS":
        raise RuntimeError("Static preflight paired GEO belum PASS")
    dataset_audit = audit_dataset(data_root, reports_root / "dataset_audit.json", near_threshold=-1)
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    config_payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    config = GeometryConditioningConfig.from_mapping(config_payload["geometry_conditioning"])
    train_args = dict(config_payload["train"])

    seed42_control = seed42["results"]["GEO-C0"]
    seed42_geo = seed42["results"]["GEO1"]
    seed42_d0ft = seed42["results"]["D0FT"]
    seed42_record = {
        "seed": 42,
        "source": str(Path(seed42_summary).expanduser().resolve()),
        "results": {"D0FT": seed42_d0ft, "GEO-C0": seed42_control, "GEO1": seed42_geo},
        "control_validity": seed42["control_validity"],
        "geo_minus_geoc0": {
            **{metric: float(seed42_geo[metric]) - float(seed42_control[metric]) for metric in METRICS},
            "size_class_mean_map50_95": (
                float(seed42_geo["size_class_mean_map50_95"])
                - float(seed42_control["size_class_mean_map50_95"])
            ),
        },
    }
    per_seed = {"42": seed42_record}
    training_executed = {}

    for seed in NEW_SEEDS:
        print(f"\n=== PAIRED GEO CONFIRMATION | SEED {seed} ===", flush=True)
        reused = _validate_reused_pair(paired_root, seed)
        d0_checkpoint = Path(reused["d0_checkpoint"])
        d0ft_report = _load_json(reused["d0ft_report"], f"D0FT seed{seed} val")
        reports = {}
        for arm, mode in (("GEO-C0", "zero"), ("GEO1", "geometry")):
            best, executed = _train_arm(
                arm=arm,
                signal_mode=mode,
                seed=seed,
                d0_checkpoint=d0_checkpoint,
                data_root=data_root,
                output_root=output_root,
                config=config,
                train_args=train_args,
                device=device,
            )
            report = evaluate(
                best,
                data_root,
                reports_root / f"{arm}_seed{seed}_val.json",
                split="val",
                device=device,
            )
            if report["metrics"].get("classes_without_ground_truth", []):
                raise RuntimeError(f"Validation {arm} seed{seed} kehilangan kelas")
            reports[arm] = report
            training_executed[f"{arm}_seed{seed}"] = executed
        record = _seed_record(seed, d0ft_report, reports["GEO-C0"], reports["GEO1"])
        record["provenance"] = reused
        per_seed[str(seed)] = record

    aggregate, criteria = _aggregate(per_seed)
    decision = "PASS" if all(criteria.values()) else "FAIL"
    result = {
        "protocol": PROTOCOL,
        "seeds": list(ALL_SEEDS),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "control_definition": "parameter-matched zero-information control; not functional equivalence",
        "geometry_source": "detached decoded predicted boxes; no GT geometry input",
        "per_seed": per_seed,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": decision,
        "next_action": (
            "GEOMETRY_VALIDATION_CONFIRMED_READY_FOR_SEPARATE_FINAL_STAGE_REVIEW"
            if decision == "PASS"
            else "STOP_GEOMETRY_CAUSAL_CLAIM_WITHOUT_LOCKED_TEST_OR_POSTHOC_TUNING"
        ),
        "training_executed_this_call": training_executed,
    }
    summary = reports_root / "geometry_conditioning_paired_three_seed_confirmation.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 paired three-seed GEO confirmation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed42-summary", required=True)
    parser.add_argument("--paired-control-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("static", "train"), required=True)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_confirmation(
        args.data_root,
        args.seed42_summary,
        args.paired_control_root,
        args.output_root,
        stage=args.stage,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
