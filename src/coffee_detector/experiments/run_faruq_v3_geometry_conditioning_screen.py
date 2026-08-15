"""Matched GEO-C0 vs GEO1 seed42 screening for Faruq-v3."""

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

import numpy as np
import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.dataset import discover_layout
from coffee_detector.evaluate import evaluate
from coffee_detector.geometry_conditioning.audit import static_geometry_conditioning_audit
from coffee_detector.geometry_conditioning.model import GeometryConditioningConfig, _is_size_class
from coffee_detector.geometry_conditioning.trainer import make_geometry_conditioning_trainer

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG = REPO_ROOT / "configs/geometry_conditioning/GEO1_yolo26n_predbox_geometry.yaml"
MODEL_YAML = REPO_ROOT / "configs/coffee_fg/models/yolo26n-p3.yaml"
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


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


def _metrics(payload: dict, result_key: str | None = None) -> dict[str, float]:
    candidates = []
    if isinstance(payload.get("metrics"), dict):
        candidates.append(payload["metrics"])
    if all(name in payload for name in METRICS):
        candidates.append(payload)
    if result_key is not None and isinstance(payload.get("results"), dict):
        row = payload["results"].get(result_key)
        if isinstance(row, dict):
            candidates.append(row)
    for source in candidates:
        if all(name in source for name in METRICS):
            return {name: float(source[name]) for name in METRICS}
    raise KeyError(f"Metrik {METRICS} tidak ditemukan untuk {result_key or 'report'}")


def _size_mean(report: dict) -> tuple[float, dict[str, float]]:
    by_class = report.get("metrics", {}).get("map50_95_by_class", {})
    selected = {
        str(name): float(value)
        for name, value in by_class.items()
        if _is_size_class(str(name))
    }
    if len(selected) < 2:
        raise RuntimeError("Validation report tidak memiliki cukup kelas size-defined")
    return float(np.mean(list(selected.values()))), selected


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
        return (epoch == -1 and not resumable) or (
            epoch is not None and epoch + 1 >= epochs
        )
    return epoch == -1 and not resumable


@contextmanager
def _training_lock(output_root: Path, arm: str, stale_seconds: int = 300):
    lock = output_root / f"{arm}_seed42.training.lock"
    token = uuid.uuid4().hex
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(
                descriptor,
                json.dumps({"token": token, "pid": os.getpid()}).encode(),
            )
            os.close(descriptor)
            break
        except FileExistsError:
            age = time.time() - lock.stat().st_mtime
            if age <= stale_seconds:
                raise RuntimeError(f"{arm} sedang ditulis runtime lain ({age:.0f}s)")
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


def _control_validity(control: dict, d0ft: dict) -> dict:
    delta = {name: control[name] - d0ft[name] for name in METRICS}
    criteria = {
        "macro_drop_no_more_than_1_point": delta["macro_map50_95"] >= -0.01,
        "bottom3_drop_no_more_than_2_points": delta["bottom3_class_map50_95"] >= -0.02,
        "worst_drop_no_more_than_3_points": delta["worst_class_map50_95"] >= -0.03,
    }
    return {
        "deltas": delta,
        "criteria": criteria,
        "decision": "PASS" if all(criteria.values()) else "FAIL",
    }


def _geometry_gate(geo: dict, control: dict, d0ft: dict) -> dict:
    delta = {name: geo[name] - control[name] for name in METRICS}
    tail_gain = max(
        delta["bottom3_class_map50_95"], delta["worst_class_map50_95"]
    )
    criteria = {
        "macro_gain_vs_geoc0_at_least_0_2_point": delta["macro_map50_95"] >= 0.002,
        "bottom3_drop_vs_geoc0_no_more_than_0_5_point": delta["bottom3_class_map50_95"] >= -0.005,
        "worst_drop_vs_geoc0_no_more_than_0_5_point": delta["worst_class_map50_95"] >= -0.005,
        "at_least_one_tail_gain_vs_geoc0_at_least_0_5_point": tail_gain >= 0.005,
        "size_mean_gain_vs_geoc0_at_least_0_5_point": (
            geo["size_class_mean_map50_95"]
            - control["size_class_mean_map50_95"]
            >= 0.005
        ),
        "macro_vs_d0ft_no_worse_than_0_2_point": (
            geo["macro_map50_95"] - d0ft["macro_map50_95"] >= -0.002
        ),
    }
    return {
        "deltas": delta,
        "criteria": criteria,
        "decision": "PASS" if all(criteria.values()) else "FAIL",
    }


def _train_arm(
    *,
    arm: str,
    signal_mode: str,
    config: GeometryConditioningConfig,
    train_args: dict,
    data_root: Path,
    d0_checkpoint: Path,
    output_root: Path,
    seed: int,
    device: str | None,
) -> tuple[Path, bool]:
    from ultralytics import YOLO

    epochs = int(train_args["epochs"])
    run_dir = output_root / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    executed = False
    if not _run_complete(run_dir, epochs):
        trainer = make_geometry_conditioning_trainer(
            config,
            d0_checkpoint=d0_checkpoint,
            signal_mode=signal_mode,
        )
        epoch, resumable = _checkpoint_state(last)
        with _training_lock(output_root, arm):
            if last.is_file() and resumable and epoch is not None and epoch >= 0:
                print(f"RESUME {arm} dari epoch {epoch + 1}/{epochs}", flush=True)
                model, args = YOLO(str(last)), {"resume": True}
            else:
                print(f"START {arm} dari D0 seed42", flush=True)
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
        raise RuntimeError(f"Run {arm} belum lengkap: {run_dir}")
    return best, executed


def run_geometry_conditioning_screen(
    data_root: str | Path,
    d0_checkpoint: str | Path,
    d0ft_summary: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42 or stage not in {"static", "train"}:
        raise ValueError("Screen dikunci seed42 dan stage static/train")
    data_root = Path(data_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")
    layout = discover_layout(data_root)

    if stage == "static":
        return static_geometry_conditioning_audit(
            MODEL_YAML,
            d0_checkpoint,
            layout.names,
            output_root / "static_audit.json",
            nc=len(layout.names),
        )

    if not authorize_training:
        raise RuntimeError("Training GEO-C0/GEO1 belum diotorisasi")
    static = _load_json(output_root / "static_audit.json", "Static audit")
    if static.get("decision") != "PASS":
        raise RuntimeError("Static audit GEO belum PASS")
    d0ft_payload = _load_json(d0ft_summary, "D0FT summary")
    checkpoint_hash = _sha256(d0_checkpoint)
    if d0ft_payload.get("d0_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("D0FT reference tidak berasal dari D0 checkpoint yang sama")
    if d0ft_payload.get("test_images_accessed") is not False:
        raise RuntimeError("D0FT reference tidak mematuhi test lock")
    d0ft = _metrics(d0ft_payload, "D0FT")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(
        data_root, reports / "dataset_audit.json", near_threshold=-1
    )
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    config = GeometryConditioningConfig.from_mapping(
        payload["geometry_conditioning"]
    )
    train_args = dict(payload["train"])
    results: dict[str, dict] = {"D0FT": dict(d0ft)}
    training_executed = {}
    for arm, mode in (("GEO-C0", "zero"), ("GEO1", "geometry")):
        best, executed = _train_arm(
            arm=arm,
            signal_mode=mode,
            config=config,
            train_args=train_args,
            data_root=data_root,
            d0_checkpoint=d0_checkpoint,
            output_root=output_root,
            seed=seed,
            device=device,
        )
        report = evaluate(
            best,
            data_root,
            reports / f"{arm}_seed42_val.json",
            split="val",
            device=device,
        )
        if report["metrics"].get("classes_without_ground_truth", []):
            raise RuntimeError(f"Validation {arm} kehilangan kelas")
        size_mean, size_by_class = _size_mean(report)
        row = _metrics(report)
        row["size_class_mean_map50_95"] = size_mean
        row["size_map50_95_by_class"] = size_by_class
        row["checkpoint"] = str(best)
        results[arm] = row
        training_executed[arm] = executed

    control_validity = _control_validity(results["GEO-C0"], d0ft)
    geometry_gate = _geometry_gate(results["GEO1"], results["GEO-C0"], d0ft)
    retain = (
        control_validity["decision"] == "PASS"
        and geometry_gate["decision"] == "PASS"
    )
    result = {
        "protocol": "faruq-v3-geometry-conditioning-screening-v1",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": checkpoint_hash,
        "control_definition": (
            "parameter-matched zero-information control; not functional equivalence"
        ),
        "geometry_source": "detached decoded predicted boxes; no GT geometry input",
        "results": results,
        "capacity": static["models"],
        "added_parameters": static["added_parameters"],
        "target_size_classes": static["target_size_classes"],
        "control_validity": control_validity,
        "geometry_retain_gate": geometry_gate,
        "decision": "RETAIN" if retain else "REJECT",
        "next_action": (
            "AUTHORIZE_PAIRED_THREE_SEED_GEO_CONFIRMATION"
            if retain
            else "STOP_GEOMETRY_CONDITIONING_AFTER_SEED42"
        ),
        "training_executed_this_call": training_executed,
    }
    summary = reports / "geometry_conditioning_seed42_decision.json"
    summary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Faruq-v3 GEO-C0 vs GEO1 seed42 screen"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--d0ft-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("static", "train"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_geometry_conditioning_screen(
        args.data_root,
        args.d0_checkpoint,
        args.d0ft_summary,
        args.output_root,
        stage=args.stage,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
