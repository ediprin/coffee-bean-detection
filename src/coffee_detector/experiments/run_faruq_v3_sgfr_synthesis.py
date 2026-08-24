"""Validation-only staged SGFR synthesis on Faruq-v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.sgfr import (
    SGFRConfig,
    audit_sgfr_checkpoint_invariance,
    make_sgfr_trainer,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "SGC0": REPO_ROOT / "configs/sgfr/SGC0_stb_continued_control.yaml",
    "SGI1": REPO_ROOT / "configs/sgfr/SGI1_stb_igem_frozen_residual.yaml",
    "SGF2": REPO_ROOT / "configs/sgfr/SGF2_stb_igem_af2_frozen_residual.yaml",
}
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


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def _checkpoint_state(path: Path) -> tuple[int | None, bool]:
    if not path.is_file():
        return None, False
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None, False
    epoch = payload.get("epoch")
    return int(epoch) if epoch is not None else None, payload.get("optimizer") is not None


def _result_epochs(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return 0
    try:
        return int(float(rows[-1].get("epoch", len(rows))))
    except (TypeError, ValueError):
        return len(rows)


def _run_complete(run_dir: Path, epochs: int) -> bool:
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    if not best.is_file() or not last.is_file():
        return False
    if _result_epochs(run_dir / "results.csv") >= int(epochs):
        return True
    epoch, resumable = _checkpoint_state(last)
    return epoch == -1 and not resumable


def _train_arm(
    arm: str,
    data_root: Path,
    output_root: Path,
    source_checkpoint: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[Path, bool, dict]:
    from ultralytics import YOLO

    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    config = SGFRConfig.from_mapping(payload["sgfr"])
    epochs = int(payload["train"]["epochs"])
    run_dir = output_root / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    trainer = make_sgfr_trainer(config, source_checkpoint=source_checkpoint)
    executed = False
    if not _run_complete(run_dir, epochs):
        epoch, resumable = _checkpoint_state(last)
        if last.is_file() and resumable and epoch is not None and epoch >= 0:
            print(f"RESUME {arm} seed={seed} dari epoch {epoch + 1}/{epochs}", flush=True)
            model = YOLO(str(last))
            args: dict = {"resume": True}
        else:
            print(f"START {arm} seed={seed} source={source_checkpoint}", flush=True)
            model = YOLO(str(REPO_ROOT / payload["model"]))
            args = dict(payload["train"])
            args.update(
                {
                    "data": str(data_root / "data.yaml"),
                    "project": str(output_root),
                    "name": f"{arm}_seed{seed}",
                    "exist_ok": True,
                    "seed": seed,
                    "deterministic": True,
                    "plots": True,
                    "verbose": True,
                }
            )
        if device is not None:
            args["device"] = device
        model.train(trainer=trainer, **args)
        executed = True
    if not _run_complete(run_dir, epochs):
        raise RuntimeError(f"Run {arm} belum lengkap: {run_dir}")
    return best, executed, config.to_dict()


def _comparison(candidate: dict, reference: dict) -> dict:
    delta = {name: candidate[name] - reference[name] for name in METRICS}
    criteria = {
        "macro_gain_at_least_0_5_point": delta["macro_map50_95"] >= 0.005,
        "bottom3_not_lower": delta["bottom3_class_map50_95"] >= 0.0,
        "worst_drop_no_more_than_1_point": delta["worst_class_map50_95"] >= -0.01,
    }
    return {
        "deltas": delta,
        "criteria": criteria,
        "decision": "PASS" if all(criteria.values()) else "FAIL",
    }


def run_sgfr_synthesis(
    data_root: str | Path,
    grouped_summary: str | Path,
    stb_summary: str | Path,
    stb_checkpoint: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("Screening SGFR dikunci seed 42")
    if stage not in {"geometry", "frequency"}:
        raise ValueError("stage harus geometry atau frequency")
    if not authorize_training:
        raise RuntimeError("Training SGFR belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    stb_checkpoint = Path(stb_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")

    stb_hash = _sha256(stb_checkpoint)
    reference_payload = _load_json(stb_summary, "STB1 summary")
    if (
        reference_payload.get("seed") != seed
        or reference_payload.get("decision") != "RETAIN"
        or reference_payload.get("test_images_accessed") is not False
        or reference_payload.get("test_opened") is not False
    ):
        raise RuntimeError("STB1 reference tidak sesuai protokol")
    stb_metrics = _metrics(reference_payload["candidate"]["STB1"])

    static = _load_json(static_audit, "SGFR static audit")
    if (
        static.get("decision") != "PASS"
        or static.get("test_images_accessed") is not False
        or static.get("stb_checkpoint_sha256") != stb_hash
        or not all(
            static.get("arms", {}).get(arm, {}).get("decision") == "PASS"
            for arm in ("SGC0", "SGI1", "SGF2")
        )
    ):
        raise RuntimeError("Static audit SGFR belum PASS")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    candidates: dict[str, dict[str, float]] = {}
    configs, trained, invariance = {}, {}, {}
    if stage == "geometry":
        arms = ("SGC0", "SGI1")
        source_by_arm = {arm: stb_checkpoint for arm in arms}
    else:
        core = _load_json(reports / "sgfr_geometry_seed42_decision.json", "SGFR geometry gate")
        if core.get("decision") != "PASS" or core.get("test_opened") is not False:
            raise RuntimeError("Frequency stage memerlukan geometry PASS")
        sgi_checkpoint = output_root / f"SGI1_seed{seed}/weights/best.pt"
        if not sgi_checkpoint.is_file():
            raise FileNotFoundError(sgi_checkpoint)
        candidates.update(
            {name: _metrics(value) for name, value in core["candidates"].items()}
        )
        arms = ("SGF2",)
        source_by_arm = {"SGF2": sgi_checkpoint}

    for arm in arms:
        checkpoint, executed, config = _train_arm(
            arm,
            data_root,
            output_root,
            source_by_arm[arm],
            seed=seed,
            device=device,
        )
        report = evaluate(
            checkpoint,
            data_root,
            reports / f"{arm}_seed{seed}_val.json",
            split="val",
            device=device,
        )
        if report["metrics"].get("classes_without_ground_truth", []):
            raise RuntimeError(f"Validation {arm} kehilangan kelas")
        candidates[arm] = _metrics(report)
        configs[arm] = config
        trained[arm] = executed
        stage_name = {"SGC0": "control", "SGI1": "geometry", "SGF2": "frequency"}[arm]
        invariant = audit_sgfr_checkpoint_invariance(
            source_by_arm[arm], checkpoint, stage=stage_name
        )
        invariance[arm] = invariant
        if invariant["decision"] != "PASS":
            raise RuntimeError(f"Frozen-state invariant gagal untuk {arm}")

    if stage == "geometry":
        comparisons = {
            "STB1_vs_SGI1": _comparison(candidates["SGI1"], stb_metrics),
            "SGC0_vs_SGI1": _comparison(candidates["SGI1"], candidates["SGC0"]),
        }
        decision = "PASS" if all(
            value["decision"] == "PASS" for value in comparisons.values()
        ) else "FAIL"
        next_action = (
            "AUTHORIZE_SGFR_FREQUENCY_STAGE"
            if decision == "PASS"
            else "STOP_SGFR_WITHOUT_FREQUENCY_OR_TEST"
        )
        summary_name = "sgfr_geometry_seed42_decision.json"
    else:
        comparisons = {
            "STB1_vs_SGF2": _comparison(candidates["SGF2"], stb_metrics),
            "SGC0_vs_SGF2": _comparison(candidates["SGF2"], candidates["SGC0"]),
            "SGI1_vs_SGF2": _comparison(candidates["SGF2"], candidates["SGI1"]),
        }
        decision = "PASS" if all(
            value["decision"] == "PASS" for value in comparisons.values()
        ) else "FAIL"
        next_action = (
            "AUTHORIZE_SGF2_MULTI_SEED_CONFIRMATION"
            if decision == "PASS"
            else "RETAIN_SGI1_AND_STOP_SGF2_WITHOUT_TEST"
        )
        summary_name = "sgfr_frequency_seed42_decision.json"

    payload = {
        "protocol": "faruq-v3-sgfr-frozen-synthesis-v1",
        "stage": stage,
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "stb_checkpoint_sha256": stb_hash,
        "reference": {"STB1": stb_metrics},
        "candidates": candidates,
        "comparisons": comparisons,
        "checkpoint_invariance": invariance,
        "configs": configs,
        "training_executed_this_call": trained,
        "decision": decision,
        "next_action": next_action,
    }
    summary = reports / summary_name
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 SGFR frozen synthesis")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--stb-summary", required=True)
    parser.add_argument("--stb-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("geometry", "frequency"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_sgfr_synthesis(
        args.data_root,
        args.grouped_summary,
        args.stb_summary,
        args.stb_checkpoint,
        args.static_audit,
        args.output_root,
        stage=args.stage,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
