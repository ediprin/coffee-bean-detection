"""Fail-fast Faruq-v3 AGSF synthesis screening on validation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.agsf import AGSFConfig, make_agsf_trainer
from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "SYN0": REPO_ROOT / "configs/agsf/SYN0_stb_ambiguity.yaml",
    "SYN1": REPO_ROOT / "configs/agsf/SYN1_stb_ambiguity_af2_additive.yaml",
    "SYN2": REPO_ROOT / "configs/agsf/SYN2_stb_ambiguity_af2_gated.yaml",
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
    return (
        int(epoch) if epoch is not None else None,
        payload.get("optimizer") is not None,
    )


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


def _run_is_complete(run_dir: Path, expected_epochs: int) -> bool:
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    if not best.is_file() or not last.is_file():
        return False
    if _result_epochs(run_dir / "results.csv") >= int(expected_epochs):
        return True
    epoch, has_optimizer = _checkpoint_state(last)
    return epoch == -1 and not has_optimizer


def _train_arm(
    arm: str,
    data_root: Path,
    output_root: Path,
    d0_checkpoint: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[Path, bool, dict]:
    from ultralytics import YOLO

    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    config = AGSFConfig.from_mapping(payload["agsf"])
    expected_epochs = int(payload["train"]["epochs"])
    run_dir = output_root / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    trainer = make_agsf_trainer(config, d0_checkpoint=d0_checkpoint)
    executed = False
    if not _run_is_complete(run_dir, expected_epochs):
        epoch, resumable = _checkpoint_state(last)
        if last.is_file() and resumable and epoch is not None and epoch >= 0:
            print(f"RESUME {arm} seed={seed} dari epoch {epoch + 1}/{expected_epochs}", flush=True)
            model = YOLO(str(last))
            args = {"resume": True}
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        else:
            print(f"START {arm} seed={seed}", flush=True)
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
    if not _run_is_complete(run_dir, expected_epochs):
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


def run_agsf_synthesis(
    data_root: str | Path,
    grouped_summary: str | Path,
    stb_summary: str | Path,
    d0_checkpoint: str | Path,
    static_audit: str | Path,
    output_root: str | Path,
    *,
    stage: str,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("AGSF synthesis screening dikunci seed 42")
    if stage not in {"core", "frequency"}:
        raise ValueError("stage harus core atau frequency")
    if not authorize_training:
        raise RuntimeError("AGSF training belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos test")

    checkpoint_hash = _sha256(d0_checkpoint)
    reference_payload = _load_json(stb_summary, "STB1 reference summary")
    if (
        reference_payload.get("seed") != seed
        or reference_payload.get("test_images_accessed") is not False
        or reference_payload.get("test_opened") is not False
        or reference_payload.get("d0_checkpoint_sha256") != checkpoint_hash
        or reference_payload.get("decision") != "RETAIN"
    ):
        raise RuntimeError("STB1 reference tidak sesuai protokol")
    reference = _metrics(reference_payload["candidate"]["STB1"])
    controls = {
        name: _metrics(value)
        for name, value in reference_payload["controls"].items()
    }

    static = _load_json(static_audit, "AGSF static audit")
    if (
        static.get("decision") != "PASS"
        or static.get("test_images_accessed") is not False
        or static.get("d0_checkpoint_sha256") != checkpoint_hash
        or not all(
            static.get("capacity_gates", {}).get(name) is True
            for name in (
                "syn1_syn2_same_parameter_count",
                "syn1_syn2_same_state_schema",
                "frequency_is_classification_side_only",
            )
        )
    ):
        raise RuntimeError("AGSF static audit belum PASS")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    arms = ("SYN0",) if stage == "core" else ("SYN1", "SYN2")
    candidates, configs, trained = {}, {}, {}
    for arm in arms:
        checkpoint, executed, config = _train_arm(
            arm,
            data_root,
            output_root,
            d0_checkpoint,
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

    if stage == "core":
        comparisons = {"STB1_vs_SYN0": _comparison(candidates["SYN0"], reference)}
        decision = comparisons["STB1_vs_SYN0"]["decision"]
        next_action = (
            "AUTHORIZE_AGSF_FREQUENCY_STAGE"
            if decision == "PASS"
            else "STOP_AGSF_WITHOUT_FREQUENCY_OR_TEST"
        )
        summary_name = "agsf_core_seed42_decision.json"
    else:
        core = _load_json(reports / "agsf_core_seed42_decision.json", "AGSF core decision")
        if core.get("decision") != "PASS" or core.get("test_opened") is not False:
            raise RuntimeError("Frequency stage memerlukan core PASS")
        syn0 = _metrics(core["candidates"]["SYN0"])
        candidates = {"SYN0": syn0, **candidates}
        comparisons = {
            "STB1_vs_SYN2": _comparison(candidates["SYN2"], reference),
            "SYN0_vs_SYN2": _comparison(candidates["SYN2"], candidates["SYN0"]),
            "SYN1_vs_SYN2": _comparison(candidates["SYN2"], candidates["SYN1"]),
        }
        decision = "PASS" if all(
            value["decision"] == "PASS" for value in comparisons.values()
        ) else "FAIL"
        next_action = (
            "AUTHORIZE_SYN2_MULTI_SEED_CONFIRMATION"
            if decision == "PASS"
            else "STOP_AGSF_WITHOUT_TEST_OR_EXTRA_SEEDS"
        )
        summary_name = "agsf_frequency_seed42_decision.json"

    payload = {
        "protocol": "faruq-v3-agsf-synthesis-v1",
        "stage": stage,
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": checkpoint_hash,
        "reference": {"STB1": reference, **controls},
        "candidates": candidates,
        "comparisons": comparisons,
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
    parser = argparse.ArgumentParser(description="Faruq-v3 AGSF fail-fast synthesis")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--stb-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("core", "frequency"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_agsf_synthesis(
        args.data_root,
        args.grouped_summary,
        args.stb_summary,
        args.d0_checkpoint,
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
