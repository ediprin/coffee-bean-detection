from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.drnet_refinement import DRNetRefinementConfig, make_drnet_refinement_trainer
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    "DRF1": REPO_ROOT / "configs/drnet_refinement/DRF1_yolo26n_dual_refinement.yaml",
    "DRC1": REPO_ROOT / "configs/drnet_refinement/DRC1_yolo26n_dual_refinement_cml.yaml",
}
METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


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


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def _load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"code", "model", "drnet_refinement", "train"}
    if not required <= set(payload):
        raise ValueError(f"Config DRNet tidak lengkap: {path}")
    DRNetRefinementConfig.from_mapping(payload["drnet_refinement"])
    return payload


def _run_candidate(
    code: str,
    config_path: Path,
    data_root: Path,
    d0_checkpoint: Path,
    output_root: Path,
    *,
    seed: int,
    device: str | None,
) -> dict:
    from ultralytics import YOLO

    config = _load_config(config_path)
    if config["code"] != code:
        raise RuntimeError(f"Code config mismatch: {config['code']} != {code}")
    run_dir = output_root / f"{code}_seed{seed}"
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    training_executed = False

    if not best.is_file():
        if last.is_file():
            print(f"RESUME {code} seed={seed}", flush=True)
            model = YOLO(str(last))
            args = {"resume": True}
            if device is not None:
                args["device"] = device
            model.train(**args)
        else:
            print(f"START {code} seed={seed}", flush=True)
            model_path = REPO_ROOT / str(config["model"])
            model = YOLO(str(model_path))
            train_args = dict(config["train"])
            train_args.update(
                {
                    "data": str(data_root / "data.yaml"),
                    "project": str(output_root),
                    "name": f"{code}_seed{seed}",
                    "exist_ok": True,
                    "seed": int(seed),
                    "deterministic": True,
                    "plots": True,
                    "verbose": True,
                }
            )
            if device is not None:
                train_args["device"] = device
            trainer = make_drnet_refinement_trainer(
                config["drnet_refinement"], d0_checkpoint=d0_checkpoint
            )
            model.train(trainer=trainer, **train_args)
        training_executed = True

    if not best.is_file():
        raise FileNotFoundError(f"{code} best.pt tidak ditemukan setelah training: {best}")
    report_path = output_root / "val_reports" / f"{code}_seed{seed}_val.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate(best, data_root, report_path, split="val", device=device)
    if report["metrics"].get("classes_without_ground_truth", []):
        raise RuntimeError(f"Validation {code} kehilangan kelas")
    return {
        "metrics": _metrics(report),
        "checkpoint": str(best),
        "report": str(report_path),
        "training_executed_this_call": training_executed,
    }


def _discovery_gate(candidate: dict[str, float], d0ft: dict[str, float]) -> dict:
    delta = {name: candidate[name] - d0ft[name] for name in METRICS}
    criteria = {
        "macro_not_below_d0ft_by_more_than_0_2_point": delta["macro_map50_95"] >= -0.002,
        "bottom3_not_below_d0ft_by_more_than_2_points": delta["bottom3_class_map50_95"] >= -0.02,
        "worst_not_below_d0ft_by_more_than_3_points": delta["worst_class_map50_95"] >= -0.03,
        "has_discovery_signal": (
            delta["macro_map50_95"] >= 0.002
            or delta["bottom3_class_map50_95"] >= 0.005
            or delta["worst_class_map50_95"] >= 0.005
        ),
    }
    return {
        "delta_vs_D0FT": delta,
        "criteria": criteria,
        "decision": "RETAIN" if all(criteria.values()) else "REJECT",
    }


def run_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    control_summary: str | Path,
    d0_checkpoint: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("DRNet discovery screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("DRNet screening belum diotorisasi")

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh memiliki split test")
    control = _load_json(control_summary, "D0FT/ACMC1 control summary")
    if control.get("test_images_accessed") is not False or control.get("test_opened") is not False:
        raise RuntimeError("Control summary tidak membuktikan test lock")
    if control.get("d0_checkpoint_sha256") != _sha256(d0_checkpoint):
        raise RuntimeError("Checkpoint D0 berbeda dari control summary")
    for name in ("D0", "D0FT", "ACMC1"):
        if name not in control.get("results", {}):
            raise RuntimeError(f"Control summary tidak memiliki {name}")

    reports_root = output_root / "val_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(
        data_root, reports_root / "dataset_audit.json", near_threshold=-1
    )
    if not dataset_audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    candidates = {
        code: _run_candidate(
            code,
            config,
            data_root,
            d0_checkpoint,
            output_root,
            seed=seed,
            device=device,
        )
        for code, config in CONFIGS.items()
    }
    controls = {name: _metrics(control["results"][name]) for name in ("D0", "D0FT", "ACMC1")}
    decisions = {
        code: _discovery_gate(payload["metrics"], controls["D0FT"])
        for code, payload in candidates.items()
    }
    for code, payload in candidates.items():
        decisions[code]["delta_vs_ACMC1"] = {
            name: payload["metrics"][name] - controls["ACMC1"][name] for name in METRICS
        }
    decisions["DRC1"]["delta_vs_DRF1"] = {
        name: candidates["DRC1"]["metrics"][name] - candidates["DRF1"]["metrics"][name]
        for name in METRICS
    }

    result = {
        "protocol": "faruq-v3-drnet-dual-refinement-discovery-v1",
        "stage": "breadth_discovery",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "paper_transfer_boundary": (
            "DRNet FGB equations (1)-(4) are transferred from 7x7 RoI features "
            "to dense YOLO26 P3/P4/P5 classification fields. CML uses the paper's "
            "separability weighting on positive assigned samples only because YOLO26 "
            "has no explicit background subclass. This is not a literal ORCNN reproduction."
        ),
        "controls": controls,
        "candidates": candidates,
        "decisions": decisions,
    }
    summary = reports_root / "drnet_refinement_seed42_screening.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 DRNet FGB/CML breadth screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_screening(
        args.data_root,
        args.grouped_summary,
        args.control_summary,
        args.d0_checkpoint,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
