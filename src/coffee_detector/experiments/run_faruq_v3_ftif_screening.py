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
from coffee_detector.ftif import (
    FTIFConfig,
    load_prompt_manifest,
    load_text_embedding_payload,
    make_ftif_trainer,
    prompt_manifest_sha256,
    validate_manifest_against_class_names,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT_MANIFEST = REPO_ROOT / "configs/ftif/sni21_prompts.yaml"
CONFIGS = {
    "FT1": REPO_ROOT / "configs/ftif/FT1_specific_crossattn.yaml",
    "FT2": REPO_ROOT / "configs/ftif/FT2_base_specific_crossattn.yaml",
    "FT3": REPO_ROOT / "configs/ftif/FT3_base_specific_bidirectional.yaml",
}


def _checkpoint_state(path: Path) -> tuple[int | None, bool]:
    """Return checkpoint epoch and whether resumable optimizer state exists."""
    if not path.is_file():
        return None, False
    from ultralytics.utils.patches import torch_load

    payload = torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None, False
    epoch = payload.get("epoch")
    return (int(epoch) if epoch is not None else None), payload.get("optimizer") is not None


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
    best = run_dir / "weights/best.pt"
    last = run_dir / "weights/last.pt"
    if not best.is_file() or not last.is_file():
        return False
    if _result_epochs(run_dir / "results.csv") >= int(expected_epochs):
        return True
    epoch, has_optimizer = _checkpoint_state(last)
    # Ultralytics strips completed (including legitimately early-stopped)
    # checkpoints to epoch=-1 and removes optimizer state during finalization.
    return epoch == -1 and not has_optimizer


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


def _metrics(payload: dict) -> dict[str, float]:
    source = payload.get("metrics", payload)
    return {name: float(source[name]) for name in METRICS}


def _decision(candidate: dict[str, float], d0ft: dict[str, float]) -> tuple[dict, dict, str]:
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
    return delta, criteria, "RETAIN" if all(criteria.values()) else "REJECT"


def _dataset_names(data_yaml: Path) -> list[str]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(int(k) for k in names)]
    return [str(value) for value in names]


def _train_arm(
    arm: str,
    data_root: Path,
    output_root: Path,
    d0_checkpoint: Path,
    text_embedding_path: Path,
    *,
    seed: int,
    device: str | None,
) -> tuple[Path, bool, dict]:
    payload = yaml.safe_load(CONFIGS[arm].read_text(encoding="utf-8")) or {}
    config = FTIFConfig.from_mapping(payload["ftif"])
    from ultralytics import YOLO

    run_dir = output_root / f"{arm}_seed{seed}"
    best, last = run_dir / "weights/best.pt", run_dir / "weights/last.pt"
    expected_epochs = int(payload["train"]["epochs"])
    trainer = make_ftif_trainer(
        config,
        text_embedding_path=text_embedding_path,
        d0_checkpoint=d0_checkpoint,
    )
    training_executed = False
    if not _run_is_complete(run_dir, expected_epochs):
        last_epoch, last_resumable = _checkpoint_state(last)
        if last.is_file() and last_resumable and last_epoch is not None and last_epoch >= 0:
            print(
                f"RESUME {arm} seed={seed} dari epoch {last_epoch + 1}/{expected_epochs}",
                flush=True,
            )
            model = YOLO(str(last))
            args = {"resume": True}
            if device is not None:
                args["device"] = device
            model.train(trainer=trainer, **args)
        else:
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
        training_executed = True
    if not _run_is_complete(run_dir, expected_epochs):
        raise RuntimeError(
            f"Run {arm} belum lengkap setelah trainer selesai: {run_dir}. "
            "Checkpoint parsial tidak boleh dievaluasi."
        )
    return best, training_executed, config.to_dict()


def run_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    control_summary: str | Path,
    d0_checkpoint: str | Path,
    text_embeddings: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str | None = None,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("FTIF breadth discovery dikunci seed 42")
    if not authorize_training:
        raise RuntimeError("FTIF screening belum diotorisasi")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    d0_checkpoint = Path(d0_checkpoint).expanduser().resolve()
    text_embeddings = Path(text_embeddings).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Development root tidak boleh mengekspos split test")

    data_names = _dataset_names(data_root / "data.yaml")
    manifest = load_prompt_manifest(PROMPT_MANIFEST)
    validate_manifest_against_class_names(manifest, data_names)
    _, cache_meta = load_text_embedding_payload(
        text_embeddings, class_names=data_names, prompt_mode="base_specific"
    )
    manifest_hash = prompt_manifest_sha256(PROMPT_MANIFEST)
    if cache_meta.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("FTIF embedding cache dibuat dari prompt manifest berbeda")

    control = _load_json(control_summary, "D0FT/ACMC1 control summary")
    if control.get("test_images_accessed") is not False or control.get("test_opened") is not False:
        raise RuntimeError("Control summary tidak membuktikan test lock")
    if control.get("d0_checkpoint_sha256") != _sha256(d0_checkpoint):
        raise RuntimeError("Checkpoint D0 berbeda dari control summary")

    reports = output_root / "val_reports"
    reports.mkdir(parents=True, exist_ok=True)
    audit = audit_dataset(data_root, reports / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    controls = {name: _metrics(control["results"][name]) for name in ("D0", "D0FT", "ACMC1")}
    candidates, decisions, configs, trained = {}, {}, {}, {}
    for arm in ("FT1", "FT2", "FT3"):
        best, executed, config = _train_arm(
            arm,
            data_root,
            output_root,
            d0_checkpoint,
            text_embeddings,
            seed=seed,
            device=device,
        )
        report = evaluate(
            best,
            data_root,
            reports / f"{arm}_seed{seed}_val.json",
            split="val",
            device=device,
        )
        metrics = _metrics(report)
        delta, criteria, decision = _decision(metrics, controls["D0FT"])
        candidates[arm] = metrics
        decisions[arm] = {
            "delta_vs_D0FT": delta,
            "delta_vs_ACMC1": {
                name: metrics[name] - controls["ACMC1"][name] for name in METRICS
            },
            "criteria": criteria,
            "decision": decision,
        }
        configs[arm] = config
        trained[arm] = executed

    result = {
        "protocol": "faruq-v3-lfdet-ftif-breadth-screening-v1",
        "stage": "breadth_discovery",
        "seed": seed,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "d0_checkpoint_sha256": _sha256(d0_checkpoint),
        "text_embedding_sha256": _sha256(text_embeddings),
        "prompt_manifest_sha256": manifest_hash,
        "text_encoder": cache_meta.get("encoder", {}),
        "paper_settings_retained": {
            "temperature": 0.07,
            "frozen_text_encoder": True,
            "base_plus_specific_embedding_addition": True,
            "visual_query_text_key_value": True,
            "bidirectional_alignment_for_FT3": True,
        },
        "transfer_choices": {
            "text_encoder": "OpenAI CLIP ViT-B/32; paper text available to this project does not expose the exact CLIP variant",
            "identity_start": "native YOLO leaf logits plus zero-initialized FTIF residual correction",
            "alignment_weight": 1.0,
            "alignment_reduction": "positive and negative terms mean-reduced separately before LFDet Eq.20 combination",
            "prompt_source": "frozen SNI-21 ontology only; validation confusions forbidden",
        },
        "controls": controls,
        "candidate": candidates,
        "decisions": decisions,
        "configs": configs,
        "training_executed_this_call": trained,
    }
    summary = reports / "lfdet_ftif_seed42_screening.json"
    summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 LFDet FTIF breadth screening")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--text-embeddings", required=True)
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
        args.text_embeddings,
        args.output_root,
        seed=args.seed,
        device=args.device,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
