"""Development-val DC2 final-stage MSFA mechanism screen."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.dc2_crop.metrics import classification_summary
from coffee_detector.dc2_crop.msfa import (
    DC2MSFAClassifier,
    MSFAMatchedDataset,
    extract_p5_global_descriptors,
    load_dc2b_local_checkpoint,
    trainable_parameter_count,
)
from coffee_detector.dc2_crop.predicted import (
    MatchedRawObjectCropDataset,
    _sha256_file,
    collect_predicted_crop_records,
)
from coffee_detector.dc2_crop.model import predict_logits
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary


PROTOCOL = "faruq-v3-dc2-msfa-screening-v1"
DC2B_PROTOCOL = "faruq-v3-dc2-predicted-raw-crop-screening-v2"
RESOLUTION = 128
GLOBAL_LEVEL = "P5"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_loader(dataset, batch_size: int, *, shuffle: bool, workers: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _metric_view(metrics: dict, names: dict[int, str]) -> dict:
    output = dict(metrics)
    per_class = output.pop("per_class_f1", None)
    if per_class is not None:
        output["per_class"] = {
            names[index]: float(value) for index, value in enumerate(per_class)
        }
    return output


def _evaluate_msfa(
    model: DC2MSFAClassifier,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    logits_rows, labels_rows = [], []
    with torch.inference_mode():
        for images, global_features, labels in loader:
            images = images.to(device, non_blocking=True)
            global_features = global_features.to(device, non_blocking=True)
            logits_rows.append(model(images, global_features).cpu())
            labels_rows.append(labels.cpu())
    if not logits_rows:
        raise RuntimeError("MSFA validation loader kosong")
    return classification_summary(
        torch.cat(logits_rows, dim=0),
        torch.cat(labels_rows, dim=0),
        num_classes,
    )


def decide_dc2_msfa(
    *,
    train_coverage: float,
    val_coverage: float,
    replay_metrics: dict,
    dc2b_metrics: dict,
    msfa_metrics: dict,
    global_feature_std: float,
    replay_tolerance: float = 1e-5,
) -> dict:
    replay_delta = {
        key: float(replay_metrics[key] - dc2b_metrics[key])
        for key in ("macro_f1", "bottom3_f1", "worst_f1")
    }
    gains = {
        key: float(msfa_metrics[key] - replay_metrics[key])
        for key in ("macro_f1", "bottom3_f1", "worst_f1")
    }
    criteria = {
        "train_matched_coverage_at_least_90_percent": train_coverage >= 0.90,
        "val_matched_coverage_at_least_90_percent": val_coverage >= 0.90,
        "dc2b_replay_matches_frozen_checkpoint": all(
            abs(value) <= replay_tolerance for value in replay_delta.values()
        ),
        "global_descriptor_has_nontrivial_variation": global_feature_std > 1e-6,
        "msfa_macro_gain_at_least_half_point": gains["macro_f1"] >= 0.005,
        "msfa_bottom3_drop_no_more_than_quarter_point": gains["bottom3_f1"] >= -0.0025,
        "msfa_worst_drop_no_more_than_one_point": gains["worst_f1"] >= -0.01,
    }
    passed = all(criteria.values())
    return {
        "replay_delta_vs_dc2b_report": replay_delta,
        "deltas_msfa_vs_local_only": gains,
        "criteria": criteria,
        "decision": "PASS" if passed else "FAIL",
        "next_action": (
            "AUTHORIZE_DC2_END_TO_END_INTEGRATION_SCREENING"
            if passed
            else "DO_NOT_ESCALATE_DC2_TO_END_TO_END_FROM_MSFA_ARM"
        ),
    }


def _train_msfa_projection(
    local_checkpoint: Path,
    train_records,
    val_records,
    train_global: np.ndarray,
    val_global: np.ndarray,
    names: dict[int, str],
    output_root: Path,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[dict, dict]:
    local_model = load_dc2b_local_checkpoint(local_checkpoint, len(names)).to(device).eval()

    # Exact local-only replay on the same predicted-box records is a mandatory
    # control before the MSFA projection is trained.
    local_val = MatchedRawObjectCropDataset(
        val_records, RESOLUTION, training=False, source="predicted", context=1.0
    )
    local_loader = _make_loader(
        local_val, batch_size, shuffle=False, workers=workers, seed=seed
    )
    replay_logits, replay_labels = predict_logits(local_model, local_loader, device)
    replay_metrics = classification_summary(replay_logits, replay_labels, len(names))

    train_dataset = MSFAMatchedDataset(
        train_records, train_global, resolution=RESOLUTION, training=True
    )
    val_dataset = MSFAMatchedDataset(
        val_records, val_global, resolution=RESOLUTION, training=False
    )
    train_loader = _make_loader(
        train_dataset, batch_size, shuffle=True, workers=workers, seed=seed
    )
    val_loader = _make_loader(
        val_dataset, batch_size, shuffle=False, workers=workers, seed=seed
    )

    model = DC2MSFAClassifier(local_model, int(train_global.shape[1])).to(device)
    if trainable_parameter_count(model) != sum(
        parameter.numel() for parameter in model.global_projection.parameters()
    ):
        raise RuntimeError("DC2c hanya boleh melatih global projection")

    # Zero projection must reproduce the transferred local-only checkpoint.
    initial_metrics = _evaluate_msfa(model, val_loader, device, len(names))
    for key in ("macro_f1", "bottom3_f1", "worst_f1"):
        if abs(float(initial_metrics[key]) - float(replay_metrics[key])) > 1e-6:
            raise RuntimeError(f"Zero-init MSFA tidak identik dengan local-only untuk {key}")

    optimizer = torch.optim.AdamW(
        model.global_projection.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    criterion = nn.CrossEntropyLoss()

    run_dir = output_root / f"msfa_p5_gap_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_macro = -1.0
    best_epoch = -1
    history = []
    _seed_everything(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, global_features, labels in train_loader:
            images = images.to(device, non_blocking=True)
            global_features = global_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, global_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        scheduler.step()
        metrics = _evaluate_msfa(model, val_loader, device, len(names))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{key: value for key, value in metrics.items() if key != "per_class_f1"},
        }
        history.append(row)
        print(
            f"DC2c MSFA epoch={epoch:02d}/{epochs} loss={row['train_loss']:.4f} "
            f"macro={metrics['macro_f1']:.4f} bottom3={metrics['bottom3_f1']:.4f} "
            f"worst={metrics['worst_f1']:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_macro:
            best_macro = float(metrics["macro_f1"])
            best_epoch = epoch
            torch.save(
                {
                    "projection": model.global_projection.state_dict(),
                    "seed": seed,
                    "global_level": GLOBAL_LEVEL,
                    "global_dim": int(train_global.shape[1]),
                    "local_dim": model.local_dim,
                    "resolution": RESOLUTION,
                    "protocol": PROTOCOL,
                },
                best_path,
            )

    payload = torch.load(best_path, map_location=device, weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("Checkpoint MSFA protocol mismatch")
    model.global_projection.load_state_dict(payload["projection"], strict=True)
    best_metrics = _evaluate_msfa(model, val_loader, device, len(names))
    result = {
        "best_epoch": best_epoch,
        "trainable_parameters": trainable_parameter_count(model),
        "global_dim": int(train_global.shape[1]),
        "local_dim": model.local_dim,
        "checkpoint": str(best_path),
        "metrics": _metric_view(best_metrics, names),
        "history": history,
    }
    return _metric_view(replay_metrics, names), result


def run_dc2_msfa_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    detector_checkpoint: str | Path,
    dc2b_summary: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    epochs: int = 20,
    batch_size: int = 64,
    workers: int = 2,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("DC2c screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")

    data_root = Path(data_root).expanduser().resolve()
    detector_checkpoint = Path(detector_checkpoint).expanduser().resolve()
    dc2b_summary_path = Path(dc2b_summary).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("DC2c menolak dataset yang mengekspos test")
    audit = audit_dataset(data_root, output_root / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    dc2b = _load_json(dc2b_summary_path, "DC2b summary")
    if (
        dc2b.get("protocol") != DC2B_PROTOCOL
        or dc2b.get("decision") != "PASS"
        or dc2b.get("next_action") != "AUTHORIZE_DC2_GLOBAL_LOCAL_MSFA_SCREENING"
        or dc2b.get("test_images_accessed") is not False
        or int(dc2b.get("resolution", -1)) != RESOLUTION
    ):
        raise RuntimeError("DC2b belum mengotorisasi MSFA screening")

    detector_hash = _sha256_file(detector_checkpoint)
    if dc2b.get("detector_checkpoint_sha256") != detector_hash:
        raise RuntimeError("Detector DC2c berbeda dari detector DC2b")

    dc2b_root = dc2b_summary_path.parent
    local_checkpoint = Path(
        dc2b["results"]["predicted_crop_local"]["checkpoint"]
    ).expanduser()
    if not local_checkpoint.is_file():
        local_checkpoint = dc2b_root / f"predicted_crop{RESOLUTION}_seed{seed}" / "best.pt"
    if not local_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint local DC2b tidak ditemukan: {local_checkpoint}")

    train_records, names, train_meta = collect_predicted_crop_records(
        detector_checkpoint,
        data_root,
        "train",
        dc2b_root / "cache/predicted_train.json",
        device=device,
    )
    val_records, val_names, val_meta = collect_predicted_crop_records(
        detector_checkpoint,
        data_root,
        "val",
        dc2b_root / "cache/predicted_val.json",
        device=device,
    )
    if names != val_names or len(names) != 21:
        raise RuntimeError("Ontology DC2c tidak konsisten")
    if float(train_meta["matched_coverage"]) < 0.90 or float(val_meta["matched_coverage"]) < 0.90:
        raise RuntimeError("Coverage DC2b di bawah minimum DC2c")

    if not torch.cuda.is_available() and str(device) != "cpu":
        raise RuntimeError("GPU tidak tersedia")
    torch_device = torch.device("cpu" if str(device) == "cpu" else f"cuda:{device}")

    train_global, train_global_meta = extract_p5_global_descriptors(
        detector_checkpoint,
        train_records,
        output_root / "cache/p5_global_train.npz",
        detector_sha256=detector_hash,
        device=torch_device,
    )
    val_global, val_global_meta = extract_p5_global_descriptors(
        detector_checkpoint,
        val_records,
        output_root / "cache/p5_global_val.npz",
        detector_sha256=detector_hash,
        device=torch_device,
    )
    if train_global.shape[1] != val_global.shape[1]:
        raise RuntimeError("Dimensi global train/val berbeda")

    replay_metrics, msfa_result = _train_msfa_projection(
        local_checkpoint,
        train_records,
        val_records,
        train_global,
        val_global,
        names,
        output_root,
        seed=seed,
        device=torch_device,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    dc2b_metrics = dc2b["results"]["predicted_crop_local"]["metrics"]
    decision = decide_dc2_msfa(
        train_coverage=float(train_meta["matched_coverage"]),
        val_coverage=float(val_meta["matched_coverage"]),
        replay_metrics=replay_metrics,
        dc2b_metrics=dc2b_metrics,
        msfa_metrics=msfa_result["metrics"],
        global_feature_std=float(val_global_meta["feature_std"]),
    )
    payload = {
        "protocol": PROTOCOL,
        "stage": "broad_search_msfa_mechanism_screen",
        "seed": seed,
        "evaluation_split": "development_val_detector_matched_targets",
        "test_images_accessed": False,
        "test_opened": False,
        "paper_operator": "Fl = Fl + GAP(Fg)",
        "transfer_scope": "final-stage additive MSFA using frozen detector P5 and frozen DC2b local stream",
        "literal_full_dc2_reproduction": False,
        "detector_checkpoint": str(detector_checkpoint),
        "detector_checkpoint_sha256": detector_hash,
        "dc2b_summary": str(dc2b_summary_path),
        "local_checkpoint": str(local_checkpoint),
        "resolution": RESOLUTION,
        "global_level": GLOBAL_LEVEL,
        "pooling": "GAP",
        "coverage": {
            "train": float(train_meta["matched_coverage"]),
            "val": float(val_meta["matched_coverage"]),
        },
        "global_cache": {
            "train": train_global_meta,
            "val": val_global_meta,
        },
        "results": {
            "dc2b_local_replay": replay_metrics,
            "dc2b_reported": dc2b_metrics,
            "msfa": msfa_result,
        },
        **decision,
    }
    summary_path = output_root / "dc2_msfa_screening.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 DC2 MSFA mechanism screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--dc2b-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_dc2_msfa_screening(
        args.data_root,
        args.grouped_summary,
        args.detector_checkpoint,
        args.dc2b_summary,
        args.output_root,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        authorize_training=args.authorize_training,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
