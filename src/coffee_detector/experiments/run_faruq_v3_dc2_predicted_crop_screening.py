"""Validation-only predicted raw-RGB crop local-stream screen inspired by DC2."""

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
from coffee_detector.dc2_crop.model import build_local_classifier, predict_logits, trainable_parameter_count
from coffee_detector.dc2_crop.predicted import (
    MatchedRawObjectCropDataset,
    PredictedCropRecord,
    collect_predicted_crop_records,
)
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_json(path: str | Path, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    return json.loads(source.read_text(encoding="utf-8"))


def _make_loader(dataset, batch_size: int, *, shuffle: bool, workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=workers > 0,
    )


def _native_detector_summary(records: list[PredictedCropRecord], num_classes: int) -> dict:
    labels = torch.tensor([record.class_id for record in records], dtype=torch.long)
    predictions = torch.tensor(
        [record.predicted_class_id for record in records], dtype=torch.long
    )
    if bool(((predictions < 0) | (predictions >= num_classes)).any()):
        raise RuntimeError("Prediksi detector mengandung class id di luar ontology")
    logits = torch.full((len(records), num_classes), -1.0, dtype=torch.float32)
    logits[torch.arange(len(records)), predictions] = 1.0
    return classification_summary(logits, labels, num_classes)


def _train_local_arm(
    train_records: list[PredictedCropRecord],
    val_records: list[PredictedCropRecord],
    names: dict[int, str],
    resolution: int,
    source: str,
    output_root: Path,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    weight_decay: float,
) -> dict:
    run_dir = output_root / f"{source}_crop{resolution}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    if summary_path.is_file() and best_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    _seed_everything(seed)
    train_dataset = MatchedRawObjectCropDataset(
        train_records, resolution, training=True, source=source, context=1.0
    )
    val_dataset = MatchedRawObjectCropDataset(
        val_records, resolution, training=False, source=source, context=1.0
    )
    train_loader = _make_loader(
        train_dataset, batch_size, shuffle=True, workers=workers, seed=seed
    )
    val_loader = _make_loader(
        val_dataset, batch_size, shuffle=False, workers=workers, seed=seed
    )

    model = build_local_classifier(len(names), imagenet_pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    best_macro = -1.0
    best_epoch = -1
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        scheduler.step()

        logits, labels = predict_logits(model, val_loader, device)
        metrics = classification_summary(logits, labels, len(names))
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{key: value for key, value in metrics.items() if key != "per_class_f1"},
        }
        history.append(row)
        print(
            f"DC2b {source:9s} crop={resolution} epoch={epoch:02d}/{epochs} "
            f"loss={row['train_loss']:.4f} macro={metrics['macro_f1']:.4f} "
            f"bottom3={metrics['bottom3_f1']:.4f} worst={metrics['worst_f1']:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "resolution": resolution,
                    "source": source,
                    "seed": seed,
                    "num_classes": len(names),
                    "class_names": names,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    logits, labels = predict_logits(model, val_loader, device)
    metrics = classification_summary(logits, labels, len(names))
    metrics["per_class"] = {
        names[index]: float(value)
        for index, value in enumerate(metrics.pop("per_class_f1"))
    }
    result = {
        "source": source,
        "resolution": int(resolution),
        "seed": seed,
        "best_epoch": best_epoch,
        "train_instances": len(train_records),
        "val_instances": len(val_records),
        "trainable_parameters": trainable_parameter_count(model),
        "metrics": metrics,
        "history": history,
        "checkpoint": str(best_path),
    }
    summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def decide_dc2_predicted(
    *,
    train_coverage: float,
    val_coverage: float,
    detector_metrics: dict,
    predicted_metrics: dict,
    gt_matched_metrics: dict,
) -> dict:
    macro_gain = float(predicted_metrics["macro_f1"] - detector_metrics["macro_f1"])
    bottom3_gain = float(
        predicted_metrics["bottom3_f1"] - detector_metrics["bottom3_f1"]
    )
    worst_gain = float(predicted_metrics["worst_f1"] - detector_metrics["worst_f1"])
    macro_retention = float(
        predicted_metrics["macro_f1"] / max(gt_matched_metrics["macro_f1"], 1e-12)
    )
    bottom3_retention = float(
        predicted_metrics["bottom3_f1"]
        / max(gt_matched_metrics["bottom3_f1"], 1e-12)
    )
    criteria = {
        "train_matched_coverage_at_least_90_percent": train_coverage >= 0.90,
        "val_matched_coverage_at_least_90_percent": val_coverage >= 0.90,
        "predicted_local_macro_gain_vs_native_at_least_1_point": macro_gain >= 0.01,
        "predicted_local_bottom3_not_lower_than_native": bottom3_gain >= 0.0,
        "predicted_local_worst_drop_vs_native_no_more_than_5_points": worst_gain >= -0.05,
        "predicted_crop_macro_retains_90_percent_of_gt_matched": macro_retention >= 0.90,
        "predicted_crop_bottom3_retains_80_percent_of_gt_matched": bottom3_retention >= 0.80,
    }
    passed = all(criteria.values())
    return {
        "deltas_vs_native_matched": {
            "macro_f1": macro_gain,
            "bottom3_f1": bottom3_gain,
            "worst_f1": worst_gain,
        },
        "retention_vs_gt_matched": {
            "macro_f1_ratio": macro_retention,
            "bottom3_f1_ratio": bottom3_retention,
        },
        "criteria": criteria,
        "decision": "PASS" if passed else "FAIL",
        "next_action": (
            "AUTHORIZE_DC2_GLOBAL_LOCAL_MSFA_SCREENING"
            if passed
            else "DO_NOT_ESCALATE_DC2_TO_MSFA_FROM_THIS_ARM"
        ),
    }


def run_dc2_predicted_crop_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    detector_checkpoint: str | Path,
    raw_crop_summary: str | Path,
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
        raise ValueError("DC2 predicted-crop screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")

    data_root = Path(data_root).expanduser().resolve()
    detector_checkpoint = Path(detector_checkpoint).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki split test")
    audit = audit_dataset(data_root, output_root / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    raw = _load_json(raw_crop_summary, "DC2a raw-crop summary")
    if (
        raw.get("decision") != "RETAIN_DC2_LOCAL_STREAM"
        or raw.get("test_images_accessed") is not False
    ):
        raise RuntimeError("DC2a belum mengotorisasi predicted raw-crop integration")
    resolution = int(raw["best_resolution"])
    if resolution not in {32, 64, 128, 224}:
        raise RuntimeError("Resolusi DC2a tidak termasuk arm predeclared")

    train_records, names, train_meta = collect_predicted_crop_records(
        detector_checkpoint,
        data_root,
        "train",
        output_root / "cache/predicted_train.json",
        device=device,
    )
    val_records, val_names, val_meta = collect_predicted_crop_records(
        detector_checkpoint,
        data_root,
        "val",
        output_root / "cache/predicted_val.json",
        device=device,
    )
    if names != val_names or len(names) != 21:
        raise RuntimeError("Nama/jumlah kelas predicted crop tidak konsisten dengan SNI-21")
    if {record.class_id for record in val_records} != set(names):
        raise RuntimeError("Validation matched predicted crops kehilangan setidaknya satu kelas")

    if not torch.cuda.is_available() and str(device) != "cpu":
        raise RuntimeError("GPU tidak tersedia")
    torch_device = torch.device("cpu" if str(device) == "cpu" else f"cuda:{device}")

    detector_metrics = _native_detector_summary(val_records, len(names))
    detector_metrics["per_class"] = {
        names[index]: float(value)
        for index, value in enumerate(detector_metrics.pop("per_class_f1"))
    }
    gt_matched = _train_local_arm(
        train_records,
        val_records,
        names,
        resolution,
        "gt",
        output_root,
        seed=seed,
        device=torch_device,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    predicted = _train_local_arm(
        train_records,
        val_records,
        names,
        resolution,
        "predicted",
        output_root,
        seed=seed,
        device=torch_device,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    decision = decide_dc2_predicted(
        train_coverage=float(train_meta["matched_coverage"]),
        val_coverage=float(val_meta["matched_coverage"]),
        detector_metrics=detector_metrics,
        predicted_metrics=predicted["metrics"],
        gt_matched_metrics=gt_matched["metrics"],
    )
    payload = {
        "protocol": "faruq-v3-dc2-predicted-raw-crop-screening-v1",
        "stage": "broad_search_predicted_crop_screen",
        "seed": seed,
        "evaluation_split": "val_detector_matched_targets",
        "test_images_accessed": False,
        "test_opened": False,
        "detector_checkpoint": str(detector_checkpoint),
        "detector_checkpoint_sha256": val_meta["checkpoint_sha256"],
        "resolution_from_dc2a": resolution,
        "crop_context_factor": 1.0,
        "match_iou": float(val_meta["match_iou"]),
        "coverage": {
            "train": float(train_meta["matched_coverage"]),
            "val": float(val_meta["matched_coverage"]),
            "train_mean_iou": float(train_meta["mean_matched_iou"]),
            "val_mean_iou": float(val_meta["mean_matched_iou"]),
        },
        "results": {
            "native_detector_on_matched_val": detector_metrics,
            "gt_crop_local_on_same_matched_objects": gt_matched,
            "predicted_crop_local": predicted,
        },
        **decision,
    }
    summary = output_root / "dc2_predicted_raw_crop_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 DC2 predicted raw-crop screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--raw-crop-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_dc2_predicted_crop_screening(
        args.data_root,
        args.grouped_summary,
        args.detector_checkpoint,
        args.raw_crop_summary,
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
