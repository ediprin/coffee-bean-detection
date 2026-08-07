"""Broad-search raw RGB object-crop resolution screen inspired by DC2."""

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
from coffee_detector.dc2_crop.data import RawObjectCropDataset, collect_crop_records
from coffee_detector.dc2_crop.metrics import classification_summary
from coffee_detector.dc2_crop.model import build_local_classifier, predict_logits, trainable_parameter_count
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary


RESOLUTIONS = (32, 64, 128, 224)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def _train_one_resolution(
    train_records,
    val_records,
    names: dict[int, str],
    resolution: int,
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
    run_dir = output_root / f"crop{resolution}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    if summary_path.is_file() and best_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    _seed_everything(seed)
    train_dataset = RawObjectCropDataset(train_records, resolution, training=True, context=1.0)
    val_dataset = RawObjectCropDataset(val_records, resolution, training=False, context=1.0)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_macro = -1.0
    best_epoch = -1
    history = []
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
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{key: value for key, value in metrics.items() if key != "per_class_f1"},
        }
        history.append(epoch_row)
        print(
            f"crop={resolution:3d} epoch={epoch:02d}/{epochs} "
            f"loss={epoch_row['train_loss']:.4f} macro={metrics['macro_f1']:.4f} "
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
        names[index]: float(value) for index, value in enumerate(metrics.pop("per_class_f1"))
    }
    result = {
        "resolution": resolution,
        "seed": seed,
        "best_epoch": best_epoch,
        "train_instances": len(train_records),
        "val_instances": len(val_records),
        "trainable_parameters": trainable_parameter_count(model),
        "metrics": metrics,
        "history": history,
        "checkpoint": str(best_path),
    }
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def run_dc2_crop_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
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
        raise ValueError("DC2 crop discovery screening dikunci untuk seed 42")
    if not authorize_training:
        raise RuntimeError("Gunakan --authorize-training setelah protocol/CI dibekukan")
    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    load_faruq_grouped_summary(grouped_summary, data_root)
    if (data_root / "test").exists():
        raise RuntimeError("Faruq-v3 development tidak boleh memiliki split test")
    audit = audit_dataset(data_root, output_root / "dataset_audit.json", near_threshold=-1)
    if not audit["safe_for_training"]:
        raise RuntimeError("Audit dataset gagal")

    train_records, names = collect_crop_records(data_root, "train")
    val_records, val_names = collect_crop_records(data_root, "val")
    if names != val_names or len(names) != 21:
        raise RuntimeError("Nama/jumlah kelas crop tidak konsisten dengan SNI-21")
    val_present = {record.class_id for record in val_records}
    if val_present != set(names):
        raise RuntimeError(f"Validation crop kehilangan kelas: {sorted(set(names) - val_present)}")

    if not torch.cuda.is_available() and str(device) != "cpu":
        raise RuntimeError("GPU tidak tersedia")
    torch_device = torch.device("cpu" if str(device) == "cpu" else f"cuda:{device}")

    results = {}
    for resolution in RESOLUTIONS:
        results[str(resolution)] = _train_one_resolution(
            train_records,
            val_records,
            names,
            resolution,
            output_root,
            seed=seed,
            device=torch_device,
            epochs=epochs,
            batch_size=batch_size,
            workers=workers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    metrics_by_resolution = {
        int(key): value["metrics"] for key, value in results.items()
    }
    best_resolution = max(
        metrics_by_resolution,
        key=lambda value: metrics_by_resolution[value]["macro_f1"],
    )
    baseline32 = metrics_by_resolution[32]
    best = metrics_by_resolution[best_resolution]
    deltas_vs_32 = {
        metric: float(best[metric] - baseline32[metric])
        for metric in ("macro_f1", "bottom3_f1", "worst_f1")
    }
    resolution_signal = (
        best_resolution != 32
        and deltas_vs_32["macro_f1"] >= 0.02
        and deltas_vs_32["bottom3_f1"] >= 0.0
    )
    # This is a mechanistic diagnostic, not detector mAP. It authorizes a real
    # predicted-crop/local-stream integration only if resolution clearly helps.
    decision = "RETAIN_DC2_LOCAL_STREAM" if resolution_signal else "WEAK_RAW_RESOLUTION_SIGNAL"
    payload = {
        "protocol": "faruq-v3-dc2-raw-crop-resolution-search-v1",
        "stage": "broad_search_mechanistic_screen",
        "seed": seed,
        "evaluation_split": "val_gt_crops",
        "test_images_accessed": False,
        "test_opened": False,
        "resolutions": list(RESOLUTIONS),
        "local_backbone": "torchvision_mobilenet_v3_small_imagenet",
        "context_factor": 1.0,
        "results": results,
        "best_resolution": best_resolution,
        "best_metrics": best,
        "deltas_best_vs_32": deltas_vs_32,
        "criteria": {
            "best_not_32": best_resolution != 32,
            "macro_gain_vs_32_at_least_2_points": deltas_vs_32["macro_f1"] >= 0.02,
            "bottom3_not_lower_than_32": deltas_vs_32["bottom3_f1"] >= 0.0,
        },
        "decision": decision,
        "next_action": (
            "AUTHORIZE_PREDICTED_RAW_CROP_LOCAL_STREAM_INTEGRATION"
            if resolution_signal
            else "CONTINUE_BROAD_SEARCH_WITHOUT_CLAIMING_DC2_RESOLUTION_BOTTLENECK"
        ),
    }
    summary = output_root / "dc2_raw_crop_resolution_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 DC2 raw-crop resolution screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--authorize-training", action="store_true")
    args = parser.parse_args()
    result = run_dc2_crop_screening(
        args.data_root,
        args.grouped_summary,
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
