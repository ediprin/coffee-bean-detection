"""Frozen-detector global/local feature aggregation screen inspired by DC2 MSFA."""

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
    MatchedCropGlobalDataset,
    extract_global_descriptors,
)
from coffee_detector.dc2_crop.predicted import collect_predicted_crop_records
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


@torch.inference_mode()
def _predict(model: DC2MSFAClassifier, loader, device: torch.device, *, enable_global: bool):
    model.eval()
    logits, labels = [], []
    for images, global_features, target in loader:
        images = images.to(device, non_blocking=True)
        global_features = global_features.to(device, non_blocking=True)
        logits.append(
            model(images, global_features, enable_global=enable_global).cpu()
        )
        labels.append(target.cpu())
    if not logits:
        raise RuntimeError("DataLoader kosong")
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def _train_arm(
    name: str,
    train_dataset: MatchedCropGlobalDataset,
    val_dataset: MatchedCropGlobalDataset,
    local_checkpoint: Path,
    num_classes: int,
    global_dim: int,
    output_root: Path,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    weight_decay: float,
    enable_global: bool,
) -> dict:
    run_dir = output_root / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    if best_path.is_file() and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    _seed_everything(seed)
    train_loader = _make_loader(
        train_dataset, batch_size, shuffle=True, workers=workers, seed=seed
    )
    val_loader = _make_loader(
        val_dataset, batch_size, shuffle=False, workers=workers, seed=seed
    )
    model = DC2MSFAClassifier(
        num_classes, global_dim, imagenet_pretrained=False
    )
    model.load_local_checkpoint(local_checkpoint)
    model = model.to(device)
    parameters = (
        model.parameters() if enable_global else model.local_model.parameters()
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    criterion = nn.CrossEntropyLoss()

    best_macro = -1.0
    best_epoch = -1
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, global_features, labels in train_loader:
            images = images.to(device, non_blocking=True)
            global_features = global_features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, global_features, enable_global=enable_global)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        scheduler.step()
        logits, labels = _predict(
            model, val_loader, device, enable_global=enable_global
        )
        metrics = classification_summary(logits, labels, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{key: value for key, value in metrics.items() if key != "per_class_f1"},
        }
        history.append(row)
        print(
            f"DC2c {name:8s} epoch={epoch:02d}/{epochs} loss={row['train_loss']:.4f} "
            f"macro={metrics['macro_f1']:.4f} bottom3={metrics['bottom3_f1']:.4f} "
            f"worst={metrics['worst_f1']:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "num_classes": num_classes,
                    "global_dim": global_dim,
                    "enable_global": enable_global,
                },
                best_path,
            )

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    logits, labels = _predict(model, val_loader, device, enable_global=enable_global)
    metrics = classification_summary(logits, labels, num_classes)
    metrics["per_class"] = {
        str(index): float(value)
        for index, value in enumerate(metrics.pop("per_class_f1"))
    }
    result = {
        "arm": name,
        "seed": seed,
        "enable_global": enable_global,
        "best_epoch": best_epoch,
        "train_instances": len(train_dataset),
        "val_instances": len(val_dataset),
        "metrics": metrics,
        "history": history,
        "checkpoint": str(best_path),
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def decide_dc2_msfa(local_ft: dict, msfa: dict) -> dict:
    local_metrics = local_ft["metrics"]
    msfa_metrics = msfa["metrics"]
    deltas = {
        metric: float(msfa_metrics[metric] - local_metrics[metric])
        for metric in ("macro_f1", "bottom3_f1", "worst_f1")
    }
    criteria = {
        "macro_gain_vs_local_ft_at_least_0_5_point": deltas["macro_f1"] >= 0.005,
        "bottom3_not_lower_than_local_ft": deltas["bottom3_f1"] >= 0.0,
        "worst_drop_vs_local_ft_no_more_than_1_point": deltas["worst_f1"] >= -0.01,
    }
    passed = all(criteria.values())
    return {
        "deltas_msfa_vs_local_ft": deltas,
        "criteria": criteria,
        "decision": "PASS" if passed else "FAIL",
        "next_action": (
            "AUTHORIZE_DC2_END_TO_END_INTEGRATION"
            if passed
            else "STOP_MSFA_ESCALATION_WITHOUT_END_TO_END_CLAIM"
        ),
    }


def run_dc2_msfa_screening(
    data_root: str | Path,
    grouped_summary: str | Path,
    detector_checkpoint: str | Path,
    dc2b_summary: str | Path,
    output_root: str | Path,
    *,
    seed: int = 42,
    device: str = "0",
    epochs: int = 10,
    batch_size: int = 64,
    workers: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    authorize_training: bool = False,
) -> dict:
    if seed != 42:
        raise ValueError("DC2 MSFA screening dikunci untuk seed 42")
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

    dc2b = _load_json(dc2b_summary, "DC2b summary")
    if dc2b.get("decision") != "PASS" or dc2b.get("test_images_accessed") is not False:
        raise RuntimeError("DC2b belum PASS; MSFA belum diotorisasi")
    resolution = int(dc2b["resolution_from_dc2a"])
    local_checkpoint = Path(
        dc2b["results"]["predicted_crop_local"]["checkpoint"]
    ).expanduser().resolve()
    if not local_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint local DC2b tidak ditemukan: {local_checkpoint}")

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
        raise RuntimeError("Ontology predicted crop tidak konsisten")
    if float(train_meta["matched_coverage"]) < 0.90 or float(val_meta["matched_coverage"]) < 0.90:
        raise RuntimeError("Coverage predicted crop jatuh di bawah prerequisite 90%")

    train_global, train_global_meta = extract_global_descriptors(
        detector_checkpoint,
        train_records,
        output_root / "cache/global_train.npz",
        split="train",
        device=device,
    )
    val_global, val_global_meta = extract_global_descriptors(
        detector_checkpoint,
        val_records,
        output_root / "cache/global_val.npz",
        split="val",
        device=device,
    )
    if train_global.shape[1] != val_global.shape[1]:
        raise RuntimeError("Dimensi global train/val berbeda")
    global_dim = int(train_global.shape[1])
    train_dataset = MatchedCropGlobalDataset(
        train_records, train_global, resolution, training=True
    )
    val_dataset = MatchedCropGlobalDataset(
        val_records, val_global, resolution, training=False
    )

    torch_device = torch.device("cpu" if str(device) == "cpu" else f"cuda:{device}")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU tidak tersedia")

    # Static preservation audit: zero MSFA projection must exactly preserve the
    # local-stream checkpoint before either arm receives extra optimization.
    probe = DC2MSFAClassifier(len(names), global_dim, imagenet_pretrained=False)
    probe.load_local_checkpoint(local_checkpoint)
    probe.eval()
    sample_image, sample_global, _ = val_dataset[0]
    with torch.inference_mode():
        local_logits = probe(
            sample_image.unsqueeze(0), sample_global.unsqueeze(0), enable_global=False
        )
        zero_msfa_logits = probe(
            sample_image.unsqueeze(0), sample_global.unsqueeze(0), enable_global=True
        )
    zero_preserves_local = bool(torch.equal(local_logits, zero_msfa_logits))
    if not zero_preserves_local:
        raise RuntimeError("Zero-init MSFA tidak mempertahankan local classifier")

    local_ft = _train_arm(
        "LOCAL_FT",
        train_dataset,
        val_dataset,
        local_checkpoint,
        len(names),
        global_dim,
        output_root,
        seed=seed,
        device=torch_device,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        enable_global=False,
    )
    msfa = _train_arm(
        "MSFA",
        train_dataset,
        val_dataset,
        local_checkpoint,
        len(names),
        global_dim,
        output_root,
        seed=seed,
        device=torch_device,
        epochs=epochs,
        batch_size=batch_size,
        workers=workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        enable_global=True,
    )
    decision = decide_dc2_msfa(local_ft, msfa)
    payload = {
        "protocol": "faruq-v3-dc2-msfa-global-local-screening-v1",
        "stage": "broad_search_frozen_detector_msfa",
        "seed": seed,
        "evaluation_split": "val_detector_matched_targets",
        "test_images_accessed": False,
        "test_opened": False,
        "paper_equation_transferred": "Fl = Fl + tau(Fg) (DC2 Eq. 8)",
        "adaptation_boundary": (
            "YOLO26 P3/P4/P5 predicted-box ROI GAP descriptors are concatenated, "
            "projected to the terminal MobileNetV3 local descriptor, then added residually; "
            "this is not literal stage-paired DC2 MSFA and the detector is frozen."
        ),
        "resolution_from_dc2b": resolution,
        "local_checkpoint": str(local_checkpoint),
        "global_levels": ["P3", "P4", "P5"],
        "global_dimensions": global_dim,
        "global_cache": {"train": train_global_meta, "val": val_global_meta},
        "static_gates": {"zero_msfa_preserves_local_logits_bitwise": zero_preserves_local},
        "results": {"LOCAL_FT": local_ft, "MSFA": msfa},
        **decision,
    }
    summary = output_root / "dc2_msfa_global_local_screening.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 DC2 global/local MSFA screen")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--dc2b-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=10)
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
