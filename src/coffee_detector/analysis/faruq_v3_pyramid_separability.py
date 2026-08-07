from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _letterbox_sample,
    _split_samples,
    _unwrap_head,
)


REPRESENTATIONS = {
    "P3": ("P3",),
    "P4": ("P4",),
    "P5": ("P5",),
    "P3+P4": ("P3", "P4"),
    "P3+P4+P5": ("P3", "P4", "P5"),
}


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_signature(samples: Iterable[tuple[Path, tuple]]) -> str:
    digest = hashlib.sha256()
    for image_path, annotations in samples:
        digest.update(image_path.name.encode("utf-8"))
        digest.update(str(image_path.stat().st_size).encode("ascii"))
        for item in annotations:
            digest.update(
                (
                    f"{item.class_id}:{item.x_center:.8f}:{item.y_center:.8f}:"
                    f"{item.width:.8f}:{item.height:.8f}"
                ).encode("ascii")
            )
    return digest.hexdigest()


def _pyramid_spec(network: torch.nn.Module) -> list[tuple[str, int, float]]:
    head = _unwrap_head(network)
    indices = list(head.f) if isinstance(head.f, (tuple, list)) else [int(head.f)]
    strides = [float(value) for value in head.stride.detach().cpu().tolist()]
    if len(indices) != len(strides):
        raise ValueError("Jumlah input Detect dan stride tidak cocok")
    output = []
    for index, stride in zip(indices, strides):
        level = int(round(math.log2(stride)))
        if not math.isclose(stride, 2**level):
            raise ValueError(f"Stride bukan pangkat dua: {stride}")
        output.append((f"P{level}", int(index), stride))
    names = [item[0] for item in output]
    if names != ["P3", "P4", "P5"]:
        raise ValueError(f"D0 harus menyediakan P3-P5, diterima {names}")
    return output


def _roi_descriptor(
    feature: torch.Tensor,
    boxes: torch.Tensor,
    image_size: int,
    roi_size: int,
) -> torch.Tensor:
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f"Feature map harus [1,C,H,W], diterima {tuple(feature.shape)}")
    batch_column = boxes.new_zeros((len(boxes), 1))
    rois = torch.cat((batch_column, boxes), dim=1)
    aligned = roi_align(
        feature,
        rois,
        output_size=(roi_size, roi_size),
        spatial_scale=float(feature.shape[-1]) / float(image_size),
        sampling_ratio=2,
        aligned=True,
    )
    return torch.cat(
        (aligned.mean(dim=(-2, -1)), aligned.amax(dim=(-2, -1))), dim=1
    )


def _extract_split(
    network: torch.nn.Module,
    data_root: Path,
    split: str,
    cache_path: Path,
    *,
    checkpoint_sha256: str,
    device: torch.device,
    image_size: int,
    roi_size: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    layout, samples = _split_samples(data_root, split)
    if "test" in layout.splits:
        raise RuntimeError("Pyramid audit menolak dataset yang mengekspos test")
    signature = _sample_signature(samples)
    expected = {
        "protocol": "faruq-v3-pyramid-feature-cache-v1",
        "split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_signature": signature,
        "sample_count": len(samples),
        "image_size": image_size,
        "roi_size": roi_size,
    }
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            with np.load(cache_path, allow_pickle=False) as payload:
                labels = payload["labels"].astype(np.int64, copy=False)
                features = {
                    name: payload[name].astype(np.float32, copy=False)
                    for name in metadata["feature_names"]
                }
            print(f"CACHE {split}: {len(labels)} instances", flush=True)
            return features, labels, metadata

    spec = _pyramid_spec(network)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, index, _ in spec:
        def capture(_module, _inputs, output, *, feature_name=name):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Output {feature_name} bukan tensor")
            captured[feature_name] = output.detach()

        handles.append(network.model[index].register_forward_hook(capture))

    feature_rows: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in spec}
    label_rows: list[np.ndarray] = []
    try:
        with torch.inference_mode():
            for image_index, (image_path, annotations) in enumerate(samples, 1):
                image, boxes, labels, _ = _letterbox_sample(
                    image_path, annotations, image_size, device
                )
                captured.clear()
                network(image)
                if set(captured) != set(feature_rows):
                    raise RuntimeError(
                        f"Feature hook tidak lengkap: {sorted(captured)}"
                    )
                for name in feature_rows:
                    descriptor = _roi_descriptor(
                        captured[name], boxes, image_size, roi_size
                    )
                    feature_rows[name].append(
                        descriptor.float().cpu().numpy().astype(np.float32)
                    )
                label_rows.append(labels.cpu().numpy().astype(np.int64))
                if image_index % 100 == 0 or image_index == len(samples):
                    total = sum(len(item) for item in label_rows)
                    print(
                        f"FEATURE {split} {image_index}/{len(samples)} | "
                        f"instances={total}",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    features = {
        name: np.concatenate(rows, axis=0) for name, rows in feature_rows.items()
    }
    labels = np.concatenate(label_rows, axis=0)
    if any(len(values) != len(labels) for values in features.values()):
        raise RuntimeError("Jumlah descriptor dan label tidak konsisten")
    metadata = {
        **expected,
        "feature_names": list(features),
        "instances": len(labels),
        "feature_dimensions": {
            name: int(values.shape[1]) for name, values in features.items()
        },
        "detector_training_executed": False,
        "test_images_accessed": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, labels=labels, **features)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return features, labels, metadata


def _prepare_features(
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train = (train - mean) / scale
    validation = (validation - mean) / scale
    train /= np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-8)
    validation /= np.maximum(
        np.linalg.norm(validation, axis=1, keepdims=True), 1e-8
    )
    return train.astype(np.float64), validation.astype(np.float64)


def fit_balanced_ridge_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    *,
    num_classes: int,
    ridge: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    if ridge <= 0:
        raise ValueError("ridge harus positif")
    if len(train_features) != len(train_labels):
        raise ValueError("Jumlah train feature dan label berbeda")
    train, validation = _prepare_features(train_features, validation_features)
    train = np.concatenate((train, np.ones((len(train), 1))), axis=1)
    validation = np.concatenate(
        (validation, np.ones((len(validation), 1))), axis=1
    )
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Semua kelas wajib hadir di train untuk linear probe")
    sample_weights = len(train_labels) / (num_classes * counts[train_labels])
    targets = np.eye(num_classes, dtype=np.float64)[train_labels]
    weighted = train * sample_weights[:, None]
    gram = train.T @ weighted
    gram.flat[:: gram.shape[0] + 1] += ridge
    coefficient = np.linalg.solve(gram, train.T @ (targets * sample_weights[:, None]))
    return train @ coefficient, validation @ coefficient


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    class_rows = []
    f1_values = []
    recalls = []
    for class_id in range(num_classes):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        support = int(confusion[class_id, :].sum())
        recall = true_positive / support if support else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        class_rows.append(
            {"class_id": class_id, "support": support, "f1": float(f1)}
        )
    order = np.argsort(logits, axis=1)[:, ::-1]
    top3 = float(np.mean(np.any(order[:, :3] == labels[:, None], axis=1)))
    sorted_f1 = np.sort(np.asarray(f1_values))
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "bottom3_f1": float(np.mean(sorted_f1[:3])),
        "worst_class_f1": float(sorted_f1[0]),
        "top3_accuracy": top3,
        "per_class": class_rows,
        "confusion": confusion.tolist(),
    }


def decide_pyramid_route(
    results: dict[str, dict[str, Any]],
    *,
    minimum_macro_f1: float = 0.75,
    minimum_bottom3_f1: float = 0.50,
    minimum_route_gain: float = 0.02,
) -> dict[str, Any]:
    singles = {name: results[name]["validation"] for name in ("P3", "P4", "P5")}
    fusions = {
        name: results[name]["validation"] for name in ("P3+P4", "P3+P4+P5")
    }
    best_single_name = max(singles, key=lambda name: singles[name]["macro_f1"])
    best_fusion_name = max(fusions, key=lambda name: fusions[name]["macro_f1"])
    best_name = max(
        results, key=lambda name: results[name]["validation"]["macro_f1"]
    )
    best = results[best_name]["validation"]
    best_single = singles[best_single_name]
    best_fusion = fusions[best_fusion_name]
    usable = (
        best["macro_f1"] >= minimum_macro_f1
        and best["bottom3_f1"] >= minimum_bottom3_f1
    )
    fusion_gain = best_fusion["macro_f1"] - best_single["macro_f1"]
    fusion_rational = (
        usable
        and fusion_gain >= minimum_route_gain
        and best_fusion["bottom3_f1"] >= best_single["bottom3_f1"]
    )
    deeper = max(singles["P4"], singles["P5"], key=lambda row: row["macro_f1"])
    p3_gain = singles["P3"]["macro_f1"] - deeper["macro_f1"]
    high_resolution_rational = (
        usable
        and p3_gain >= minimum_route_gain
        and singles["P3"]["bottom3_f1"] >= deeper["bottom3_f1"]
    )
    if not usable:
        action = "STOP_HEAD_ONLY_SEARCH_REPRESENTATION_OR_LABEL_LIMITED"
        decision = "FAIL"
    elif fusion_rational:
        action = "AUTHORIZE_MULTILEVEL_CLASSIFICATION_PROTOCOL"
        decision = "PASS"
    elif high_resolution_rational:
        action = "AUTHORIZE_HIGH_RES_CLASSIFICATION_PROTOCOL"
        decision = "PASS"
    else:
        action = "HEAD_LIMITED_WITHOUT_SCALE_SPECIFIC_ROUTE"
        decision = "INCONCLUSIVE"
    return {
        "decision": decision,
        "next_action": action,
        "best_representation": best_name,
        "best_single": best_single_name,
        "best_fusion": best_fusion_name,
        "fusion_macro_gain": float(fusion_gain),
        "p3_macro_gain_over_best_deeper": float(p3_gain),
        "criteria": {
            "usable_validation_signal": usable,
            "fusion_rational": fusion_rational,
            "high_resolution_rational": high_resolution_rational,
        },
        "thresholds": {
            "minimum_macro_f1": minimum_macro_f1,
            "minimum_bottom3_f1": minimum_bottom3_f1,
            "minimum_route_gain": minimum_route_gain,
        },
        "detector_training_authorized": False,
        "test_access_authorized": False,
    }


def run_faruq_v3_pyramid_separability(
    checkpoint: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    device: str = "0",
    image_size: int = 640,
    roi_size: int = 3,
    ridge: float = 0.01,
) -> dict[str, Any]:
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    layout, _ = _split_samples(data_root, "train")
    if "val" not in layout.splits or "test" in layout.splits:
        raise RuntimeError("Audit memerlukan train+val dan menolak test")
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    network = YOLO(str(checkpoint)).model.to(torch_device).eval()
    head = _unwrap_head(network)
    if int(head.nc) != len(layout.names):
        raise ValueError("Jumlah kelas checkpoint dan dataset berbeda")
    checkpoint_hash = _checkpoint_sha256(checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extracted = {}
    metadata = {}
    for split in ("train", "val"):
        split_features, split_labels, split_metadata = _extract_split(
            network,
            data_root,
            split,
            destination.parent / f"feature_cache_{split}.npz",
            checkpoint_sha256=checkpoint_hash,
            device=torch_device,
            image_size=image_size,
            roi_size=roi_size,
        )
        extracted[split] = (split_features, split_labels)
        metadata[split] = split_metadata

    train_features, train_labels = extracted["train"]
    val_features, val_labels = extracted["val"]
    results = {}
    for representation, levels in REPRESENTATIONS.items():
        train_matrix = np.concatenate([train_features[level] for level in levels], axis=1)
        val_matrix = np.concatenate([val_features[level] for level in levels], axis=1)
        train_logits, val_logits = fit_balanced_ridge_probe(
            train_matrix,
            train_labels,
            val_matrix,
            num_classes=len(layout.names),
            ridge=ridge,
        )
        train_metrics = classification_metrics(
            train_logits, train_labels, num_classes=len(layout.names)
        )
        val_metrics = classification_metrics(
            val_logits, val_labels, num_classes=len(layout.names)
        )
        for row in val_metrics["per_class"]:
            row["class_name"] = layout.names[int(row["class_id"])]
        results[representation] = {
            "levels": list(levels),
            "dimensions": int(train_matrix.shape[1]),
            "train": train_metrics,
            "validation": val_metrics,
            "macro_f1_generalization_gap": float(
                train_metrics["macro_f1"] - val_metrics["macro_f1"]
            ),
        }
        print(
            f"PROBE {representation}: train Macro={train_metrics['macro_f1']:.2%} | "
            f"val Macro={val_metrics['macro_f1']:.2%} | "
            f"bottom3={val_metrics['bottom3_f1']:.2%}",
            flush=True,
        )

    decision = decide_pyramid_route(results)
    payload = {
        "protocol": "faruq-v3-pyramid-separability-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_root": str(data_root),
        "detector_training_executed": False,
        "probe_fitting_executed": True,
        "validation_images_accessed": True,
        "test_images_accessed": False,
        "image_size": image_size,
        "roi_size": roi_size,
        "ridge": ridge,
        "cache": metadata,
        "results": results,
        "decision": decision,
    }
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen D0 P3-P5 ground-truth ROI separability audit"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--roi-size", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=0.01)
    args = parser.parse_args()
    result = run_faruq_v3_pyramid_separability(
        args.checkpoint,
        args.data_root,
        args.output,
        device=args.device,
        image_size=args.image_size,
        roi_size=args.roi_size,
        ridge=args.ridge,
    )
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
