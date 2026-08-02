from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _decode_branch,
    _greedy_match,
    _letterbox_sample,
    _raw_branches,
    _split_samples,
    _unwrap_head,
)
from coffee_detector.analysis.faruq_v3_pyramid_separability import (
    _checkpoint_sha256,
    _pyramid_spec,
    _roi_descriptor,
    _sample_signature,
    classification_metrics,
    fit_balanced_ridge_probe,
)


RAW_REPRESENTATIONS = {
    "P5_RAW": ("P5",),
    "P3+P4+P5_RAW": ("P3", "P4", "P5"),
}


def fit_pca_projection(
    train: np.ndarray,
    validation: np.ndarray,
    *,
    components: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if components <= 0:
        raise ValueError("components harus positif")
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != validation.shape[1]:
        raise ValueError("Train dan validation PCA harus berupa matriks kompatibel")
    if components > min(train.shape):
        raise ValueError(
            f"components={components} melebihi rank maksimum {min(train.shape)}"
        )
    mean = train.mean(axis=0, keepdims=True)
    centered_train = train - mean
    centered_validation = validation - mean
    _, singular_values, right = np.linalg.svd(
        centered_train.astype(np.float32), full_matrices=False
    )
    basis = right[:components].T
    projected_train = centered_train @ basis
    projected_validation = centered_validation @ basis
    variance = singular_values**2
    explained = float(variance[:components].sum() / max(variance.sum(), 1e-12))
    return (
        projected_train.astype(np.float32),
        projected_validation.astype(np.float32),
        {
            "input_dimensions": int(train.shape[1]),
            "output_dimensions": components,
            "explained_variance_fraction": explained,
            "fit_split": "train",
        },
    )


def _extract_predicted_split(
    network: torch.nn.Module,
    data_root: Path,
    split: str,
    cache_path: Path,
    *,
    checkpoint_sha256: str,
    device: torch.device,
    image_size: int,
    roi_size: int,
    candidate_count: int,
    iou_threshold: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    layout, samples = _split_samples(data_root, split)
    if "test" in layout.splits:
        raise RuntimeError("Predicted-ROI audit menolak dataset yang mengekspos test")
    signature = _sample_signature(samples)
    expected = {
        "protocol": "faruq-v3-predicted-roi-cache-v1",
        "split": split,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_signature": signature,
        "sample_count": len(samples),
        "image_size": image_size,
        "roi_size": roi_size,
        "candidate_count": candidate_count,
        "iou_threshold": iou_threshold,
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
            print(
                f"CACHE predicted {split}: {len(labels)}/{metadata['targets']} matched",
                flush=True,
            )
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
    targets = 0
    matched_ious: list[float] = []
    try:
        with torch.inference_mode():
            for image_index, (image_path, annotations) in enumerate(samples, 1):
                image, target_boxes, target_labels, _ = _letterbox_sample(
                    image_path, annotations, image_size, device
                )
                captured.clear()
                _, raw, decoded_head = _raw_branches(
                    network, image, candidate_count
                )
                boxes, scores = _decode_branch(decoded_head, raw["one2one"])
                confidence = scores.max(dim=1).values
                keep = confidence.topk(min(candidate_count, len(confidence))).indices
                candidate_boxes = boxes[keep]
                targets += len(target_boxes)
                matches = _greedy_match(
                    candidate_boxes, target_boxes, iou_threshold
                )
                if matches:
                    prediction_indices = torch.tensor(
                        [item[0] for item in matches],
                        device=device,
                        dtype=torch.long,
                    )
                    target_indices = torch.tensor(
                        [item[1] for item in matches],
                        device=device,
                        dtype=torch.long,
                    )
                    matched_boxes = candidate_boxes[prediction_indices]
                    for name in feature_rows:
                        descriptor = _roi_descriptor(
                            captured[name], matched_boxes, image_size, roi_size
                        )
                        feature_rows[name].append(
                            descriptor.float().cpu().numpy().astype(np.float32)
                        )
                    label_rows.append(
                        target_labels[target_indices].cpu().numpy().astype(np.int64)
                    )
                    matched_ious.extend(float(item[2]) for item in matches)
                if image_index % 100 == 0 or image_index == len(samples):
                    matched = sum(len(item) for item in label_rows)
                    print(
                        f"PREDICTED ROI {split} {image_index}/{len(samples)} | "
                        f"matched={matched}/{targets}",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    if not label_rows:
        raise RuntimeError(f"Tidak ada predicted ROI cocok pada split {split}")
    features = {
        name: np.concatenate(rows, axis=0) for name, rows in feature_rows.items()
    }
    labels = np.concatenate(label_rows, axis=0)
    if any(len(values) != len(labels) for values in features.values()):
        raise RuntimeError("Jumlah predicted descriptor dan label tidak konsisten")
    metadata = {
        **expected,
        "feature_names": list(features),
        "instances": len(labels),
        "targets": targets,
        "matched_recall": len(labels) / max(targets, 1),
        "mean_matched_iou": float(np.mean(matched_ious)),
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


def decide_predicted_roi_transfer(
    results: dict[str, dict[str, Any]],
    coverage: dict[str, float],
    gt_fusion_macro_f1: float,
    *,
    minimum_coverage: float = 0.90,
    minimum_gain: float = 0.02,
    minimum_macro_f1: float = 0.75,
    minimum_bottom3_f1: float = 0.50,
    minimum_gt_retention: float = 0.90,
) -> dict[str, Any]:
    raw_p5 = results["P5_RAW"]["validation"]
    raw_fusion = results["P3+P4+P5_RAW"]["validation"]
    cm_p5 = results["P5_CM128"]["validation"]
    cm_fusion = results["P3+P4+P5_CM128"]["validation"]
    raw_gain = raw_fusion["macro_f1"] - raw_p5["macro_f1"]
    cm_gain = cm_fusion["macro_f1"] - cm_p5["macro_f1"]
    retention = raw_fusion["macro_f1"] / max(gt_fusion_macro_f1, 1e-12)
    criteria = {
        "train_coverage_at_least_90_percent": coverage["train"] >= minimum_coverage,
        "validation_coverage_at_least_90_percent": coverage["val"] >= minimum_coverage,
        "raw_fusion_gain_at_least_2_points": raw_gain >= minimum_gain,
        "raw_bottom3_preserved": raw_fusion["bottom3_f1"] >= raw_p5["bottom3_f1"],
        "capacity_matched_gain_at_least_2_points": cm_gain >= minimum_gain,
        "capacity_matched_bottom3_preserved": (
            cm_fusion["bottom3_f1"] >= cm_p5["bottom3_f1"]
        ),
        "capacity_matched_macro_at_least_75_percent": (
            cm_fusion["macro_f1"] >= minimum_macro_f1
        ),
        "capacity_matched_bottom3_at_least_50_percent": (
            cm_fusion["bottom3_f1"] >= minimum_bottom3_f1
        ),
        "ground_truth_macro_retention_at_least_90_percent": (
            retention >= minimum_gt_retention
        ),
    }
    passed = all(criteria.values())
    coverage_ok = criteria["train_coverage_at_least_90_percent"] and criteria[
        "validation_coverage_at_least_90_percent"
    ]
    raw_ok = criteria["raw_fusion_gain_at_least_2_points"] and criteria[
        "raw_bottom3_preserved"
    ]
    cm_ok = criteria["capacity_matched_gain_at_least_2_points"] and criteria[
        "capacity_matched_bottom3_preserved"
    ]
    if passed:
        action = "AUTHORIZE_MULTILEVEL_HEAD_STATIC_AUDIT"
    elif not coverage_ok:
        action = "STOP_PREDICTED_ROI_COVERAGE"
    elif raw_ok and not cm_ok:
        action = "STOP_FUSION_GAIN_EXPLAINED_BY_CAPACITY"
    elif not criteria["ground_truth_macro_retention_at_least_90_percent"]:
        action = "STOP_PREDICTED_ROI_TRANSFER"
    elif not criteria["capacity_matched_macro_at_least_75_percent"]:
        action = "STOP_CAPACITY_MATCHED_ABSOLUTE_MACRO_BELOW_GATE"
    elif not criteria["capacity_matched_bottom3_at_least_50_percent"]:
        action = "STOP_CAPACITY_MATCHED_ABSOLUTE_BOTTOM3_BELOW_GATE"
    else:
        action = "STOP_FUSION_ADVANTAGE_NOT_ROBUST"
    return {
        "decision": "PASS" if passed else "FAIL",
        "next_action": action,
        "raw_fusion_macro_gain": float(raw_gain),
        "capacity_matched_fusion_macro_gain": float(cm_gain),
        "ground_truth_macro_retention": float(retention),
        "criteria": criteria,
        "thresholds": {
            "minimum_coverage": minimum_coverage,
            "minimum_gain": minimum_gain,
            "minimum_macro_f1": minimum_macro_f1,
            "minimum_bottom3_f1": minimum_bottom3_f1,
            "minimum_gt_retention": minimum_gt_retention,
        },
        "detector_training_authorized": False,
        "test_access_authorized": False,
    }


def run_faruq_v3_predicted_roi_transfer(
    checkpoint: str | Path,
    data_root: str | Path,
    gt_report: str | Path,
    output: str | Path,
    *,
    device: str = "0",
    image_size: int = 640,
    roi_size: int = 3,
    candidate_count: int = 500,
    iou_threshold: float = 0.50,
    pca_components: int = 128,
    ridge: float = 0.01,
) -> dict[str, Any]:
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    gt_report = Path(gt_report).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not gt_report.is_file():
        raise FileNotFoundError(gt_report)
    reference = json.loads(gt_report.read_text(encoding="utf-8"))
    if reference.get("protocol") != "faruq-v3-pyramid-separability-v1":
        raise ValueError("GT report bukan pyramid separability v1")
    if reference.get("checkpoint_sha256") != _checkpoint_sha256(checkpoint):
        raise ValueError("Checkpoint GT report tidak sama dengan audit ini")
    layout, _ = _split_samples(data_root, "train")
    if "val" not in layout.splits or "test" in layout.splits:
        raise RuntimeError("Audit memerlukan train+val dan menolak test")
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    network = YOLO(str(checkpoint)).model.to(torch_device).eval()
    if int(_unwrap_head(network).nc) != len(layout.names):
        raise ValueError("Jumlah kelas checkpoint dan dataset berbeda")
    checkpoint_hash = reference["checkpoint_sha256"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    extracted = {}
    cache_metadata = {}
    for split in ("train", "val"):
        features, labels, metadata = _extract_predicted_split(
            network,
            data_root,
            split,
            destination.parent / f"predicted_roi_cache_{split}.npz",
            checkpoint_sha256=checkpoint_hash,
            device=torch_device,
            image_size=image_size,
            roi_size=roi_size,
            candidate_count=candidate_count,
            iou_threshold=iou_threshold,
        )
        extracted[split] = (features, labels)
        cache_metadata[split] = metadata

    train_features, train_labels = extracted["train"]
    val_features, val_labels = extracted["val"]
    matrices = {}
    for name, levels in RAW_REPRESENTATIONS.items():
        matrices[name] = (
            np.concatenate([train_features[level] for level in levels], axis=1),
            np.concatenate([val_features[level] for level in levels], axis=1),
        )
    pca_metadata = {}
    for raw_name, cm_name in (
        ("P5_RAW", "P5_CM128"),
        ("P3+P4+P5_RAW", "P3+P4+P5_CM128"),
    ):
        projected_train, projected_val, metadata = fit_pca_projection(
            *matrices[raw_name], components=pca_components
        )
        matrices[cm_name] = (projected_train, projected_val)
        pca_metadata[cm_name] = metadata

    results = {}
    for representation, (train_matrix, val_matrix) in matrices.items():
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
            "dimensions": int(train_matrix.shape[1]),
            "train": train_metrics,
            "validation": val_metrics,
            "macro_f1_generalization_gap": float(
                train_metrics["macro_f1"] - val_metrics["macro_f1"]
            ),
        }
        print(
            f"PREDICTED PROBE {representation}: "
            f"val Macro={val_metrics['macro_f1']:.2%} | "
            f"bottom3={val_metrics['bottom3_f1']:.2%}",
            flush=True,
        )

    coverage = {
        split: float(cache_metadata[split]["matched_recall"])
        for split in ("train", "val")
    }
    gt_fusion_macro = float(
        reference["results"]["P3+P4+P5"]["validation"]["macro_f1"]
    )
    decision = decide_predicted_roi_transfer(
        results, coverage, gt_fusion_macro
    )
    payload = {
        "protocol": "faruq-v3-predicted-roi-transfer-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_root": str(data_root),
        "ground_truth_reference": str(gt_report),
        "detector_training_executed": False,
        "probe_fitting_executed": True,
        "pca_fitting_executed": True,
        "validation_images_accessed": True,
        "test_images_accessed": False,
        "candidate_count": candidate_count,
        "iou_threshold": iou_threshold,
        "pca_components": pca_components,
        "ridge": ridge,
        "coverage": coverage,
        "cache": cache_metadata,
        "pca": pca_metadata,
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
        description="D0 predicted-ROI multilevel transfer and capacity audit"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--gt-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--roi-size", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=500)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=0.01)
    args = parser.parse_args()
    result = run_faruq_v3_predicted_roi_transfer(
        args.checkpoint,
        args.data_root,
        args.gt_report,
        args.output,
        device=args.device,
        image_size=args.image_size,
        roi_size=args.roi_size,
        candidate_count=args.candidate_count,
        iou_threshold=args.iou_threshold,
        pca_components=args.pca_components,
        ridge=args.ridge,
    )
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
