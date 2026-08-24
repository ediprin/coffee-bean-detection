from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from coffee_detector.analysis.faruq_v3_predicted_roi_transfer import (
    fit_pca_projection,
)
from coffee_detector.analysis.faruq_v3_pyramid_separability import (
    _checkpoint_sha256,
    classification_metrics,
    fit_balanced_ridge_probe,
)


def decide_fusion_cm512(
    p5: dict[str, Any],
    fusion: dict[str, Any],
    *,
    minimum_macro_gain: float = 0.02,
    maximum_worst_drop: float = 0.01,
    minimum_macro_f1: float = 0.75,
    minimum_bottom3_f1: float = 0.50,
) -> dict[str, Any]:
    macro_gain = fusion["macro_f1"] - p5["macro_f1"]
    bottom3_gain = fusion["bottom3_f1"] - p5["bottom3_f1"]
    worst_gain = fusion["worst_class_f1"] - p5["worst_class_f1"]
    top3_gain = fusion["top3_accuracy"] - p5["top3_accuracy"]
    criteria = {
        "macro_gain_at_least_2_points": macro_gain >= minimum_macro_gain,
        "bottom3_preserved": bottom3_gain >= 0.0,
        "worst_drop_no_more_than_1_point": worst_gain >= -maximum_worst_drop,
        "top3_preserved": top3_gain >= 0.0,
        "fusion_macro_at_least_75_percent": fusion["macro_f1"] >= minimum_macro_f1,
        "fusion_bottom3_at_least_50_percent": (
            fusion["bottom3_f1"] >= minimum_bottom3_f1
        ),
    }
    passed = all(criteria.values())
    return {
        "decision": "PASS" if passed else "FAIL",
        "next_action": (
            "AUTHORIZE_MULTILEVEL_HEAD_PROTOCOL"
            if passed
            else "STOP_MULTILEVEL_HEAD_CAPACITY_CONTROL"
        ),
        "deltas": {
            "macro_f1": float(macro_gain),
            "bottom3_f1": float(bottom3_gain),
            "worst_class_f1": float(worst_gain),
            "top3_accuracy": float(top3_gain),
        },
        "criteria": criteria,
        "thresholds": {
            "minimum_macro_gain": minimum_macro_gain,
            "maximum_worst_drop": maximum_worst_drop,
            "minimum_macro_f1": minimum_macro_f1,
            "minimum_bottom3_f1": minimum_bottom3_f1,
        },
        "detector_training_authorized": False,
        "test_access_authorized": False,
    }


def _load_cache(
    report_root: Path,
    split: str,
    report: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    metadata = report["cache"][split]
    cache_path = report_root / f"predicted_roi_cache_{split}.npz"
    metadata_path = report_root / f"predicted_roi_cache_{split}.json"
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Cache predicted ROI {split} tidak lengkap")
    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in (
        "protocol",
        "split",
        "checkpoint_sha256",
        "sample_signature",
        "instances",
        "targets",
        "candidate_count",
        "iou_threshold",
    ):
        if stored_metadata.get(key) != metadata.get(key):
            raise ValueError(f"Metadata cache {split} berubah pada {key}")
    with np.load(cache_path, allow_pickle=False) as payload:
        labels = payload["labels"].astype(np.int64, copy=False)
        features = {
            level: payload[level].astype(np.float32, copy=False)
            for level in ("P3", "P4", "P5")
        }
    if len(labels) != int(metadata["instances"]):
        raise ValueError(f"Jumlah label cache {split} tidak cocok")
    return features, labels


def run_faruq_v3_fusion_cm512(
    checkpoint: str | Path,
    transfer_report: str | Path,
    output: str | Path,
    *,
    components: int = 512,
    ridge: float = 0.01,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    transfer_report = Path(transfer_report).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not transfer_report.is_file():
        raise FileNotFoundError(transfer_report)
    if components != 512:
        raise ValueError("CM512 protocol mengunci components=512")
    report = json.loads(transfer_report.read_text(encoding="utf-8"))
    if report.get("protocol") != "faruq-v3-predicted-roi-transfer-v1":
        raise ValueError("Transfer report bukan predicted-ROI v1")
    checkpoint_hash = _checkpoint_sha256(checkpoint)
    if report.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Checkpoint tidak cocok dengan transfer report")
    if report.get("test_images_accessed") is not False:
        raise RuntimeError("Transfer report tidak menjamin test terkunci")
    report_root = transfer_report.parent
    train_features, train_labels = _load_cache(report_root, "train", report)
    val_features, val_labels = _load_cache(report_root, "val", report)
    class_count = len(report["results"]["P5_RAW"]["validation"]["per_class"])

    raw_matrices = {
        "P5_CM512": (train_features["P5"], val_features["P5"]),
        "P3+P4+P5_CM512": (
            np.concatenate(
                [train_features[level] for level in ("P3", "P4", "P5")], axis=1
            ),
            np.concatenate(
                [val_features[level] for level in ("P3", "P4", "P5")], axis=1
            ),
        ),
    }
    results = {}
    pca = {}
    for name, (train_matrix, val_matrix) in raw_matrices.items():
        projected_train, projected_val, pca_metadata = fit_pca_projection(
            train_matrix, val_matrix, components=components
        )
        train_logits, val_logits = fit_balanced_ridge_probe(
            projected_train,
            train_labels,
            projected_val,
            num_classes=class_count,
            ridge=ridge,
        )
        train_metrics = classification_metrics(
            train_logits, train_labels, num_classes=class_count
        )
        val_metrics = classification_metrics(
            val_logits, val_labels, num_classes=class_count
        )
        source_per_class = report["results"]["P5_RAW"]["validation"]["per_class"]
        names = {int(row["class_id"]): row["class_name"] for row in source_per_class}
        for row in val_metrics["per_class"]:
            row["class_name"] = names[int(row["class_id"])]
        results[name] = {
            "dimensions": components,
            "train": train_metrics,
            "validation": val_metrics,
            "macro_f1_generalization_gap": float(
                train_metrics["macro_f1"] - val_metrics["macro_f1"]
            ),
        }
        pca[name] = pca_metadata
        print(
            f"CM512 {name}: train Macro={train_metrics['macro_f1']:.2%} | "
            f"val Macro={val_metrics['macro_f1']:.2%} | "
            f"bottom3={val_metrics['bottom3_f1']:.2%}",
            flush=True,
        )

    decision = decide_fusion_cm512(
        results["P5_CM512"]["validation"],
        results["P3+P4+P5_CM512"]["validation"],
    )
    payload = {
        "protocol": "faruq-v3-fusion-cm512-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "transfer_report": str(transfer_report),
        "detector_inference_executed": False,
        "detector_training_executed": False,
        "pca_fitting_executed": True,
        "probe_fitting_executed": True,
        "validation_features_accessed": True,
        "test_images_accessed": False,
        "components": components,
        "ridge": ridge,
        "results": results,
        "pca": pca,
        "decision": decision,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache-only P5 versus P3+P4+P5 PCA-512 control"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transfer-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--components", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=0.01)
    args = parser.parse_args()
    result = run_faruq_v3_fusion_cm512(
        args.checkpoint,
        args.transfer_report,
        args.output,
        components=args.components,
        ridge=args.ridge,
    )
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
