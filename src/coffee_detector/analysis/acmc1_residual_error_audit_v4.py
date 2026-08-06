"""ACMC1 residual-error audit V4.

Fixes the Ultralytics AP-matrix source used by V2/V3. In Ultralytics 8.4.96,
``Metric.ap`` is the per-class mean across IoU thresholds (shape nc,), while
``Metric.all_ap`` retains the full per-class x 10 IoU-threshold matrix.

V4 keeps the V3 legacy-checkpoint compatibility shim and replaces only the
per-seed evaluator so AP50/AP75/AP95 are read from ``all_ap``. No training or
test access is introduced.
"""

from __future__ import annotations

from pathlib import Path

import torch

from coffee_detector.analysis import acmc1_residual_error_audit_v2 as audit_v2
from coffee_detector.analysis.acmc1_residual_error_audit_v3 import (
    install_legacy_acmc1_checkpoint_compatibility,
)


def _run_seed_v4(checkpoint, data_root, *, seed, device):
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint seed {seed} tidak ditemukan: {checkpoint}")
    layout = audit_v2._validate_only_layout(data_root)

    print(f"SEED {seed}: validation AP matrix", flush=True)
    metrics = YOLO(str(checkpoint)).val(
        data=str(layout.yaml_path),
        split="val",
        plots=False,
        verbose=False,
        device=device,
    )
    metric_object = getattr(metrics, "box", None)
    if metric_object is None or getattr(metric_object, "all_ap", None) is None:
        raise RuntimeError("Ultralytics tidak mengembalikan box all_ap matrix")
    classwise = audit_v2._profile_from_ap_matrix(
        metric_object.all_ap,
        metric_object.ap_class_index,
        layout.names,
    )
    if set(classwise) != set(layout.names.values()):
        missing = sorted(set(layout.names.values()) - set(classwise))
        raise RuntimeError(f"Validation kehilangan kelas: {missing}")

    print(f"SEED {seed}: final-detection matching", flush=True)
    model = YOLO(str(checkpoint))
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")

    counts = audit_v2._new_counts(len(layout.names))
    samples = audit_v2._val_samples(layout)
    for index, (image_path, annotations) in enumerate(samples, 1):
        target_boxes, target_labels = audit_v2._target_tensors(
            image_path, annotations, torch_device
        )
        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=audit_v2.CONFIDENCE_THRESHOLD,
            iou=0.70,
            max_det=500,
            device=device,
            verbose=False,
        )[0]
        if result.boxes is None:
            pred_boxes = target_boxes.new_zeros((0, 4))
            pred_conf = target_boxes.new_zeros((0,))
            pred_labels = target_labels.new_zeros((0,), dtype=torch.long)
        else:
            pred_boxes = result.boxes.xyxy.to(torch_device)
            pred_conf = result.boxes.conf.to(torch_device)
            pred_labels = result.boxes.cls.long().to(torch_device)
        audit_v2._update_counts(
            counts,
            pred_boxes,
            pred_labels,
            pred_conf,
            target_boxes,
            target_labels,
        )
        if index % 100 == 0 or index == len(samples):
            print(f"SEED {seed}: {index}/{len(samples)} images", flush=True)

    final = audit_v2._finalize_counts(counts, layout.names)
    rows = {}
    for class_name, profile in classwise.items():
        row = {**profile, **final["per_class"][class_name]}
        row["flags"] = audit_v2._evidence_flags(row)
        row["attribution"] = audit_v2._attribution_label(row)
        rows[class_name] = row

    return {
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "global": {
            "precision": float(metrics.results_dict["metrics/precision(B)"]),
            "recall": float(metrics.results_dict["metrics/recall(B)"]),
            "map50": float(metrics.results_dict["metrics/mAP50(B)"]),
            "map50_95": float(metrics.results_dict["metrics/mAP50-95(B)"]),
            "detection_accessibility_iou50": final["detection_accessibility_iou50"],
            "matched_recall_iou50": final["matched_recall_iou50"],
            "class_accuracy_given_iou50_match": final[
                "class_accuracy_given_iou50_match"
            ],
            "classification_headroom_iou50": final["classification_headroom_iou50"],
        },
        "per_class": rows,
        "top_directional_confusions": audit_v2._top_confusions(final["confusion"]),
        "confusion": final["confusion"],
    }


def install_v4_fixes() -> None:
    install_legacy_acmc1_checkpoint_compatibility()
    audit_v2._run_seed = _run_seed_v4


def main() -> None:
    install_v4_fixes()
    audit_v2.main()


if __name__ == "__main__":
    main()
