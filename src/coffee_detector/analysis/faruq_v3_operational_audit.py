from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou, nms

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _confidence_ordered_match,
    _letterbox_sample,
    _raw_branches,
    _split_samples,
    _unwrap_head,
)


THRESHOLDS = (0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50)
POLICIES = ("native", "class_agnostic_nms")


def _new_totals(class_count: int) -> dict:
    return {
        "images": 0,
        "predictions": 0,
        "targets": 0,
        "accessible": 0,
        "matched": 0,
        "correct": 0,
        "wrong": 0,
        "correct_class_available": 0,
        "ranking_conflicts": 0,
        "no_correct_class_candidate": 0,
        "multi_label_conflicts": 0,
        "class_targets": np.zeros(class_count, dtype=np.int64),
        "class_accessible": np.zeros(class_count, dtype=np.int64),
        "class_correct": np.zeros(class_count, dtype=np.int64),
    }


def _update_totals(
    totals: dict,
    predictions: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    iou_threshold: float,
) -> None:
    totals["images"] += 1
    totals["predictions"] += len(predictions)
    totals["targets"] += len(target_boxes)
    if not len(target_boxes):
        return
    class_count = len(totals["class_targets"])
    totals["class_targets"] += np.bincount(
        target_labels.detach().cpu().numpy(), minlength=class_count
    )
    if not len(predictions):
        return

    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    labels = predictions[:, 5].long()
    matrix = box_iou(boxes, target_boxes)
    overlap = matrix >= iou_threshold
    accessible_mask = overlap.any(dim=0)
    totals["accessible"] += int(accessible_mask.sum())
    if accessible_mask.any():
        totals["class_accessible"] += np.bincount(
            target_labels[accessible_mask].detach().cpu().numpy(),
            minlength=class_count,
        )

    correct_available = torch.zeros(
        len(target_boxes), dtype=torch.bool, device=target_boxes.device
    )
    multi_label = torch.zeros_like(correct_available)
    for target_index in range(len(target_boxes)):
        candidates = torch.where(overlap[:, target_index])[0]
        if not len(candidates):
            continue
        candidate_labels = labels[candidates]
        correct_available[target_index] = bool(
            (candidate_labels == target_labels[target_index]).any()
        )
        multi_label[target_index] = len(candidate_labels.unique()) > 1
    totals["correct_class_available"] += int(correct_available.sum())
    totals["multi_label_conflicts"] += int(multi_label.sum())

    matches = _confidence_ordered_match(
        boxes, scores, target_boxes, iou_threshold
    )
    totals["matched"] += len(matches)
    for prediction, target, _ in matches:
        expected = int(target_labels[target])
        actual = int(labels[prediction])
        if actual == expected:
            totals["correct"] += 1
            totals["class_correct"][expected] += 1
        else:
            totals["wrong"] += 1
            if bool(correct_available[target]):
                totals["ranking_conflicts"] += 1
            else:
                totals["no_correct_class_candidate"] += 1


def _finalize(totals: dict, names: dict[int, str]) -> dict:
    targets = max(int(totals["targets"]), 1)
    matched = max(int(totals["matched"]), 1)
    accessible = max(int(totals["accessible"]), 1)
    class_targets = totals.pop("class_targets")
    class_accessible = totals.pop("class_accessible")
    class_correct = totals.pop("class_correct")
    return {
        **{key: int(value) for key, value in totals.items()},
        "proposal_accessibility": float(totals["accessible"] / targets),
        "conditional_top1_accuracy": float(totals["correct"] / matched),
        "correct_decision_recall": float(totals["correct"] / targets),
        "correct_class_availability": float(
            totals["correct_class_available"] / targets
        ),
        "ranking_conflict_rate": float(totals["ranking_conflicts"] / accessible),
        "no_correct_class_candidate_rate": float(
            totals["no_correct_class_candidate"] / accessible
        ),
        "multi_label_conflict_rate": float(
            totals["multi_label_conflicts"] / accessible
        ),
        "mean_predictions_per_image": float(
            totals["predictions"] / max(totals["images"], 1)
        ),
        "per_class": {
            names[index]: {
                "targets": int(class_targets[index]),
                "accessible": int(class_accessible[index]),
                "correct": int(class_correct[index]),
                "proposal_accessibility": float(
                    class_accessible[index] / max(class_targets[index], 1)
                ),
                "correct_decision_recall": float(
                    class_correct[index] / max(class_targets[index], 1)
                ),
            }
            for index in range(len(names))
        },
    }


def _select(final: torch.Tensor, threshold: float, policy: str) -> torch.Tensor:
    selected = final[final[:, 4] >= threshold]
    if policy == "native" or not len(selected):
        return selected
    if policy != "class_agnostic_nms":
        raise ValueError(f"Policy tidak dikenal: {policy}")
    indices = nms(selected[:, :4], selected[:, 4], iou_threshold=0.50)
    return selected[indices]


def _select_operating_point(rows: list[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            row["correct_decision_recall"],
            row["conditional_top1_accuracy"],
            row["threshold"],
        ),
    )


def audit_faruq_v3_operating_points(
    checkpoint: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    split: str = "val",
    device: str = "cpu",
) -> dict:
    if split != "val":
        raise RuntimeError("Operational audit dikunci pada validation")
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    layout, samples = _split_samples(data_root, "val")
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    network = YOLO(str(checkpoint)).model.to(torch_device).eval()
    if _unwrap_head(network).nc != len(layout.names):
        raise ValueError("Jumlah kelas checkpoint dan dataset berbeda")
    checkpoint_names = {
        int(index): str(name) for index, name in network.names.items()
    }
    dataset_names = {
        int(index): str(name) for index, name in layout.names.items()
    }
    if checkpoint_names != dataset_names:
        raise ValueError("Urutan kelas checkpoint dan dataset berbeda")

    totals = {
        (policy, threshold): _new_totals(len(layout.names))
        for policy in POLICIES
        for threshold in THRESHOLDS
    }
    with torch.inference_mode():
        for index, (image_path, annotations) in enumerate(samples, 1):
            image, target_boxes, target_labels, _ = _letterbox_sample(
                image_path, annotations, 640, torch_device
            )
            final, _, _ = _raw_branches(network, image, max_det=500)
            for policy in POLICIES:
                for threshold in THRESHOLDS:
                    predictions = _select(final, threshold, policy)
                    _update_totals(
                        totals[(policy, threshold)],
                        predictions,
                        target_boxes,
                        target_labels,
                        0.50,
                    )
            if index % 100 == 0 or index == len(samples):
                print(f"OPERATIONAL AUDIT {index}/{len(samples)}", flush=True)

    rows = []
    finalized_by_key = {}
    for (policy, threshold), values in totals.items():
        finalized = _finalize(values, layout.names)
        finalized_by_key[(policy, threshold)] = finalized
        rows.append(
            {
                "policy": policy,
                "threshold": threshold,
                **{key: value for key, value in finalized.items() if key != "per_class"},
            }
        )
    baseline = next(
        row for row in rows if row["policy"] == "native" and row["threshold"] == 0.25
    )
    selected = _select_operating_point(rows)
    gain = selected["correct_decision_recall"] - baseline["correct_decision_recall"]
    accessibility_delta = (
        selected["proposal_accessibility"] - baseline["proposal_accessibility"]
    )
    gate = gain >= 0.02 and accessibility_delta >= -0.01
    selected_details = finalized_by_key[(selected["policy"], selected["threshold"])]
    payload = {
        "protocol": "faruq-v3-operational-audit-v1",
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "thresholds": list(THRESHOLDS),
        "policies": list(POLICIES),
        "image_size": 640,
        "matching_iou": 0.50,
        "class_agnostic_nms_iou": 0.50,
        "rows": rows,
        "baseline": baseline,
        "selected": selected,
        "selected_per_class": selected_details["per_class"],
        "comparison": {
            "correct_decision_recall_gain": float(gain),
            "proposal_accessibility_delta": float(accessibility_delta),
            "postprocessing_sufficient": bool(gate),
            "decision": "PASS" if gate else "FAIL",
        },
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Faruq-v3 threshold and suppression audit.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = audit_faruq_v3_operating_points(
        args.checkpoint,
        args.data_root,
        args.output,
        split=args.split,
        device=args.device,
    )
    print(json.dumps(result["comparison"], indent=2))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
