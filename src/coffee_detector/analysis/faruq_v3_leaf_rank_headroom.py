from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _decode_branch,
    _greedy_match,
    _letterbox_sample,
    _raw_branches,
    _split_samples,
)


def summarize_true_class_ranks(ranks: Iterable[int]) -> dict[str, Any]:
    values = np.asarray([int(rank) for rank in ranks], dtype=np.int64)
    if values.size == 0 or np.any(values < 1):
        raise ValueError("True-class ranks harus berisi integer positif")
    result: dict[str, Any] = {
        "matched": int(values.size),
        "mean_reciprocal_rank": float(np.mean(1.0 / values)),
        "mean_true_class_rank": float(np.mean(values)),
        "median_true_class_rank": float(np.median(values)),
        "rank_distribution": {
            str(rank): int(count)
            for rank, count in sorted(Counter(values.tolist()).items())
        },
    }
    for k in (1, 2, 3, 5):
        result[f"conditional_top{k}_accuracy"] = float(np.mean(values <= k))
    result["top3_recovery_over_top1"] = float(
        result["conditional_top3_accuracy"]
        - result["conditional_top1_accuracy"]
    )
    return result


def decide_leaf_rank_headroom(
    summary: dict[str, Any],
    *,
    maximum_top1: float = 0.80,
    minimum_top3: float = 0.80,
    minimum_recovery: float = 0.15,
) -> dict[str, Any]:
    top1 = float(summary["conditional_top1_accuracy"])
    top3 = float(summary["conditional_top3_accuracy"])
    recovery = float(summary["top3_recovery_over_top1"])
    criteria = {
        "top1_below_80_percent": top1 < maximum_top1,
        "top3_at_least_80_percent": top3 >= minimum_top3,
        "top3_recovery_at_least_15_points": recovery >= minimum_recovery,
    }
    passed = all(criteria.values())
    if passed:
        action = "AUTHORIZE_LEAF_RERANKING_PROTOCOL"
    elif top1 >= maximum_top1:
        action = "STOP_HEAD_REFINEMENT_NEAR_SATURATION"
    elif top3 < minimum_top3:
        action = "STOP_RERANKING_REPRESENTATION_LIMITED"
    else:
        action = "STOP_RERANKING_HEADROOM_TOO_SMALL"
    return {
        "decision": "PASS" if passed else "FAIL",
        "criteria": criteria,
        "next_action": action,
        "thresholds": {
            "maximum_top1": maximum_top1,
            "minimum_top3": minimum_top3,
            "minimum_top3_recovery": minimum_recovery,
        },
        "training_authorized": False,
    }


def _rank_row(
    scores: torch.Tensor,
    expected: int,
    names: dict[int, str],
) -> dict[str, Any]:
    order = scores.argsort(descending=True)
    position = int((order == expected).nonzero(as_tuple=False)[0, 0]) + 1
    top1 = int(order[0])
    true_score = float(scores[expected])
    other = scores.clone()
    other[expected] = -1.0
    strongest_other = float(other.max())
    return {
        "expected_id": expected,
        "expected": names[expected],
        "predicted_id": top1,
        "predicted": names[top1],
        "true_class_rank": position,
        "true_class_probability": true_score,
        "strongest_other_probability": strongest_other,
        "true_minus_other_margin": true_score - strongest_other,
    }


def _per_class(rows: list[dict[str, Any]], names: dict[int, str]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["expected_id"])].append(row)
    output = []
    for class_id, class_name in names.items():
        class_rows = grouped.get(class_id, [])
        ranks = [int(row["true_class_rank"]) for row in class_rows]
        if not ranks:
            output.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "matched": 0,
                    "conditional_top1_accuracy": None,
                    "conditional_top3_accuracy": None,
                    "top3_recovery_over_top1": None,
                }
            )
            continue
        summary = summarize_true_class_ranks(ranks)
        output.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "matched": summary["matched"],
                "conditional_top1_accuracy": summary["conditional_top1_accuracy"],
                "conditional_top3_accuracy": summary["conditional_top3_accuracy"],
                "top3_recovery_over_top1": summary["top3_recovery_over_top1"],
                "mean_true_class_rank": summary["mean_true_class_rank"],
            }
        )
    return output


def _confusion_pairs(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["expected"] != row["predicted"]:
            grouped[(row["expected"], row["predicted"])].append(row)
    output = []
    for (expected, predicted), pair_rows in grouped.items():
        ranks = [int(row["true_class_rank"]) for row in pair_rows]
        output.append(
            {
                "expected": expected,
                "predicted": predicted,
                "count": len(pair_rows),
                "true_class_top3_fraction": float(np.mean(np.asarray(ranks) <= 3)),
                "median_true_class_rank": float(np.median(ranks)),
                "median_true_minus_other_margin": float(
                    np.median(
                        [row["true_minus_other_margin"] for row in pair_rows]
                    )
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (-row["count"], row["expected"], row["predicted"]),
    )[:limit]


def run_faruq_v3_leaf_rank_headroom(
    checkpoint: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    device: str = "0",
    image_size: int = 640,
    candidate_count: int = 500,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    from ultralytics import YOLO

    if candidate_count <= 0:
        raise ValueError("candidate_count harus positif")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    layout, samples = _split_samples(data_root, "val")
    if "test" in layout.splits:
        raise RuntimeError("Leaf-rank audit menolak dataset yang mengekspos test")
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    network = YOLO(str(checkpoint)).model.to(torch_device).eval()
    head = network.model[-1]
    if int(head.nc) != len(layout.names):
        raise ValueError("Jumlah kelas checkpoint dan dataset berbeda")

    rows: list[dict[str, Any]] = []
    targets_total = 0
    accessible_total = 0
    with torch.inference_mode():
        for image_index, (image_path, annotations) in enumerate(samples, 1):
            image, target_boxes, target_labels, _ = _letterbox_sample(
                image_path, annotations, image_size, torch_device
            )
            _, raw, decoded_head = _raw_branches(network, image, candidate_count)
            boxes, scores = _decode_branch(decoded_head, raw["one2one"])
            confidence = scores.max(dim=1).values
            keep = confidence.topk(min(candidate_count, len(confidence))).indices
            candidate_boxes = boxes[keep]
            candidate_scores = scores[keep]
            targets_total += len(target_boxes)
            if len(candidate_boxes) and len(target_boxes):
                from torchvision.ops import box_iou

                accessible_total += int(
                    (box_iou(candidate_boxes, target_boxes).max(dim=0).values >= iou_threshold)
                    .sum()
                    .cpu()
                )
            matches = _greedy_match(candidate_boxes, target_boxes, iou_threshold)
            for prediction, target, iou in matches:
                row = _rank_row(
                    candidate_scores[prediction],
                    int(target_labels[target]),
                    layout.names,
                )
                row.update(
                    {
                        "image": str(image_path),
                        "iou": float(iou),
                    }
                )
                rows.append(row)
            if image_index % 50 == 0 or image_index == len(samples):
                print(
                    f"LEAF RANK {image_index}/{len(samples)} | matched={len(rows)}",
                    flush=True,
                )

    ranks = [int(row["true_class_rank"]) for row in rows]
    summary = summarize_true_class_ranks(ranks)
    margins = [float(row["true_minus_other_margin"]) for row in rows]
    summary.update(
        {
            "targets": targets_total,
            "accessible": accessible_total,
            "proposal_accessibility": accessible_total / max(targets_total, 1),
            "matched_recall": len(rows) / max(targets_total, 1),
            "median_true_minus_other_margin": float(np.median(margins)),
            "wrong_top1_with_true_in_top3": int(
                sum(1 < rank <= 3 for rank in ranks)
            ),
        }
    )
    decision = decide_leaf_rank_headroom(summary)
    payload = {
        "protocol": "faruq-v3-leaf-rank-headroom-v1",
        "checkpoint": str(checkpoint),
        "dataset_root": str(layout.root),
        "split": "val",
        "training_executed": False,
        "validation_images_accessed": True,
        "test_images_accessed": False,
        "image_size": image_size,
        "candidate_count": candidate_count,
        "iou_threshold": iou_threshold,
        "global": summary,
        "decision": decision,
        "per_class": _per_class(rows, layout.names),
        "top_confusion_pairs": _confusion_pairs(rows),
        "matches": rows,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only D0 true-class rank headroom audit"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--candidate-count", type=int, default=500)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    args = parser.parse_args()
    result = run_faruq_v3_leaf_rank_headroom(
        args.checkpoint,
        args.data_root,
        args.output,
        device=args.device,
        image_size=args.image_size,
        candidate_count=args.candidate_count,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(result["global"], indent=2, ensure_ascii=False))
    print(json.dumps(result["decision"], indent=2, ensure_ascii=False))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()

