from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from coffee_detector.coffee_fg.model import CoffeeFGDetectHead
from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout, parse_label


def _split_samples(data_root: str | Path, split: str) -> tuple[Any, list[tuple[Path, tuple]]]:
    layout = discover_layout(data_root)
    if split not in layout.splits:
        raise FileNotFoundError(f"Split {split} tidak ditemukan di {layout.root}")
    image_root, label_root = layout.splits[split]
    samples = []
    for image_path in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        samples.append((image_path, parse_label(label_path, set(layout.names))))
    if not samples:
        raise RuntimeError(f"Split {split} kosong: {image_root}")
    return layout, samples


def _letterbox_sample(
    image_path: Path,
    annotations: tuple,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    from ultralytics.data.augment import LetterBox

    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Gambar tidak dapat dibaca: {image_path}")
    original_height, original_width = image.shape[:2]
    transform = LetterBox(
        new_shape=(image_size, image_size),
        auto=False,
        scale_fill=False,
        scaleup=True,
        center=True,
        stride=32,
    )
    params = transform.get_params({"img": image})
    resized = transform(image=image)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
        .unsqueeze(0)
    )

    boxes = []
    labels = []
    ratio_x, ratio_y = params["ratio"]
    for item in annotations:
        x1 = (item.x_center - item.width * 0.5) * original_width
        y1 = (item.y_center - item.height * 0.5) * original_height
        x2 = (item.x_center + item.width * 0.5) * original_width
        y2 = (item.y_center + item.height * 0.5) * original_height
        boxes.append(
            [
                x1 * ratio_x + params["left"],
                y1 * ratio_y + params["top"],
                x2 * ratio_x + params["left"],
                y2 * ratio_y + params["top"],
            ]
        )
        labels.append(item.class_id)
    return (
        tensor,
        torch.tensor(boxes, device=device, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(labels, device=device, dtype=torch.long),
        (original_height, original_width),
    )


def _unwrap_head(model: torch.nn.Module):
    head = model.model[-1]
    return head.base_head if isinstance(head, CoffeeFGDetectHead) else head


def _raw_branches(model: torch.nn.Module, image: torch.Tensor, max_det: int):
    head = model.model[-1]
    base_head = _unwrap_head(model)
    head.max_det = max_det
    base_head.max_det = max_det
    output = model(image)
    if not isinstance(output, tuple) or not isinstance(output[1], dict):
        raise TypeError("Checkpoint tidak menyediakan raw one-to-one/one-to-many branches")
    final, raw = output
    if not {"one2one", "one2many"} <= set(raw):
        raise KeyError("Raw checkpoint tidak memuat kedua branch YOLO26")
    return final[0], raw, base_head


def _decode_branch(
    head,
    branch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    boxes = head._get_decode_boxes(branch).transpose(1, 2)[0]
    scores = branch["scores"].sigmoid().transpose(1, 2)[0]
    return boxes, scores


def _rank_candidates(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    confidence, labels = scores.max(dim=1)
    count = min(int(count), int(len(confidence)))
    indices = confidence.topk(count).indices
    return boxes[indices], labels[indices], confidence[indices]


def _greedy_match(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    threshold: float,
) -> list[tuple[int, int, float]]:
    if not len(predicted_boxes) or not len(target_boxes):
        return []
    matrix = box_iou(predicted_boxes, target_boxes)
    order = matrix.flatten().argsort(descending=True)
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matches = []
    target_count = matrix.shape[1]
    for flat_index in order.tolist():
        prediction = int(flat_index // target_count)
        target = int(flat_index % target_count)
        value = float(matrix[prediction, target])
        if value < threshold:
            break
        if prediction in used_predictions or target in used_targets:
            continue
        used_predictions.add(prediction)
        used_targets.add(target)
        matches.append((prediction, target, value))
    return matches


def _new_branch_totals(num_classes: int) -> dict[str, Any]:
    return {
        "targets": 0,
        "accessible": 0,
        "matched": 0,
        "correct_class": 0,
        "wrong_class": 0,
        "missed": 0,
        "duplicates": 0,
        "confusion": np.zeros((num_classes, num_classes), dtype=np.int64),
        "class_targets": np.zeros(num_classes, dtype=np.int64),
        "class_accessible": np.zeros(num_classes, dtype=np.int64),
        "class_matched": np.zeros(num_classes, dtype=np.int64),
        "class_correct": np.zeros(num_classes, dtype=np.int64),
    }


def _update_branch(
    totals: dict[str, Any],
    predicted_boxes: torch.Tensor,
    predicted_labels: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    iou_threshold: float,
) -> None:
    target_count = len(target_boxes)
    totals["targets"] += target_count
    if not target_count:
        return
    target_label_values = target_labels.detach().cpu().numpy()
    totals["class_targets"] += np.bincount(
        target_label_values, minlength=len(totals["class_targets"])
    )
    matrix = (
        box_iou(predicted_boxes, target_boxes)
        if len(predicted_boxes)
        else target_boxes.new_zeros((0, target_count))
    )
    accessible_mask = (
        matrix.max(dim=0).values >= iou_threshold
        if len(matrix)
        else target_boxes.new_zeros(target_count, dtype=torch.bool)
    )
    accessible = int(accessible_mask.sum())
    if accessible:
        accessible_labels = target_labels[accessible_mask].detach().cpu().numpy()
        totals["class_accessible"] += np.bincount(
            accessible_labels, minlength=len(totals["class_accessible"])
        )
    matches = _greedy_match(predicted_boxes, target_boxes, iou_threshold)
    totals["accessible"] += accessible
    totals["matched"] += len(matches)
    totals["missed"] += target_count - len(matches)
    positive_candidates = (
        int((matrix.max(dim=1).values >= iou_threshold).sum()) if len(matrix) else 0
    )
    totals["duplicates"] += max(positive_candidates - len(matches), 0)
    for prediction, target, _ in matches:
        expected = int(target_labels[target])
        actual = int(predicted_labels[prediction])
        totals["class_matched"][expected] += 1
        totals["confusion"][expected, actual] += 1
        if actual == expected:
            totals["correct_class"] += 1
            totals["class_correct"][expected] += 1
        else:
            totals["wrong_class"] += 1


def _finalize_branch(totals: dict[str, Any], names: dict[int, str]) -> dict:
    targets = max(int(totals["targets"]), 1)
    matched = max(int(totals["matched"]), 1)
    confusion = totals.pop("confusion")
    class_targets = totals.pop("class_targets")
    class_accessible = totals.pop("class_accessible")
    class_matched = totals.pop("class_matched")
    class_correct = totals.pop("class_correct")
    return {
        **{key: int(value) for key, value in totals.items()},
        "proposal_accessibility": float(totals["accessible"] / targets),
        "matched_recall": float(totals["matched"] / targets),
        "localization_conditioned_class_accuracy": float(
            totals["correct_class"] / matched
        ),
        "oracle_class_accuracy_headroom": float(totals["wrong_class"] / matched),
        "confusion": {
            names[row]: {
                names[column]: int(confusion[row, column])
                for column in range(len(names))
                if confusion[row, column]
            }
            for row in range(len(names))
            if confusion[row].sum()
        },
        "per_class": {
            names[index]: {
                "targets": int(class_targets[index]),
                "accessible": int(class_accessible[index]),
                "matched": int(class_matched[index]),
                "correct_class": int(class_correct[index]),
                "proposal_accessibility": float(
                    class_accessible[index] / max(class_targets[index], 1)
                ),
                "matched_recall": float(
                    class_matched[index] / max(class_targets[index], 1)
                ),
                "localization_conditioned_class_accuracy": float(
                    class_correct[index] / max(class_matched[index], 1)
                ),
            }
            for index in range(len(names))
        },
    }


def _nms_one2many(
    head,
    branch: dict[str, torch.Tensor],
    *,
    confidence: float,
    iou: float,
    max_det: int,
) -> torch.Tensor:
    from ultralytics.utils.nms import non_max_suppression

    original = bool(head.end2end)
    head.end2end = False
    try:
        decoded = head._inference(branch)
    finally:
        head.end2end = original
    return non_max_suppression(
        decoded,
        conf_thres=confidence,
        iou_thres=iou,
        max_det=max_det,
        max_time_img=1.0,
        nc=head.nc,
        end2end=False,
    )[0]


def _count_summary(rows: list[dict[str, int]]) -> dict:
    errors = np.asarray([row["predicted"] - row["target"] for row in rows], dtype=float)
    return {
        "images": len(rows),
        "exact_count_accuracy": float(np.mean(errors == 0)) if len(errors) else 0.0,
        "count_mae": float(np.mean(np.abs(errors))) if len(errors) else 0.0,
        "signed_count_bias": float(np.mean(errors)) if len(errors) else 0.0,
        "target_count_min": int(min((row["target"] for row in rows), default=0)),
        "target_count_median": float(
            np.median([row["target"] for row in rows]) if rows else 0.0
        ),
        "target_count_p95": float(
            np.percentile([row["target"] for row in rows], 95) if rows else 0.0
        ),
        "target_count_max": int(max((row["target"] for row in rows), default=0)),
    }


def diagnose_checkpoint(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    split: str = "val",
    image_size: int = 640,
    candidate_counts: tuple[int, ...] = (50, 100, 300, 500),
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
    nms_iou: float = 0.7,
    max_det: int = 500,
    device: str = "cpu",
) -> dict:
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    layout, samples = _split_samples(data_root, split)
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    network = YOLO(str(checkpoint)).model.to(torch_device).eval()
    checkpoint_classes = int(_unwrap_head(network).nc)
    if checkpoint_classes != len(layout.names):
        raise ValueError(
            f"Jumlah kelas checkpoint ({checkpoint_classes}) tidak cocok dengan "
            f"data.yaml ({len(layout.names)})"
        )
    branch_totals = {
        branch: {
            count: _new_branch_totals(len(layout.names))
            for count in sorted(set(candidate_counts))
        }
        for branch in ("one2one", "one2many")
    }
    count_rows = {"one2one": [], "one2many_nms": []}

    with torch.inference_mode():
        for index, (image_path, annotations) in enumerate(samples, 1):
            image, target_boxes, target_labels, _ = _letterbox_sample(
                image_path, annotations, image_size, torch_device
            )
            final, raw, head = _raw_branches(network, image, max_det)
            for branch_name in ("one2one", "one2many"):
                boxes, scores = _decode_branch(head, raw[branch_name])
                for count, totals in branch_totals[branch_name].items():
                    selected_boxes, selected_labels, _ = _rank_candidates(
                        boxes, scores, count
                    )
                    _update_branch(
                        totals,
                        selected_boxes,
                        selected_labels,
                        target_boxes,
                        target_labels,
                        iou_threshold,
                    )

            final_kept = final[final[:, 4] >= confidence_threshold]
            traditional = _nms_one2many(
                head,
                raw["one2many"],
                confidence=confidence_threshold,
                iou=nms_iou,
                max_det=max_det,
            )
            target_count = len(target_boxes)
            count_rows["one2one"].append(
                {"target": target_count, "predicted": len(final_kept)}
            )
            count_rows["one2many_nms"].append(
                {"target": target_count, "predicted": len(traditional)}
            )
            if index % 100 == 0 or index == len(samples):
                print(f"DIAGNOSTIC {index}/{len(samples)}", flush=True)

    finalized = {
        branch: {
            str(count): _finalize_branch(totals, layout.names)
            for count, totals in counts.items()
        }
        for branch, counts in branch_totals.items()
    }
    density = _count_summary(count_rows["one2one"])
    density["configured_max_det"] = int(max_det)
    density["max_det_covers_validation"] = density["target_count_max"] <= max_det
    return {
        "checkpoint": str(checkpoint),
        "data_root": str(layout.root),
        "split": split,
        "image_size": image_size,
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "candidate_counts": list(sorted(set(candidate_counts))),
        "branches": finalized,
        "counting": {
            name: _count_summary(rows) for name, rows in count_rows.items()
        },
        "density_contract": density,
    }


def compare_p3_p2_diagnostics(
    p3: dict,
    p2: dict,
    *,
    min_proposal_recall: float = 0.90,
    min_oracle_headroom: float = 0.02,
    min_p2_gain: float = 0.01,
) -> dict:
    if p3["candidate_counts"] != p2["candidate_counts"]:
        raise ValueError("Candidate counts P3 dan P2 harus identik")
    largest = str(max(p3["candidate_counts"]))
    p3_row = p3["branches"]["one2one"][largest]
    p2_row = p2["branches"]["one2one"][largest]
    gain = p2_row["proposal_accessibility"] - p3_row["proposal_accessibility"]
    foundation = "D1" if gain >= min_p2_gain else "D0"
    selected = p2_row if foundation == "D1" else p3_row
    rational = (
        selected["proposal_accessibility"] >= min_proposal_recall
        and selected["oracle_class_accuracy_headroom"] >= min_oracle_headroom
    )
    return {
        "candidate_count": int(largest),
        "p2_accessibility_gain": float(gain),
        "recommended_foundation": foundation,
        "recommended_refiners": (
            ["R2", "R3"] if foundation == "D1" else ["R0", "R1"]
        ),
        "classification_refinement_rational": bool(rational),
        "criteria": {
            "proposal_accessibility_sufficient": (
                selected["proposal_accessibility"] >= min_proposal_recall
            ),
            "oracle_class_headroom_sufficient": (
                selected["oracle_class_accuracy_headroom"] >= min_oracle_headroom
            ),
            "p2_materially_improves_accessibility": gain >= min_p2_gain,
        },
        "thresholds": {
            "min_proposal_recall": min_proposal_recall,
            "min_oracle_headroom": min_oracle_headroom,
            "min_p2_gain": min_p2_gain,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only proposal and classification-headroom audit for CoffeeFG."
    )
    parser.add_argument("--p3-checkpoint", required=True)
    parser.add_argument("--p2-checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--open-test", action="store_true")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--candidate-counts", nargs="+", type=int, default=[50, 100, 300, 500])
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.split == "test" and not args.open_test:
        raise RuntimeError("Test terkunci; diagnostic screening hanya memakai validation")

    common = {
        "split": args.split,
        "image_size": args.image_size,
        "candidate_counts": tuple(args.candidate_counts),
        "iou_threshold": args.iou_threshold,
        "confidence_threshold": args.confidence_threshold,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "device": args.device,
    }
    p3 = diagnose_checkpoint(args.p3_checkpoint, args.data_root, **common)
    p2 = diagnose_checkpoint(args.p2_checkpoint, args.data_root, **common)
    payload = {
        "protocol": "coffee-fg-diagnostic-v2",
        "test_opened": args.split == "test",
        "D0": p3,
        "D1": p2,
        "decision": compare_p3_p2_diagnostics(p3, p2),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))
    print("SAVED:", output)


if __name__ == "__main__":
    main()
