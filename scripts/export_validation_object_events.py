"""Export validation-only object events for complementarity audits.

Run this script from the model's native repository branch so checkpoint-specific
wrapper classes can be imported safely. It never trains and rejects any data root
that exposes a test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou

from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout, parse_label

MATCH_IOU = 0.50
CONFIDENCE_THRESHOLD = 0.25
PREDICTION_NMS_IOU = 0.70
MAX_DET = 500


def _validate_layout(data_root: str | Path):
    layout = discover_layout(data_root)
    if "val" not in layout.splits:
        raise RuntimeError("Validation split tidak tersedia")
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Exporter menolak data root yang mengekspos test")
    return layout


def _checkpoint_seed(path: str | Path) -> int:
    from ultralytics.utils.patches import torch_load
    source = Path(path).expanduser().resolve()
    payload = torch_load(source, map_location="cpu")
    args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(args, dict) or "seed" not in args:
        raise RuntimeError(f"Checkpoint tidak merekam train_args.seed: {source}")
    return int(args["seed"])


def _target_tensors(image_path, annotations, device):
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Gambar tidak dapat dibaca: {image_path}")
    height, width = image.shape[:2]
    boxes, labels = [], []
    for item in annotations:
        boxes.append([
            (item.x_center - item.width * 0.5) * width,
            (item.y_center - item.height * 0.5) * height,
            (item.x_center + item.width * 0.5) * width,
            (item.y_center + item.height * 0.5) * height,
        ])
        labels.append(int(item.class_id))
    return (
        torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def _confidence_match(pred_boxes, pred_conf, target_boxes):
    if not len(pred_boxes) or not len(target_boxes):
        return []
    matrix = box_iou(pred_boxes, target_boxes)
    available = torch.ones(len(target_boxes), dtype=torch.bool, device=target_boxes.device)
    matches = []
    for pred_index in pred_conf.argsort(descending=True).tolist():
        candidate = matrix[pred_index].clone()
        candidate[~available] = -1
        value, target_index = candidate.max(dim=0)
        if float(value) < MATCH_IOU:
            continue
        target_index = int(target_index)
        available[target_index] = False
        matches.append((int(pred_index), target_index, float(value)))
    return matches


def export_events(checkpoint, data_root, output, *, model_name: str, seed: int, device: str = "0"):
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    recorded_seed = _checkpoint_seed(checkpoint)
    if recorded_seed != int(seed):
        raise RuntimeError(f"Seed mismatch {model_name}: expected={seed}, checkpoint={recorded_seed}")

    layout = _validate_layout(data_root)
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA tidak tersedia: {torch_device}")

    image_root, label_root = layout.splits["val"]
    samples = []
    for image_path in sorted(p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        samples.append((relative.as_posix(), image_path, parse_label(label_path, set(layout.names))))
    if not samples:
        raise RuntimeError("Validation split kosong")

    model = YOLO(str(checkpoint))
    events = {}
    for image_index, (image_key, image_path, annotations) in enumerate(samples, 1):
        target_boxes, target_labels = _target_tensors(image_path, annotations, torch_device)
        result = model.predict(
            source=str(image_path), imgsz=640, conf=CONFIDENCE_THRESHOLD,
            iou=PREDICTION_NMS_IOU, max_det=MAX_DET, device=device, verbose=False,
        )[0]
        if result.boxes is None:
            pred_boxes = target_boxes.new_zeros((0, 4))
            pred_conf = target_boxes.new_zeros((0,))
            pred_labels = target_labels.new_zeros((0,), dtype=torch.long)
        else:
            pred_boxes = result.boxes.xyxy.to(torch_device)
            pred_conf = result.boxes.conf.to(torch_device)
            pred_labels = result.boxes.cls.long().to(torch_device)

        if len(pred_boxes) and len(target_boxes):
            accessible = box_iou(pred_boxes, target_boxes).max(dim=0).values >= MATCH_IOU
        else:
            accessible = torch.zeros(len(target_boxes), dtype=torch.bool, device=torch_device)
        match_by_target = {
            target_index: (pred_index, iou)
            for pred_index, target_index, iou in _confidence_match(pred_boxes, pred_conf, target_boxes)
        }
        for target_index, expected_tensor in enumerate(target_labels):
            expected = int(expected_tensor)
            key = f"{image_key}::gt{target_index}"
            event = {
                "target_key": key,
                "image": image_key,
                "target_index": target_index,
                "gt_class_id": expected,
                "gt_class_name": layout.names[expected],
                "accessible": bool(accessible[target_index]),
                "matched": False,
                "pred_class_id": None,
                "pred_class_name": None,
                "confidence": None,
                "iou": None,
                "correct": False,
            }
            if target_index in match_by_target:
                pred_index, iou = match_by_target[target_index]
                actual = int(pred_labels[pred_index])
                event.update(
                    matched=True,
                    pred_class_id=actual,
                    pred_class_name=layout.names[actual],
                    confidence=float(pred_conf[pred_index]),
                    iou=float(iou),
                    correct=(actual == expected),
                )
            events[key] = event
        if image_index % 100 == 0 or image_index == len(samples):
            print(f"{model_name}: {image_index}/{len(samples)} validation images", flush=True)

    payload = {
        "protocol": "faruq-v3-validation-object-events-v1",
        "model": model_name,
        "seed": int(seed),
        "checkpoint": str(checkpoint),
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "matching": {
            "class_agnostic": True,
            "iou_threshold": MATCH_IOU,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "prediction_nms_iou": PREDICTION_NMS_IOU,
            "max_det": MAX_DET,
        },
        "events": events,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("SAVED:", output)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    export_events(args.checkpoint, args.data_root, args.output, model_name=args.model_name, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
