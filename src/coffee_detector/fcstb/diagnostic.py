from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _confidence_ordered_match,
    _letterbox_sample,
    _split_samples,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _final_detections(model: torch.nn.Module, image: torch.Tensor, confidence: float):
    output = model(image)
    final = output[0] if isinstance(output, tuple) else output
    if final.ndim != 3 or final.shape[-1] != 6:
        raise TypeError(f"Output detector tidak dikenal: {tuple(final.shape)}")
    kept = final[0]
    return kept[kept[:, 4] >= float(confidence)]


def _target_predictions(
    detections: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    iou_threshold: float,
) -> dict[int, int]:
    matches = _confidence_ordered_match(
        detections[:, :4], detections[:, 4], target_boxes, iou_threshold
    )
    return {target: int(detections[prediction, 5]) for prediction, target, _ in matches}


def run_frequency_teacher_headroom(
    stb_checkpoint: str | Path,
    af2_checkpoint: str | Path,
    data_root: str | Path,
    output: str | Path,
    *,
    split: str = "val",
    device: str = "cpu",
    image_size: int = 640,
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.5,
    minimum_rescue_fraction: float = 0.01,
    minimum_rescue_classes: int = 3,
) -> dict[str, Any]:
    """Measure GT-bounded AF2 rescue signal before authorizing training."""

    if split != "val":
        raise RuntimeError("FC-STB diagnostic dikunci pada validation")
    from ultralytics import YOLO

    stb_path = Path(stb_checkpoint).expanduser().resolve()
    af2_path = Path(af2_checkpoint).expanduser().resolve()
    if not stb_path.is_file() or not af2_path.is_file():
        raise FileNotFoundError(f"Checkpoint hilang: STB={stb_path}, AF2={af2_path}")
    layout, samples = _split_samples(data_root, split)
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia")
    stb = YOLO(str(stb_path)).model.to(torch_device).eval()
    af2 = YOLO(str(af2_path)).model.to(torch_device).eval()
    totals = Counter()
    rescue_by_class = Counter()
    regression_by_class = Counter()

    with torch.inference_mode():
        for index, (image_path, annotations) in enumerate(samples, 1):
            image, target_boxes, target_labels, _ = _letterbox_sample(
                image_path, annotations, image_size, torch_device
            )
            left = _target_predictions(
                _final_detections(stb, image, confidence_threshold),
                target_boxes,
                iou_threshold=iou_threshold,
            )
            right = _target_predictions(
                _final_detections(af2, image, confidence_threshold),
                target_boxes,
                iou_threshold=iou_threshold,
            )
            for target_index, expected_tensor in enumerate(target_labels):
                expected = int(expected_tensor)
                left_correct = left.get(target_index) == expected
                right_correct = right.get(target_index) == expected
                totals["targets"] += 1
                totals["stb_correct"] += int(left_correct)
                totals["af2_correct"] += int(right_correct)
                if left_correct and right_correct:
                    totals["both_correct"] += 1
                elif right_correct:
                    totals["af2_rescue"] += 1
                    rescue_by_class[layout.names[expected]] += 1
                elif left_correct:
                    totals["af2_regression"] += 1
                    regression_by_class[layout.names[expected]] += 1
                else:
                    totals["both_wrong_or_missed"] += 1
            if index % 100 == 0 or index == len(samples):
                print(f"FC-STB DIAGNOSTIC {index}/{len(samples)}", flush=True)

    target_count = max(int(totals["targets"]), 1)
    rescue_fraction = float(totals["af2_rescue"] / target_count)
    criteria = {
        "af2_rescue_fraction_at_least_threshold": rescue_fraction
        >= float(minimum_rescue_fraction),
        "af2_rescues_at_least_minimum_classes": len(rescue_by_class)
        >= int(minimum_rescue_classes),
        "both_models_evaluated_on_same_targets": int(totals["targets"]) > 0,
    }
    payload = {
        "protocol": "faruq-v3-fcstb-teacher-headroom-v1",
        "split": split,
        "training_executed": False,
        "test_images_accessed": False,
        "stb_checkpoint": str(stb_path),
        "af2_checkpoint": str(af2_path),
        "stb_sha256": _sha256(stb_path),
        "af2_sha256": _sha256(af2_path),
        "images": len(samples),
        "counts": {key: int(value) for key, value in totals.items()},
        "rates": {
            "stb_correct": float(totals["stb_correct"] / target_count),
            "af2_correct": float(totals["af2_correct"] / target_count),
            "af2_rescue": rescue_fraction,
            "af2_regression": float(totals["af2_regression"] / target_count),
        },
        "af2_rescue_by_class": dict(rescue_by_class.most_common()),
        "af2_regression_by_class": dict(regression_by_class.most_common()),
        "thresholds": {
            "minimum_rescue_fraction": float(minimum_rescue_fraction),
            "minimum_rescue_classes": int(minimum_rescue_classes),
            "confidence_threshold": float(confidence_threshold),
            "iou_threshold": float(iou_threshold),
        },
        "criteria": criteria,
        "decision": "PASS" if all(criteria.values()) else "FAIL",
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["summary"] = str(destination)
    return payload
