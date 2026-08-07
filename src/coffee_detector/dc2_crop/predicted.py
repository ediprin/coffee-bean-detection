from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from coffee_detector.dataset import Box
from .data import box_to_xyxy, collect_crop_records


@dataclass(frozen=True)
class PredictedCropRecord:
    """A detector prediction matched label-agnostically to one GT object.

    ``class_id`` is the GT class and is only used as supervision/evaluation.
    ``predicted_class_id`` is the detector's native class decision for the same
    matched box. The local classifier therefore never receives validation
    confusion pairs as training knowledge.
    """

    image_path: Path
    class_id: int
    gt_box: Box
    predicted_xyxy: tuple[float, float, float, float]
    predicted_class_id: int
    predicted_confidence: float
    matched_iou: float


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xyxy_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in left)
    bx1, by1, bx2, by2 = (float(value) for value in right)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def greedy_match_xyxy(
    prediction_boxes: list[tuple[float, float, float, float]],
    prediction_scores: list[float],
    target_boxes: list[tuple[float, float, float, float]],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Class-agnostic one-to-one matching, highest IoU first.

    Class-agnostic matching is intentional: the detector's class output must
    not decide whether a sample is available to the local classifier.
    """

    if len(prediction_boxes) != len(prediction_scores):
        raise ValueError("prediction_boxes dan prediction_scores tidak sejajar")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold harus di [0,1]")
    candidates: list[tuple[float, float, int, int]] = []
    for prediction_index, prediction_box in enumerate(prediction_boxes):
        for target_index, target_box in enumerate(target_boxes):
            overlap = xyxy_iou(prediction_box, target_box)
            if overlap >= iou_threshold:
                candidates.append(
                    (overlap, float(prediction_scores[prediction_index]), prediction_index, target_index)
                )
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, _score, prediction_index, target_index in candidates:
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        matches.append((prediction_index, target_index, float(overlap)))
    return matches


def _expand_xyxy(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    context: float,
) -> tuple[int, int, int, int]:
    if context < 1.0:
        raise ValueError("context minimal 1.0")
    x1, y1, x2, y2 = (float(value) for value in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw = max(1.0, (x2 - x1) * context)
    bh = max(1.0, (y2 - y1) * context)
    left = max(0, int(round(cx - bw / 2.0)))
    top = max(0, int(round(cy - bh / 2.0)))
    right = min(width, int(round(cx + bw / 2.0)))
    bottom = min(height, int(round(cy + bh / 2.0)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


class MatchedRawObjectCropDataset(Dataset):
    """Raw RGB crop dataset on the same detector-matched target identities.

    ``source='predicted'`` crops using detector boxes. ``source='gt'`` crops
    the corresponding exact GT box, enabling a paired information-retention
    control on identical objects.
    """

    def __init__(
        self,
        records: list[PredictedCropRecord],
        resolution: int,
        *,
        training: bool,
        source: str,
        context: float = 1.0,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution harus positif")
        if source not in {"predicted", "gt"}:
            raise ValueError("source harus 'predicted' atau 'gt'")
        self.records = records
        self.resolution = int(resolution)
        self.source = source
        self.context = float(context)
        augmentation = []
        if training:
            augmentation.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ]
            )
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution), antialias=True),
                *augmentation,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(record.image_path) as source_image:
            image = source_image.convert("RGB")
            if self.source == "predicted":
                crop_box = _expand_xyxy(
                    record.predicted_xyxy, image.width, image.height, self.context
                )
            else:
                gt_xyxy = box_to_xyxy(record.gt_box, image.width, image.height, 1.0)
                crop_box = _expand_xyxy(gt_xyxy, image.width, image.height, self.context)
            crop = image.crop(crop_box)
            tensor = self.transform(crop)
        return tensor, int(record.class_id)


def _serialize_record(record: PredictedCropRecord, data_root: Path) -> dict:
    return {
        "image_path": str(record.image_path.relative_to(data_root)),
        "class_id": int(record.class_id),
        "gt_box": asdict(record.gt_box),
        "predicted_xyxy": [float(value) for value in record.predicted_xyxy],
        "predicted_class_id": int(record.predicted_class_id),
        "predicted_confidence": float(record.predicted_confidence),
        "matched_iou": float(record.matched_iou),
    }


def _deserialize_record(payload: dict, data_root: Path) -> PredictedCropRecord:
    gt = payload["gt_box"]
    return PredictedCropRecord(
        image_path=(data_root / payload["image_path"]).resolve(),
        class_id=int(payload["class_id"]),
        gt_box=Box(
            class_id=int(gt["class_id"]),
            x_center=float(gt["x_center"]),
            y_center=float(gt["y_center"]),
            width=float(gt["width"]),
            height=float(gt["height"]),
        ),
        predicted_xyxy=tuple(float(value) for value in payload["predicted_xyxy"]),
        predicted_class_id=int(payload["predicted_class_id"]),
        predicted_confidence=float(payload["predicted_confidence"]),
        matched_iou=float(payload["matched_iou"]),
    )


def collect_predicted_crop_records(
    detector_checkpoint: str | Path,
    data_root: str | Path,
    split: str,
    cache_path: str | Path,
    *,
    device: str | None = None,
    image_size: int = 640,
    confidence_threshold: float = 0.001,
    nms_iou: float = 0.7,
    match_iou: float = 0.5,
    max_det: int = 300,
) -> tuple[list[PredictedCropRecord], dict[int, str], dict]:
    """Run the frozen detector and cache raw-image predicted crop records."""

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang. Jalankan `pip install -e .`.") from error

    data_root = Path(data_root).expanduser().resolve()
    detector_checkpoint = Path(detector_checkpoint).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    if not detector_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint detector tidak ditemukan: {detector_checkpoint}")
    if (data_root / "test").exists():
        raise RuntimeError("DC2 predicted-crop screening menolak dataset yang mengekspos test")

    gt_records, names = collect_crop_records(data_root, split)
    grouped: dict[Path, list] = defaultdict(list)
    for record in gt_records:
        grouped[record.image_path].append(record)
    checkpoint_hash = _sha256_file(detector_checkpoint)
    expected = {
        "protocol": "faruq-v3-dc2-predicted-crop-cache-v1",
        "split": split,
        "checkpoint_sha256": checkpoint_hash,
        "image_size": int(image_size),
        "confidence_threshold": float(confidence_threshold),
        "nms_iou": float(nms_iou),
        "match_iou": float(match_iou),
        "max_det": int(max_det),
        "target_instances": len(gt_records),
    }
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(payload.get(key) == value for key, value in expected.items()):
            records = [_deserialize_record(item, data_root) for item in payload["records"]]
            return records, names, {key: value for key, value in payload.items() if key != "records"}

    model = YOLO(str(detector_checkpoint))
    matched_records: list[PredictedCropRecord] = []
    matched_ious: list[float] = []
    detector_correct = 0
    for image_index, (image_path, targets) in enumerate(sorted(grouped.items()), 1):
        with Image.open(image_path) as opened:
            width, height = opened.size
        target_xyxy = [box_to_xyxy(item.box, width, height, 1.0) for item in targets]
        predict_kwargs = {
            "source": str(image_path),
            "imgsz": int(image_size),
            "conf": float(confidence_threshold),
            "iou": float(nms_iou),
            "max_det": int(max_det),
            "verbose": False,
        }
        if device is not None:
            predict_kwargs["device"] = device
        result = model.predict(**predict_kwargs)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        prediction_xyxy = [
            tuple(float(value) for value in row)
            for row in boxes.xyxy.detach().cpu().tolist()
        ]
        prediction_scores = [float(value) for value in boxes.conf.detach().cpu().tolist()]
        prediction_classes = [int(value) for value in boxes.cls.detach().cpu().tolist()]
        matches = greedy_match_xyxy(
            prediction_xyxy,
            prediction_scores,
            target_xyxy,
            iou_threshold=match_iou,
        )
        for prediction_index, target_index, overlap in matches:
            target = targets[target_index]
            predicted_class = prediction_classes[prediction_index]
            matched_records.append(
                PredictedCropRecord(
                    image_path=image_path,
                    class_id=int(target.class_id),
                    gt_box=target.box,
                    predicted_xyxy=prediction_xyxy[prediction_index],
                    predicted_class_id=predicted_class,
                    predicted_confidence=prediction_scores[prediction_index],
                    matched_iou=float(overlap),
                )
            )
            matched_ious.append(float(overlap))
            detector_correct += int(predicted_class == int(target.class_id))
        if image_index % 100 == 0 or image_index == len(grouped):
            print(
                f"DC2 predicted {split}: {image_index}/{len(grouped)} images | "
                f"matched={len(matched_records)}/{len(gt_records)}",
                flush=True,
            )

    if not matched_records:
        raise RuntimeError(f"Tidak ada predicted crop yang cocok pada split {split}")
    metadata = {
        **expected,
        "matched_instances": len(matched_records),
        "matched_coverage": len(matched_records) / max(len(gt_records), 1),
        "mean_matched_iou": sum(matched_ious) / len(matched_ious),
        "native_matched_accuracy": detector_correct / len(matched_records),
        "test_images_accessed": False,
        "test_opened": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        **metadata,
        "records": [_serialize_record(record, data_root) for record in matched_records],
    }
    cache_path.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return matched_records, names, metadata
