from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout, parse_label
from .msfa import (
    DC2MSFAClassifier,
    GLOBAL_LEVELS,
    _gap_roi,
    _letterbox_image_and_boxes,
    _pyramid_spec,
)


IOU_THRESHOLDS = tuple(float(value) for value in np.arange(0.50, 0.96, 0.05))


@dataclass(frozen=True)
class IntegratedPredictionRecord:
    image_path: Path
    predicted_xyxy: tuple[float, float, float, float]
    predicted_class_id: int
    predicted_confidence: float


@dataclass(frozen=True)
class GroundTruthRecord:
    image_path: Path
    class_id: int
    xyxy: tuple[float, float, float, float]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_signature(records: list[IntegratedPredictionRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.image_path).encode("utf-8"))
        digest.update(
            (
                f"{record.predicted_xyxy[0]:.5f}:{record.predicted_xyxy[1]:.5f}:"
                f"{record.predicted_xyxy[2]:.5f}:{record.predicted_xyxy[3]:.5f}:"
                f"{record.predicted_class_id}:{record.predicted_confidence:.8f}"
            ).encode("ascii")
        )
    return digest.hexdigest()


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


class IntegratedPredictedCropDataset(Dataset):
    """Raw-RGB crops for every detector prediction, including unmatched false positives."""

    def __init__(
        self,
        records: list[IntegratedPredictionRecord],
        resolution: int,
        *,
        context: float = 1.0,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution harus positif")
        self.records = records
        self.resolution = int(resolution)
        self.context = float(context)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
            crop = image.crop(
                _expand_xyxy(
                    record.predicted_xyxy,
                    image.width,
                    image.height,
                    self.context,
                )
            )
            return self.transform(crop)


def collect_ground_truth_records(
    data_root: str | Path,
    split: str,
) -> tuple[list[GroundTruthRecord], dict[int, str], list[Path]]:
    data_root = Path(data_root).expanduser().resolve()
    layout = discover_layout(data_root)
    if split not in layout.splits:
        raise FileNotFoundError(f"Split {split} tidak ditemukan")
    image_root, label_root = layout.splits[split]
    valid_ids = set(layout.names)
    images = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    records: list[GroundTruthRecord] = []
    for image_path in images:
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        with Image.open(image_path) as opened:
            width, height = opened.size
        for box in parse_label(label_path, valid_ids):
            x1 = (box.x_center - box.width / 2.0) * width
            y1 = (box.y_center - box.height / 2.0) * height
            x2 = (box.x_center + box.width / 2.0) * width
            y2 = (box.y_center + box.height / 2.0) * height
            records.append(
                GroundTruthRecord(
                    image_path=image_path,
                    class_id=int(box.class_id),
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
    if not images:
        raise RuntimeError(f"Tidak ada image pada split {split}")
    return records, layout.names, images


def _serialize_prediction(record: IntegratedPredictionRecord, data_root: Path) -> dict:
    return {
        "image_path": str(record.image_path.relative_to(data_root)),
        "predicted_xyxy": [float(value) for value in record.predicted_xyxy],
        "predicted_class_id": int(record.predicted_class_id),
        "predicted_confidence": float(record.predicted_confidence),
    }


def _deserialize_prediction(payload: dict, data_root: Path) -> IntegratedPredictionRecord:
    return IntegratedPredictionRecord(
        image_path=(data_root / payload["image_path"]).resolve(),
        predicted_xyxy=tuple(float(value) for value in payload["predicted_xyxy"]),
        predicted_class_id=int(payload["predicted_class_id"]),
        predicted_confidence=float(payload["predicted_confidence"]),
    )


def collect_all_detector_predictions(
    detector_checkpoint: str | Path,
    data_root: str | Path,
    split: str,
    cache_path: str | Path,
    *,
    device: str | None = None,
    image_size: int = 640,
    confidence_threshold: float = 0.001,
    nms_iou: float = 0.7,
    max_det: int = 300,
) -> tuple[list[IntegratedPredictionRecord], dict[int, str], dict]:
    """Collect every post-NMS prediction without GT matching."""

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang") from error

    data_root = Path(data_root).expanduser().resolve()
    checkpoint = Path(detector_checkpoint).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if (data_root / "test").exists():
        raise RuntimeError("DC2 integrated screening menolak dataset yang mengekspos test")

    _gt, names, images = collect_ground_truth_records(data_root, split)
    expected = {
        "protocol": "faruq-v3-dc2-integrated-prediction-cache-v1",
        "split": split,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "image_size": int(image_size),
        "confidence_threshold": float(confidence_threshold),
        "nms_iou": float(nms_iou),
        "max_det": int(max_det),
        "agnostic_nms": True,
        "image_count": len(images),
    }
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(payload.get(key) == value for key, value in expected.items()):
            records = [_deserialize_prediction(item, data_root) for item in payload["records"]]
            metadata = {key: value for key, value in payload.items() if key != "records"}
            return records, names, metadata

    model = YOLO(str(checkpoint))
    records: list[IntegratedPredictionRecord] = []
    images_with_predictions = 0
    for index, image_path in enumerate(images, 1):
        kwargs = {
            "source": str(image_path),
            "imgsz": int(image_size),
            "conf": float(confidence_threshold),
            "iou": float(nms_iou),
            "max_det": int(max_det),
            "agnostic_nms": True,
            "verbose": False,
        }
        if device is not None:
            kwargs["device"] = device
        result = model.predict(**kwargs)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            images_with_predictions += 1
            xyxy = boxes.xyxy.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            confidence = boxes.conf.detach().cpu().tolist()
            for box, class_id, score in zip(xyxy, classes, confidence):
                records.append(
                    IntegratedPredictionRecord(
                        image_path=image_path,
                        predicted_xyxy=tuple(float(value) for value in box),
                        predicted_class_id=int(class_id),
                        predicted_confidence=float(score),
                    )
                )
        if index % 100 == 0 or index == len(images):
            print(
                f"DC2d detector {split}: {index}/{len(images)} images | predictions={len(records)}",
                flush=True,
            )

    metadata = {
        **expected,
        "prediction_count": len(records),
        "images_with_predictions": images_with_predictions,
        "test_images_accessed": False,
        "test_opened": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                **metadata,
                "records": [_serialize_prediction(record, data_root) for record in records],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return records, names, metadata


def extract_integrated_global_descriptors(
    detector_checkpoint: str | Path,
    records: list[IntegratedPredictionRecord],
    cache_path: str | Path,
    *,
    split: str,
    device: str = "0",
    image_size: int = 640,
) -> tuple[np.ndarray, dict]:
    """Extract frozen P3/P4/P5 descriptors for every detector prediction."""

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang") from error

    checkpoint = Path(detector_checkpoint).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    if not records:
        raise RuntimeError("Tidak ada detector prediction untuk dievaluasi")
    torch_device = torch.device("cpu" if str(device) == "cpu" else f"cuda:{device}")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia")

    expected = {
        "protocol": "faruq-v3-dc2-integrated-global-cache-v1",
        "split": split,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "record_signature": _record_signature(records),
        "record_count": len(records),
        "image_size": int(image_size),
        "levels": list(GLOBAL_LEVELS),
    }
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            matrix = np.load(cache_path, allow_pickle=False)["global_features"].astype(
                np.float32, copy=False
            )
            if len(matrix) != len(records):
                raise RuntimeError("Global cache tidak sejajar")
            return matrix, metadata

    network = YOLO(str(checkpoint)).model.to(torch_device).eval()
    spec = _pyramid_spec(network)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, index, _stride in spec:
        def capture(_module, _inputs, output, *, feature_name=name):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Output {feature_name} bukan tensor")
            captured[feature_name] = output.detach()
        handles.append(network.model[index].register_forward_hook(capture))

    grouped: dict[Path, list[tuple[int, IntegratedPredictionRecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.image_path].append((index, record))
    rows: list[np.ndarray | None] = [None] * len(records)
    feature_dims: dict[str, int] = {}
    try:
        with torch.inference_mode():
            for image_index, (image_path, items) in enumerate(sorted(grouped.items()), 1):
                original_boxes = [record.predicted_xyxy for _, record in items]
                image, boxes = _letterbox_image_and_boxes(
                    image_path, original_boxes, image_size, torch_device
                )
                captured.clear()
                network(image)
                if set(captured) != set(GLOBAL_LEVELS):
                    raise RuntimeError(f"Hook global tidak lengkap: {sorted(captured)}")
                descriptors = []
                for level in GLOBAL_LEVELS:
                    current = _gap_roi(captured[level], boxes, image_size)
                    feature_dims[level] = int(current.shape[1])
                    descriptors.append(current)
                fused = torch.cat(descriptors, dim=1).float().cpu().numpy().astype(np.float32)
                for row_index, (record_index, _record) in enumerate(items):
                    rows[record_index] = fused[row_index]
                if image_index % 100 == 0 or image_index == len(grouped):
                    print(
                        f"DC2d global {split}: {image_index}/{len(grouped)} images",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    if any(row is None for row in rows):
        raise RuntimeError("Ada prediction tanpa global descriptor")
    matrix = np.stack(rows, axis=0).astype(np.float32)
    metadata = {
        **expected,
        "feature_dimensions": feature_dims,
        "global_dimensions": int(matrix.shape[1]),
        "detector_training_executed": False,
        "test_images_accessed": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, global_features=matrix)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return matrix, metadata


@torch.inference_mode()
def classify_integrated_predictions(
    records: list[IntegratedPredictionRecord],
    global_descriptors: np.ndarray,
    classifier_checkpoint: str | Path,
    resolution: int,
    *,
    device: str = "0",
    batch_size: int = 64,
    workers: int = 2,
) -> tuple[list[IntegratedPredictionRecord], dict]:
    checkpoint = Path(classifier_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if len(records) != len(global_descriptors):
        raise ValueError("Prediction dan global descriptor tidak sejajar")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    global_dim = int(payload["global_dim"])
    num_classes = int(payload["num_classes"])
    if global_descriptors.shape[1] != global_dim:
        raise RuntimeError("Dimensi global descriptor tidak cocok dengan checkpoint")
    model = DC2MSFAClassifier(num_classes, global_dim, imagenet_pretrained=False)
    model.load_state_dict(payload["model"], strict=True)
    torch_device = torch.device("cpu" if str(device) == "cpu" else f"cuda:{device}")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia")
    model = model.to(torch_device).eval()
    dataset = IntegratedPredictedCropDataset(records, resolution)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    predictions: list[int] = []
    max_probabilities: list[float] = []
    offset = 0
    for images in loader:
        count = len(images)
        images = images.to(torch_device, non_blocking=True)
        global_tensor = torch.from_numpy(
            np.asarray(global_descriptors[offset : offset + count], dtype=np.float32)
        ).to(torch_device, non_blocking=True)
        logits = model(images, global_tensor, enable_global=True)
        probabilities = logits.softmax(dim=1)
        score, label = probabilities.max(dim=1)
        predictions.extend(int(value) for value in label.cpu().tolist())
        max_probabilities.extend(float(value) for value in score.cpu().tolist())
        offset += count
    if offset != len(records):
        raise RuntimeError("Jumlah classifier output tidak cocok")

    refined = [
        IntegratedPredictionRecord(
            image_path=record.image_path,
            predicted_xyxy=record.predicted_xyxy,
            predicted_class_id=class_id,
            predicted_confidence=record.predicted_confidence,
        )
        for record, class_id in zip(records, predictions)
    ]
    metadata = {
        "classifier_checkpoint": str(checkpoint),
        "classifier_checkpoint_sha256": _sha256_file(checkpoint),
        "resolution": int(resolution),
        "prediction_count": len(refined),
        "mean_classifier_max_probability": float(np.mean(max_probabilities)),
        "score_policy": "preserve_detector_confidence_replace_class_only",
        "test_images_accessed": False,
    }
    return refined, metadata


def _box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in left)
    bx1, by1, bx2, by2 = (float(value) for value in right)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def detection_map_summary(
    predictions: list[IntegratedPredictionRecord],
    ground_truth: list[GroundTruthRecord],
    num_classes: int,
    *,
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS,
) -> dict:
    """Paired class-aware AP using fixed detector boxes and confidence scores."""

    gt_by_class_image: dict[int, dict[Path, list[GroundTruthRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in ground_truth:
        gt_by_class_image[int(record.class_id)][record.image_path].append(record)
    predictions_by_class: dict[int, list[IntegratedPredictionRecord]] = defaultdict(list)
    for record in predictions:
        predictions_by_class[int(record.predicted_class_id)].append(record)

    per_class_ap: list[float] = []
    per_class_ap50: list[float] = []
    gt_counts: list[int] = []
    for class_id in range(int(num_classes)):
        class_gt = gt_by_class_image.get(class_id, {})
        n_gt = sum(len(value) for value in class_gt.values())
        gt_counts.append(n_gt)
        class_predictions = sorted(
            predictions_by_class.get(class_id, []),
            key=lambda record: -float(record.predicted_confidence),
        )
        class_aps: list[float] = []
        for threshold in iou_thresholds:
            matched = {
                image_path: [False] * len(items)
                for image_path, items in class_gt.items()
            }
            tp = np.zeros(len(class_predictions), dtype=np.float64)
            fp = np.zeros(len(class_predictions), dtype=np.float64)
            for prediction_index, prediction in enumerate(class_predictions):
                candidates = class_gt.get(prediction.image_path, [])
                best_iou = -1.0
                best_index = -1
                for gt_index, target in enumerate(candidates):
                    if matched[prediction.image_path][gt_index]:
                        continue
                    overlap = _box_iou(prediction.predicted_xyxy, target.xyxy)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_index = gt_index
                if best_index >= 0 and best_iou >= float(threshold):
                    matched[prediction.image_path][best_index] = True
                    tp[prediction_index] = 1.0
                else:
                    fp[prediction_index] = 1.0
            if n_gt <= 0:
                class_aps.append(float("nan"))
                continue
            cumulative_tp = np.cumsum(tp)
            cumulative_fp = np.cumsum(fp)
            recall = cumulative_tp / float(n_gt)
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
            class_aps.append(_compute_ap(recall, precision))
        finite = [value for value in class_aps if math.isfinite(value)]
        per_class_ap.append(float(np.mean(finite)) if finite else float("nan"))
        per_class_ap50.append(float(class_aps[0]))

    active = [index for index, count in enumerate(gt_counts) if count > 0]
    if not active:
        raise RuntimeError("Tidak ada GT class aktif")
    active_ap = [per_class_ap[index] for index in active]
    active_ap50 = [per_class_ap50[index] for index in active]
    ordered = sorted(active_ap)
    return {
        "map50_95": float(np.mean(active_ap)),
        "map50": float(np.mean(active_ap50)),
        "bottom3_ap50_95": float(np.mean(ordered[: min(3, len(ordered))])),
        "worst_ap50_95": float(ordered[0]),
        "per_class_ap50_95": {str(index): float(per_class_ap[index]) for index in active},
        "per_class_ap50": {str(index): float(per_class_ap50[index]) for index in active},
        "gt_counts": {str(index): int(gt_counts[index]) for index in active},
        "iou_thresholds": [float(value) for value in iou_thresholds],
        "prediction_count": len(predictions),
        "ground_truth_count": len(ground_truth),
        "ap_interpolation": "101-point precision envelope",
    }


def decide_dc2_integrated(native: dict, refined: dict) -> dict:
    deltas = {
        metric: float(refined[metric] - native[metric])
        for metric in ("map50_95", "bottom3_ap50_95", "worst_ap50_95")
    }
    criteria = {
        "map50_95_gain_vs_native_at_least_0_5_point": deltas["map50_95"] >= 0.005,
        "bottom3_not_lower_than_native": deltas["bottom3_ap50_95"] >= 0.0,
        "worst_drop_vs_native_no_more_than_1_point": deltas["worst_ap50_95"] >= -0.01,
    }
    passed = all(criteria.values())
    return {
        "deltas_refined_vs_native": deltas,
        "criteria": criteria,
        "decision": "PASS" if passed else "FAIL",
        "next_action": (
            "CONSIDER_JOINT_END_TO_END_DC2_TRANSFER"
            if passed
            else "STOP_DC2_ESCALATION_AND_RETURN_TO_BREADTH_SEARCH"
        ),
    }
