from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.ops import roi_align

from .model import build_local_classifier
from .predicted import MatchedRawObjectCropDataset, PredictedCropRecord


GLOBAL_LEVELS = ("P3", "P4", "P5")


def _checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_signature(records: list[PredictedCropRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.image_path).encode("utf-8"))
        digest.update(
            (
                f"{record.class_id}:"
                f"{record.predicted_xyxy[0]:.5f}:{record.predicted_xyxy[1]:.5f}:"
                f"{record.predicted_xyxy[2]:.5f}:{record.predicted_xyxy[3]:.5f}:"
                f"{record.predicted_class_id}:{record.matched_iou:.6f}"
            ).encode("ascii")
        )
    return digest.hexdigest()


def _pyramid_spec(network: nn.Module) -> list[tuple[str, int, float]]:
    head = network.model[-1]
    indices = list(head.f) if isinstance(head.f, (tuple, list)) else [int(head.f)]
    strides = [float(value) for value in head.stride.detach().cpu().tolist()]
    if len(indices) != len(strides):
        raise ValueError("Jumlah input head dan stride tidak cocok")
    output: list[tuple[str, int, float]] = []
    for index, stride in zip(indices, strides):
        level = int(round(math.log2(stride)))
        if not math.isclose(stride, 2**level):
            raise ValueError(f"Stride bukan pangkat dua: {stride}")
        output.append((f"P{level}", int(index), stride))
    if tuple(item[0] for item in output) != GLOBAL_LEVELS:
        raise ValueError(f"DC2 MSFA transfer memerlukan P3/P4/P5, diterima {output}")
    return output


def _letterbox_image_and_boxes(
    image_path: Path,
    boxes_xyxy: list[tuple[float, float, float, float]],
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ultralytics.data.augment import LetterBox

    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Gambar tidak dapat dibaca: {image_path}")
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
    ratio_x, ratio_y = params["ratio"]
    rows = []
    for x1, y1, x2, y2 in boxes_xyxy:
        rows.append(
            [
                float(x1) * ratio_x + params["left"],
                float(y1) * ratio_y + params["top"],
                float(x2) * ratio_x + params["left"],
                float(y2) * ratio_y + params["top"],
            ]
        )
    boxes = torch.tensor(rows, device=device, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 0::2].clamp_(0.0, float(image_size))
    boxes[:, 1::2].clamp_(0.0, float(image_size))
    return tensor, boxes


def _gap_roi(feature: torch.Tensor, boxes: torch.Tensor, image_size: int) -> torch.Tensor:
    """Crop global feature patches then apply spatial GAP, matching DC2 Eq. 5/7/8 principle."""

    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f"Feature map harus [1,C,H,W], diterima {tuple(feature.shape)}")
    batch_column = boxes.new_zeros((len(boxes), 1))
    rois = torch.cat((batch_column, boxes), dim=1)
    cropped = roi_align(
        feature,
        rois,
        output_size=(3, 3),
        spatial_scale=float(feature.shape[-1]) / float(image_size),
        sampling_ratio=2,
        aligned=True,
    )
    return cropped.mean(dim=(-2, -1))


def extract_global_descriptors(
    detector_checkpoint: str | Path,
    records: list[PredictedCropRecord],
    cache_path: str | Path,
    *,
    split: str,
    device: str = "0",
    image_size: int = 640,
) -> tuple[np.ndarray, dict]:
    """Extract frozen YOLO26 P3/P4/P5 global-stream descriptors for predicted boxes."""

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang") from error

    checkpoint = Path(detector_checkpoint).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not records:
        raise ValueError("records kosong")
    torch_device = torch.device("cpu" if str(device) == "cpu" else f"cuda:{device}")
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia")

    expected = {
        "protocol": "faruq-v3-dc2-msfa-global-cache-v1",
        "split": split,
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "record_signature": record_signature(records),
        "record_count": len(records),
        "image_size": int(image_size),
        "levels": list(GLOBAL_LEVELS),
    }
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            matrix = np.load(cache_path, allow_pickle=False)["global"].astype(np.float32, copy=False)
            if len(matrix) != len(records):
                raise RuntimeError("Global cache tidak sejajar dengan predicted records")
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

    grouped: dict[Path, list[tuple[int, PredictedCropRecord]]] = defaultdict(list)
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
                        f"DC2 MSFA global {split}: {image_index}/{len(grouped)} images",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    if any(row is None for row in rows):
        raise RuntimeError("Ada predicted record tanpa global descriptor")
    matrix = np.stack(rows, axis=0).astype(np.float32)
    metadata = {
        **expected,
        "feature_dimensions": feature_dims,
        "global_dimensions": int(matrix.shape[1]),
        "detector_training_executed": False,
        "test_images_accessed": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **{"global": matrix})
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return matrix, metadata


class MatchedCropGlobalDataset(Dataset):
    def __init__(
        self,
        records: list[PredictedCropRecord],
        global_descriptors: np.ndarray,
        resolution: int,
        *,
        training: bool,
    ) -> None:
        if len(records) != len(global_descriptors):
            raise ValueError("records dan global_descriptors tidak sejajar")
        self.crop_dataset = MatchedRawObjectCropDataset(
            records,
            resolution,
            training=training,
            source="predicted",
            context=1.0,
        )
        self.global_descriptors = torch.from_numpy(
            np.asarray(global_descriptors, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.crop_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image, label = self.crop_dataset[index]
        return image, self.global_descriptors[index], label


class DC2MSFAClassifier(nn.Module):
    """YOLO26 adaptation of DC2 Eq. 8 at the terminal local descriptor.

    DC2 uses stage-paired additions where global/local channel dimensions are
    already equal. YOLO26 and MobileNetV3 do not share that stage layout, so
    P3/P4/P5 GAP descriptors are concatenated and linearly projected into the
    MobileNet local descriptor dimension before residual addition. The
    projection is zero-initialized so the injected model starts exactly from
    the local-stream classifier.
    """

    def __init__(
        self,
        num_classes: int,
        global_dim: int,
        *,
        imagenet_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.local_model = build_local_classifier(
            num_classes, imagenet_pretrained=imagenet_pretrained
        )
        local_dim = int(self.local_model.classifier[0].in_features)
        self.global_projection = nn.Linear(int(global_dim), local_dim, bias=True)
        nn.init.zeros_(self.global_projection.weight)
        nn.init.zeros_(self.global_projection.bias)

    def local_descriptor(self, image: torch.Tensor) -> torch.Tensor:
        value = self.local_model.features(image)
        value = self.local_model.avgpool(value)
        return torch.flatten(value, 1)

    def forward(
        self,
        image: torch.Tensor,
        global_descriptor: torch.Tensor,
        *,
        enable_global: bool = True,
    ) -> torch.Tensor:
        local = self.local_descriptor(image)
        if enable_global:
            local = local + self.global_projection(global_descriptor)
        return self.local_model.classifier(local)

    def load_local_checkpoint(self, checkpoint: str | Path) -> None:
        payload = torch.load(
            Path(checkpoint).expanduser().resolve(), map_location="cpu", weights_only=False
        )
        state = payload.get("model", payload)
        self.local_model.load_state_dict(state, strict=True)
