from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.ops import roi_align

from coffee_detector.analysis.faruq_v3_pyramid_separability import _pyramid_spec
from coffee_detector.dc2_crop.model import build_local_classifier
from coffee_detector.dc2_crop.predicted import MatchedRawObjectCropDataset, PredictedCropRecord


class DC2MSFAClassifier(nn.Module):
    """Final-stage transfer of DC2 Eq. (8): local + GAP(global ROI).

    The local DC2b classifier is frozen. A zero-initialized projection maps the
    detector P5 global descriptor to the MobileNetV3 pooled local dimension.
    Consequently, initialization reproduces the local-only checkpoint exactly.

    This is an adapted mechanism screen, not a literal multi-stage reproduction
    of every paired block in the original DC2 network.
    """

    def __init__(self, local_model: nn.Module, global_dim: int) -> None:
        super().__init__()
        if global_dim <= 0:
            raise ValueError("global_dim harus positif")
        if not hasattr(local_model, "features") or not hasattr(local_model, "avgpool"):
            raise TypeError("Local model harus menyediakan features dan avgpool")
        classifier = getattr(local_model, "classifier", None)
        if not isinstance(classifier, nn.Sequential) or not len(classifier):
            raise TypeError("Local model harus menyediakan classifier sequential")
        first_linear = next((layer for layer in classifier if isinstance(layer, nn.Linear)), None)
        if first_linear is None:
            raise TypeError("Classifier local tidak memiliki Linear")
        self.local_model = local_model
        self.local_dim = int(first_linear.in_features)
        self.global_dim = int(global_dim)
        for parameter in self.local_model.parameters():
            parameter.requires_grad_(False)
        self.global_projection = nn.Linear(self.global_dim, self.local_dim, bias=True)
        nn.init.zeros_(self.global_projection.weight)
        nn.init.zeros_(self.global_projection.bias)
        self.local_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the transferred DC2b branch fixed, including BN/dropout behavior.
        self.local_model.eval()
        self.global_projection.train(mode)
        return self

    def local_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.local_model.features(images)
        features = self.local_model.avgpool(features)
        return torch.flatten(features, 1)

    def forward(self, images: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        local = self.local_features(images)
        if global_features.ndim != 2 or global_features.shape[0] != local.shape[0]:
            raise ValueError("Global features harus [B,D] dan sejajar dengan local batch")
        if global_features.shape[1] != self.global_dim:
            raise ValueError(
                f"Global dim {global_features.shape[1]} tidak sama dengan {self.global_dim}"
            )
        fused = local + self.global_projection(global_features)
        return self.local_model.classifier(fused)


def load_dc2b_local_checkpoint(path: str | Path, num_classes: int) -> nn.Module:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint local DC2b tidak ditemukan: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if int(payload.get("num_classes", -1)) != int(num_classes):
        raise RuntimeError("Jumlah kelas checkpoint DC2b tidak cocok")
    if int(payload.get("resolution", -1)) != 128 or payload.get("source") != "predicted":
        raise RuntimeError("DC2c memerlukan predicted-crop DC2b 128x128")
    model = build_local_classifier(num_classes, imagenet_pretrained=False)
    model.load_state_dict(payload["model"], strict=True)
    return model


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def record_signature(records: list[PredictedCropRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.image_path).encode("utf-8"))
        digest.update(str(record.class_id).encode("ascii"))
        digest.update(
            ":".join(f"{float(value):.6f}" for value in record.predicted_xyxy).encode("ascii")
        )
        digest.update(f"{record.predicted_confidence:.8f}:{record.matched_iou:.8f}".encode("ascii"))
    return digest.hexdigest()


def _letterbox_image(
    image_path: Path, image_size: int, device: torch.device
) -> tuple[torch.Tensor, tuple[float, float], tuple[float, float]]:
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
    ratio = (float(params["ratio"][0]), float(params["ratio"][1]))
    pad = (float(params["left"]), float(params["top"]))
    return tensor, ratio, pad


def _transform_boxes_to_letterbox(
    records: list[PredictedCropRecord],
    ratio: tuple[float, float],
    pad: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    ratio_x, ratio_y = ratio
    left, top = pad
    rows = []
    for record in records:
        x1, y1, x2, y2 = record.predicted_xyxy
        rows.append(
            [
                float(x1) * ratio_x + left,
                float(y1) * ratio_y + top,
                float(x2) * ratio_x + left,
                float(y2) * ratio_y + top,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device).reshape(-1, 4)


def _gap_roi_descriptor(
    feature: torch.Tensor, boxes: torch.Tensor, image_size: int, roi_size: int
) -> torch.Tensor:
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("P5 feature harus [1,C,H,W]")
    batch = boxes.new_zeros((len(boxes), 1))
    rois = torch.cat((batch, boxes), dim=1)
    aligned = roi_align(
        feature,
        rois,
        output_size=(roi_size, roi_size),
        spatial_scale=float(feature.shape[-1]) / float(image_size),
        sampling_ratio=2,
        aligned=True,
    )
    # Literal operator transferred from DC2 Eq. (8): spatial GAP of global ROI.
    return aligned.mean(dim=(-2, -1))


def extract_p5_global_descriptors(
    detector_checkpoint: str | Path,
    records: list[PredictedCropRecord],
    cache_path: str | Path,
    *,
    detector_sha256: str,
    device: torch.device,
    image_size: int = 640,
    roi_size: int = 3,
) -> tuple[np.ndarray, dict]:
    """Extract frozen detector P5 descriptors on exactly the DC2b predicted boxes."""

    from ultralytics import YOLO

    checkpoint = Path(detector_checkpoint).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    signature = record_signature(records)
    expected = {
        "protocol": "faruq-v3-dc2-msfa-p5-global-cache-v1",
        "detector_sha256": detector_sha256,
        "record_signature": signature,
        "instances": len(records),
        "image_size": int(image_size),
        "roi_size": int(roi_size),
        "global_level": "P5",
        "pooling": "gap",
    }
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            with np.load(cache_path, allow_pickle=False) as payload:
                features = payload["features"].astype(np.float32, copy=False)
            if len(features) == len(records):
                return features, metadata

    network = YOLO(str(checkpoint)).model.to(device).eval()
    spec = _pyramid_spec(network)
    p5 = [item for item in spec if item[0] == "P5"]
    if len(p5) != 1:
        raise RuntimeError(f"P5 detector tidak unik: {spec}")
    _, layer_index, stride = p5[0]
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Output P5 bukan tensor")
        captured["P5"] = output.detach()

    handle = network.model[layer_index].register_forward_hook(capture)
    grouped: dict[Path, list[tuple[int, PredictedCropRecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.image_path].append((index, record))
    rows: list[np.ndarray | None] = [None] * len(records)
    try:
        with torch.inference_mode():
            for image_index, (image_path, indexed) in enumerate(sorted(grouped.items()), 1):
                local_records = [record for _, record in indexed]
                image, ratio, pad = _letterbox_image(image_path, image_size, device)
                boxes = _transform_boxes_to_letterbox(local_records, ratio, pad, device)
                captured.clear()
                network(image)
                if "P5" not in captured:
                    raise RuntimeError("Hook P5 tidak terpanggil")
                descriptor = _gap_roi_descriptor(captured["P5"], boxes, image_size, roi_size)
                descriptor_np = descriptor.float().cpu().numpy().astype(np.float32)
                for row_index, (original_index, _record) in enumerate(indexed):
                    rows[original_index] = descriptor_np[row_index]
                if image_index % 100 == 0 or image_index == len(grouped):
                    completed = sum(item is not None for item in rows)
                    print(
                        f"DC2c P5 GLOBAL {image_index}/{len(grouped)} images | "
                        f"instances={completed}/{len(records)}",
                        flush=True,
                    )
    finally:
        handle.remove()

    if any(row is None for row in rows):
        raise RuntimeError("Ekstraksi global P5 tidak lengkap")
    features = np.stack(rows, axis=0).astype(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("Global P5 mengandung nilai non-finite")
    metadata = {
        **expected,
        "feature_dim": int(features.shape[1]),
        "p5_stride": float(stride),
        "feature_std": float(features.std()),
        "detector_training_executed": False,
        "test_images_accessed": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return features, metadata


class MSFAMatchedDataset(Dataset):
    def __init__(
        self,
        records: list[PredictedCropRecord],
        global_features: np.ndarray,
        *,
        resolution: int = 128,
        training: bool,
    ) -> None:
        if len(records) != len(global_features):
            raise ValueError("Records dan global features tidak sejajar")
        self.local = MatchedRawObjectCropDataset(
            records,
            resolution,
            training=training,
            source="predicted",
            context=1.0,
        )
        self.global_features = torch.from_numpy(
            np.asarray(global_features, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.local)

    def __getitem__(self, index: int):
        image, label = self.local[index]
        return image, self.global_features[index], label
