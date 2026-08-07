from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from coffee_detector.dataset import Box, discover_layout, parse_label, IMAGE_SUFFIXES


@dataclass(frozen=True)
class CropRecord:
    image_path: Path
    class_id: int
    box: Box


def collect_crop_records(data_root: str | Path, split: str) -> tuple[list[CropRecord], dict[int, str]]:
    layout = discover_layout(data_root)
    if split not in layout.splits:
        raise FileNotFoundError(f"Split {split} tidak ditemukan di {layout.root}")
    image_root, label_root = layout.splits[split]
    valid_ids = set(layout.names)
    records: list[CropRecord] = []
    for image_path in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        for box in parse_label(label_path, valid_ids):
            records.append(CropRecord(image_path=image_path, class_id=box.class_id, box=box))
    if not records:
        raise RuntimeError(f"Tidak ada crop record pada split {split}")
    return records, layout.names


def box_to_xyxy(box: Box, width: int, height: int, context: float = 1.0) -> tuple[int, int, int, int]:
    if context < 1.0:
        raise ValueError("context minimal 1.0")
    cx = box.x_center * width
    cy = box.y_center * height
    bw = max(1.0, box.width * width * context)
    bh = max(1.0, box.height * height * context)
    left = max(0, int(round(cx - bw / 2)))
    top = max(0, int(round(cy - bh / 2)))
    right = min(width, int(round(cx + bw / 2)))
    bottom = min(height, int(round(cy + bh / 2)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


class RawObjectCropDataset(Dataset):
    """One GT object = one raw RGB crop, preserving the source pixels before resize."""

    def __init__(
        self,
        records: list[CropRecord],
        resolution: int,
        *,
        training: bool,
        context: float = 1.0,
    ) -> None:
        if resolution <= 0:
            raise ValueError("resolution harus positif")
        self.records = records
        self.resolution = int(resolution)
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
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
            crop = image.crop(box_to_xyxy(record.box, image.width, image.height, self.context))
            tensor = self.transform(crop)
        return tensor, int(record.class_id)
