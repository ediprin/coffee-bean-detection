from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VisibilityBin:
    name: str
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.minimum < self.maximum <= 1.0):
            raise ValueError(
                f"Visibility bin tidak valid: {self.name} "
                f"[{self.minimum}, {self.maximum}]"
            )

    def contains(self, value: float, *, is_last: bool = False) -> bool:
        if is_last:
            return self.minimum <= value <= self.maximum
        return self.minimum <= value < self.maximum


@dataclass(frozen=True)
class Cutout:
    asset_id: str
    class_id: int
    class_name: str
    image_path: Path
    source_id: str
    source_split: str = "train"


@dataclass
class TransformedCutout:
    cutout: Cutout
    rgba: Image.Image
    mask: np.ndarray


@dataclass
class PlacedInstance:
    instance_id: int
    cutout: Cutout
    x: int
    y: int
    z_order: int
    full_mask: np.ndarray
    visible_mask: np.ndarray | None = None
    visibility_ratio: float = 0.0
    visibility_bin: str = "ignored"
    is_focus: bool = False


@dataclass
class SyntheticScene:
    image: Image.Image
    instances: list[PlacedInstance]
    target_visibility_bin: str | None
    target_visibility_hit: bool
