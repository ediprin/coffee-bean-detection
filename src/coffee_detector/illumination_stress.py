from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class IlluminationCondition:
    code: str
    family: str
    severity: float
    exposure_ev: float = 0.0
    contrast: float = 1.0
    red_gain: float = 1.0
    green_gain: float = 1.0
    blue_gain: float = 1.0
    shadow_minimum: float = 1.0

    @property
    def is_clean(self) -> bool:
        return self.code == "clean"


CONDITIONS = (
    IlluminationCondition("clean", "clean", 0.0),
    IlluminationCondition("dark_ev05", "exposure_dark", 0.5, exposure_ev=-0.5),
    IlluminationCondition("dark_ev10", "exposure_dark", 1.0, exposure_ev=-1.0),
    IlluminationCondition("bright_ev05", "exposure_bright", 0.5, exposure_ev=0.5),
    IlluminationCondition("bright_ev10", "exposure_bright", 1.0, exposure_ev=1.0),
    IlluminationCondition("contrast075", "contrast_low", 1.0, contrast=0.75),
    IlluminationCondition("contrast125", "contrast_high", 1.0, contrast=1.25),
    IlluminationCondition(
        "warm", "color_temperature_warm", 1.0, red_gain=1.12, blue_gain=0.88
    ),
    IlluminationCondition(
        "cool", "color_temperature_cool", 1.0, red_gain=0.88, blue_gain=1.12
    ),
    IlluminationCondition("shadow55", "localized_shadow", 1.0, shadow_minimum=0.55),
)
CONDITION_BY_CODE = {condition.code: condition for condition in CONDITIONS}


def _orientation(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:2], 16) % 4


def _shadow_map_numpy(height: int, width: int, minimum: float, key: str) -> np.ndarray:
    orientation = _orientation(key)
    x = np.linspace(float(minimum), 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(float(minimum), 1.0, height, dtype=np.float32)[:, None]
    if orientation == 0:
        value = np.broadcast_to(x, (height, width))
    elif orientation == 1:
        value = np.broadcast_to(x[:, ::-1], (height, width))
    elif orientation == 2:
        value = np.broadcast_to(y, (height, width))
    else:
        value = np.broadcast_to(y[::-1, :], (height, width))
    return value[..., None]


def _shadow_map_tensor(
    height: int,
    width: int,
    minimum: float,
    key: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    orientation = _orientation(key)
    x = torch.linspace(float(minimum), 1.0, width, device=device, dtype=dtype)
    y = torch.linspace(float(minimum), 1.0, height, device=device, dtype=dtype)
    if orientation == 0:
        value = x.view(1, width).expand(height, width)
    elif orientation == 1:
        value = x.flip(0).view(1, width).expand(height, width)
    elif orientation == 2:
        value = y.view(height, 1).expand(height, width)
    else:
        value = y.flip(0).view(height, 1).expand(height, width)
    return value.view(1, height, width)


def apply_illumination(
    image: Image.Image, condition: IlluminationCondition, *, key: str
) -> Image.Image:
    """PIL implementation used only for the visual audit preview."""

    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    original_shape = array.shape
    array *= 2.0 ** float(condition.exposure_ev)
    if condition.contrast != 1.0:
        center = float(array.mean())
        array = (array - center) * float(condition.contrast) + center
    array *= np.asarray(
        [condition.red_gain, condition.green_gain, condition.blue_gain],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    if condition.shadow_minimum < 1.0:
        array *= _shadow_map_numpy(
            array.shape[0], array.shape[1], condition.shadow_minimum, key
        )
    output = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    if output.shape != original_shape:
        raise RuntimeError("Transformasi pencahayaan mengubah geometri gambar")
    return Image.fromarray(output, mode="RGB")


def apply_illumination_tensor(
    images: torch.Tensor,
    condition: IlluminationCondition,
    *,
    keys: list[str] | tuple[str, ...],
) -> torch.Tensor:
    """Apply photometric stress after native YOLO spatial preprocessing.

    Input and output are BCHW tensors in [0, 1]. No spatial operation or label
    mutation occurs. ``clean`` returns an exact clone for a codec-free control.
    """

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Tensor illumination harus BCHW dengan tiga kanal")
    if len(keys) != images.shape[0]:
        raise ValueError("Jumlah identity key tidak cocok dengan batch")
    output = images.clone()
    output *= 2.0 ** float(condition.exposure_ev)
    if condition.contrast != 1.0:
        center = output.mean(dim=(1, 2, 3), keepdim=True)
        output = (output - center) * float(condition.contrast) + center
    gains = output.new_tensor(
        [condition.red_gain, condition.green_gain, condition.blue_gain]
    ).view(1, 3, 1, 1)
    output *= gains
    if condition.shadow_minimum < 1.0:
        height, width = output.shape[-2:]
        for index, key in enumerate(keys):
            output[index] *= _shadow_map_tensor(
                height,
                width,
                condition.shadow_minimum,
                str(key),
                device=output.device,
                dtype=output.dtype,
            )
    return output.clamp_(0.0, 1.0)


def make_illumination_validator(condition: IlluminationCondition):
    """Bind a condition to the native Ultralytics detection validator."""

    from ultralytics.models.yolo.detect import DetectionValidator

    class ControlledIlluminationValidator(DetectionValidator):
        def preprocess(self, batch: dict) -> dict:
            batch = super().preprocess(batch)
            keys = [str(path) for path in batch.get("im_file", [])]
            batch["img"] = apply_illumination_tensor(
                batch["img"], condition, keys=keys
            )
            return batch

    ControlledIlluminationValidator.__name__ = (
        f"ControlledIlluminationValidator_{condition.code}"
    )
    return ControlledIlluminationValidator


def make_illumination_preview(
    source_root: str | Path, output: str | Path, *, limit: int = 4
) -> Path:
    source_images = Path(source_root).expanduser().resolve() / "val/images"
    paths = sorted(
        path for path in source_images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )[:limit]
    tile_width, tile_height, label_height = 192, 144, 24
    sheet = Image.new(
        "RGB",
        (tile_width * len(CONDITIONS), (tile_height + label_height) * len(paths)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, path in enumerate(paths):
        with Image.open(path) as source:
            for column, condition in enumerate(CONDITIONS):
                transformed = apply_illumination(source, condition, key=path.name)
                transformed.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
                x = column * tile_width + (tile_width - transformed.width) // 2
                y = row * (tile_height + label_height)
                sheet.paste(transformed, (x, y))
                draw.text(
                    (column * tile_width + 4, y + tile_height + 4),
                    condition.code,
                    fill="black",
                )
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)
    return output
