from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from ..dataset import IMAGE_SUFFIXES, DatasetLayout, parse_label


@dataclass(frozen=True)
class SceneCalibration:
    """Compact empirical calibration learned from the real train split only."""

    scene_counts: tuple[int, ...]
    object_long_sides: tuple[float, ...]
    bbox_width_height_ratios: tuple[float, ...]
    scene_scale_medians: tuple[float, ...]
    within_scene_scale_ratios: tuple[float, ...]
    background_colors: tuple[tuple[float, float, float], ...]
    background_gradient_std: tuple[float, ...]
    background_sensor_std: tuple[float, ...]
    source_images: int
    source_boxes: int
    bbox_width_height_ratios_by_class: dict[int, tuple[float, ...]] = field(
        default_factory=dict
    )
    split: str = "train"

    def __post_init__(self) -> None:
        if (
            not self.scene_counts
            or not self.object_long_sides
            or not self.bbox_width_height_ratios
            or not self.scene_scale_medians
            or not self.within_scene_scale_ratios
        ):
            raise ValueError("Kalibrasi scene harus memiliki count dan skala objek")
        if any(value <= 0 for value in self.scene_counts):
            raise ValueError("Scene count harus positif")
        if any(not 0 < value <= 1 for value in self.object_long_sides):
            raise ValueError("Skala long-side harus berada pada (0, 1]")
        if any(value <= 0 for value in self.bbox_width_height_ratios):
            raise ValueError("Rasio width/height bounding box harus positif")
        if any(
            value <= 0
            for values in self.bbox_width_height_ratios_by_class.values()
            for value in values
        ):
            raise ValueError("Rasio bbox per kelas harus positif")

    def to_payload(self) -> dict:
        return {
            "format": "coffee_detector.scene_calibration.v3",
            **asdict(self),
            "summary": calibration_summary(self),
        }


def _quantiles(values: tuple[float, ...] | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    names = ("q05", "q25", "q50", "q75", "q95")
    points = np.quantile(array, (0.05, 0.25, 0.50, 0.75, 0.95))
    return {name: float(value) for name, value in zip(names, points)}


def calibration_summary(calibration: SceneCalibration) -> dict:
    colors = np.asarray(calibration.background_colors, dtype=np.float64)
    return {
        "split": calibration.split,
        "source_images": calibration.source_images,
        "source_boxes": calibration.source_boxes,
        "scene_count": _quantiles(calibration.scene_counts),
        "object_long_side_fraction": _quantiles(calibration.object_long_sides),
        "bbox_width_height_ratio": _quantiles(
            calibration.bbox_width_height_ratios
        ),
        "bbox_absolute_aspect_ratio": _quantiles(
            tuple(
                max(value, 1.0 / value)
                for value in calibration.bbox_width_height_ratios
            )
        ),
        "bbox_width_height_ratio_by_class": {
            str(class_id): _quantiles(values)
            for class_id, values in sorted(
                calibration.bbox_width_height_ratios_by_class.items()
            )
            if values
        },
        "scene_scale_median_fraction": _quantiles(calibration.scene_scale_medians),
        "within_scene_scale_ratio": _quantiles(calibration.within_scene_scale_ratios),
        "background_rgb_median": (
            [float(value) for value in np.median(colors, axis=0)]
            if colors.size
            else []
        ),
        "background_gradient_std": (
            _quantiles(calibration.background_gradient_std)
            if calibration.background_gradient_std
            else {}
        ),
        "background_sensor_std": (
            _quantiles(calibration.background_sensor_std)
            if calibration.background_sensor_std
            else {}
        ),
    }


def _quantile_sample(values: list[float], limit: int = 1024) -> tuple[float, ...]:
    if not values:
        return ()
    array = np.asarray(values, dtype=np.float64)
    if array.size <= limit:
        return tuple(float(value) for value in array)
    points = np.linspace(0.0, 1.0, limit)
    return tuple(float(value) for value in np.quantile(array, points))


def _background_statistics(
    image_path: Path,
    boxes,
) -> tuple[tuple[float, float, float], float, float] | None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    scale = min(1.0, 256.0 / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    pixels = np.asarray(image, dtype=np.float32)
    height, width = pixels.shape[:2]
    background = np.ones((height, width), dtype=bool)
    # A small safety margin removes bean edges and their cast shadows.
    for box in boxes:
        margin = 0.035
        left = max(0, int(math.floor((box.x_center - box.width / 2 - margin) * width)))
        right = min(width, int(math.ceil((box.x_center + box.width / 2 + margin) * width)))
        top = max(0, int(math.floor((box.y_center - box.height / 2 - margin) * height)))
        bottom = min(height, int(math.ceil((box.y_center + box.height / 2 + margin) * height)))
        background[top:bottom, left:right] = False
    if int(background.sum()) < max(256, int(background.size * 0.02)):
        return None
    blurred = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=max(2.0, max(image.size) / 45.0))),
        dtype=np.float32,
    )
    values = pixels[background]
    smooth = blurred[background]
    median = tuple(float(value) for value in np.median(values, axis=0))
    luminance = smooth.mean(axis=1)
    gradient_std = float(np.std(luminance))
    residual = (values - smooth).mean(axis=1)
    sensor_std = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    return median, gradient_std, sensor_std


def build_scene_calibration(
    layout: DatasetLayout,
    *,
    split: str = "train",
    seed: int = 42,
    background_samples: int = 64,
) -> SceneCalibration:
    if split not in layout.splits:
        raise FileNotFoundError(f"Split kalibrasi tidak ditemukan: {split}")
    image_root, label_root = layout.splits[split]
    image_paths = sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    valid_ids = set(layout.names)
    counts: list[float] = []
    long_sides: list[float] = []
    bbox_width_height_ratios: list[float] = []
    bbox_width_height_ratios_by_class: dict[int, list[float]] = defaultdict(list)
    scene_scale_medians: list[float] = []
    within_scene_scale_ratios: list[float] = []
    labels_by_image: dict[Path, tuple] = {}
    for index, image_path in enumerate(image_paths, 1):
        relative = image_path.relative_to(image_root)
        boxes = parse_label((label_root / relative).with_suffix(".txt"), valid_ids)
        labels_by_image[image_path] = boxes
        if boxes:
            counts.append(float(len(boxes)))
            image_scales = [max(box.width, box.height) for box in boxes]
            bbox_width_height_ratios.extend(
                box.width / box.height for box in boxes if box.height > 0
            )
            for box in boxes:
                if box.height > 0:
                    bbox_width_height_ratios_by_class[box.class_id].append(
                        box.width / box.height
                    )
            median_scale = float(np.median(image_scales))
            long_sides.extend(image_scales)
            scene_scale_medians.append(median_scale)
            within_scene_scale_ratios.extend(
                value / median_scale for value in image_scales if median_scale > 0
            )
        if index % 2000 == 0 or index == len(image_paths):
            print(f"  kalibrasi geometri: {index}/{len(image_paths)} gambar", flush=True)
    if not counts or not long_sides:
        raise RuntimeError("Split train tidak memiliki bounding box untuk kalibrasi")

    # Remove annotation outliers without assuming a parametric distribution.
    scale_array = np.asarray(long_sides, dtype=np.float64)
    low, high = np.quantile(scale_array, (0.01, 0.99))
    long_sides = [value for value in long_sides if low <= value <= high]
    ratio_array = np.asarray(within_scene_scale_ratios, dtype=np.float64)
    ratio_low, ratio_high = np.quantile(ratio_array, (0.01, 0.99))
    within_scene_scale_ratios = [
        value
        for value in within_scene_scale_ratios
        if ratio_low <= value <= ratio_high
    ]
    bbox_ratio_array = np.asarray(bbox_width_height_ratios, dtype=np.float64)
    bbox_ratio_low, bbox_ratio_high = np.quantile(bbox_ratio_array, (0.01, 0.99))
    bbox_width_height_ratios = [
        value
        for value in bbox_width_height_ratios
        if bbox_ratio_low <= value <= bbox_ratio_high
    ]
    trimmed_ratios_by_class: dict[int, tuple[float, ...]] = {}
    for class_id, values in sorted(bbox_width_height_ratios_by_class.items()):
        if len(values) >= 20:
            array = np.asarray(values, dtype=np.float64)
            class_low, class_high = np.quantile(array, (0.01, 0.99))
            trimmed = [
                value for value in values if class_low <= value <= class_high
            ]
        else:
            trimmed = values
        trimmed_ratios_by_class[class_id] = _quantile_sample(trimmed, 512)

    rng = random.Random(seed)
    candidates = image_paths.copy()
    rng.shuffle(candidates)
    colors: list[tuple[float, float, float]] = []
    gradients: list[float] = []
    sensors: list[float] = []
    for image_path in candidates:
        statistics = _background_statistics(image_path, labels_by_image[image_path])
        if statistics is None:
            continue
        color, gradient, sensor = statistics
        colors.append(color)
        gradients.append(gradient)
        sensors.append(sensor)
        if len(colors) >= background_samples:
            break
    if not colors:
        colors = [(226.0, 226.0, 226.0)]
        gradients = [4.0]
        sensors = [1.5]

    return SceneCalibration(
        scene_counts=tuple(int(round(value)) for value in _quantile_sample(counts, 512)),
        object_long_sides=_quantile_sample(long_sides, 1024),
        bbox_width_height_ratios=_quantile_sample(
            bbox_width_height_ratios, 1024
        ),
        scene_scale_medians=_quantile_sample(scene_scale_medians, 512),
        within_scene_scale_ratios=_quantile_sample(within_scene_scale_ratios, 1024),
        background_colors=tuple(colors),
        background_gradient_std=tuple(gradients),
        background_sensor_std=tuple(sensors),
        source_images=len(image_paths),
        source_boxes=int(sum(counts)),
        bbox_width_height_ratios_by_class=trimmed_ratios_by_class,
        split=split,
    )


def save_scene_calibration(calibration: SceneCalibration, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calibration.to_payload(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_scene_calibration(path: str | Path) -> SceneCalibration:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") not in {
        "coffee_detector.scene_calibration.v1",
        "coffee_detector.scene_calibration.v2",
        "coffee_detector.scene_calibration.v3",
    }:
        raise ValueError(f"Format scene calibration tidak dikenal: {path}")
    if "bbox_width_height_ratios" not in payload:
        raise ValueError(
            "Scene calibration lama belum memiliki distribusi aspect ratio; "
            f"bangun ulang profil dengan profile_vadcp_source: {path}"
        )
    return SceneCalibration(
        scene_counts=tuple(int(value) for value in payload["scene_counts"]),
        object_long_sides=tuple(float(value) for value in payload["object_long_sides"]),
        bbox_width_height_ratios=tuple(
            float(value) for value in payload["bbox_width_height_ratios"]
        ),
        scene_scale_medians=tuple(
            float(value)
            for value in payload.get("scene_scale_medians", payload["object_long_sides"])
        ),
        within_scene_scale_ratios=tuple(
            float(value) for value in payload.get("within_scene_scale_ratios", [1.0])
        ),
        background_colors=tuple(
            tuple(float(channel) for channel in value)
            for value in payload["background_colors"]
        ),
        background_gradient_std=tuple(
            float(value) for value in payload["background_gradient_std"]
        ),
        background_sensor_std=tuple(
            float(value) for value in payload["background_sensor_std"]
        ),
        source_images=int(payload["source_images"]),
        source_boxes=int(payload["source_boxes"]),
        bbox_width_height_ratios_by_class={
            int(class_id): tuple(float(value) for value in values)
            for class_id, values in payload.get(
                "bbox_width_height_ratios_by_class", {}
            ).items()
        },
        split=str(payload.get("split", "train")),
    )
