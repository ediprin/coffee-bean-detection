from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .masks import mask_bbox, principal_mask_geometry
from .profile import SceneCalibration
from .types import (
    Cutout,
    PlacedInstance,
    SyntheticScene,
    TransformedCutout,
    VisibilityBin,
)


DEFAULT_VISIBILITY_BINS = (
    VisibilityBin("extreme", 0.10, 0.25),
    VisibilityBin("severe", 0.25, 0.50),
    VisibilityBin("mild", 0.50, 0.75),
    VisibilityBin("clear", 0.75, 1.00),
)


@dataclass(frozen=True)
class CompositionSpec:
    canvas_size: tuple[int, int] = (640, 640)
    object_range: tuple[int, int] = (12, 30)
    # Fallback only. With a SceneCalibration, scale is bootstrapped from real train.
    object_scale: tuple[float, float] = (0.08, 0.18)
    calibrated_scale_limits: tuple[float, float] = (0.02, 0.30)
    minimum_visibility: float = 0.10
    mode: str = "visibility"
    visibility_bins: tuple[VisibilityBin, ...] = DEFAULT_VISIBILITY_BINS
    target_bin_weights: dict[str, float] = field(
        default_factory=lambda: {
            "clear": 0.25,
            "mild": 0.25,
            "severe": 0.30,
            "extreme": 0.20,
        }
    )
    scene_mode_weights: dict[str, float] = field(
        default_factory=lambda: {"spread": 0.40, "cluster": 0.35, "pile": 0.25}
    )
    support_overlap_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "spread": (0.00, 0.05),
            "cluster": (0.00, 0.22),
            "pile": (0.08, 0.35),
        }
    )
    controlled_fraction: float = 0.25
    use_shadows: bool = True
    per_object_reflectance_jitter: float = 0.02
    class_balanced: bool = True
    max_position_attempts: int = 180

    def __post_init__(self) -> None:
        width, height = self.canvas_size
        if width <= 0 or height <= 0:
            raise ValueError("Ukuran canvas harus positif")
        if self.object_range[0] <= 0 or self.object_range[0] > self.object_range[1]:
            raise ValueError("object_range tidak valid")
        if not (0 < self.object_scale[0] <= self.object_scale[1] < 1):
            raise ValueError("object_scale harus berupa fraksi canvas antara 0 dan 1")
        if not (0 < self.calibrated_scale_limits[0] < self.calibrated_scale_limits[1] < 1):
            raise ValueError("calibrated_scale_limits tidak valid")
        if self.mode not in {"naive", "visibility"}:
            raise ValueError("mode harus naive atau visibility")
        if not (0.0 <= self.minimum_visibility < 1.0):
            raise ValueError("minimum_visibility harus di antara 0 dan 1")
        if not (0.0 < self.controlled_fraction <= 0.50):
            raise ValueError("controlled_fraction harus berada pada (0, 0.5]")
        known = {item.name for item in self.visibility_bins}
        if set(self.target_bin_weights) - known:
            raise ValueError("target_bin_weights memuat visibility bin yang tidak dikenal")
        if sum(self.target_bin_weights.values()) <= 0:
            raise ValueError("Bobot target visibility harus positif")
        if set(self.scene_mode_weights) != set(self.support_overlap_ranges):
            raise ValueError("Scene mode dan overlap range harus memiliki key yang sama")
        for name, (minimum, maximum) in self.support_overlap_ranges.items():
            if not 0 <= minimum <= maximum < 1:
                raise ValueError(f"Overlap range {name} tidak valid")


@dataclass(frozen=True)
class _SceneLighting:
    exposure: float
    contrast: float
    gamma: float
    red_gain: float
    blue_gain: float
    shadow_dx: int
    shadow_dy: int
    shadow_blur: float
    shadow_opacity: float
    contact_blur: float
    contact_opacity: float
    sensor_noise_std: float
    defocus_radius: float


def _bleed_transparent_rgb(image: Image.Image, iterations: int = 4) -> Image.Image:
    """Propagate edge colors into transparent padding to prevent white halos."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    rgb = rgba[:, :, :3].astype(np.float32)
    known = rgba[:, :, 3] >= 32
    height, width = known.shape
    for _ in range(iterations):
        accum = np.zeros_like(rgb)
        counts = np.zeros((height, width), dtype=np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            source_y = slice(max(0, -dy), min(height, height - dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_y = slice(max(0, dy), min(height, height + dy))
            target_x = slice(max(0, dx), min(width, width + dx))
            valid = known[source_y, source_x]
            accum[target_y, target_x] += rgb[source_y, source_x] * valid[..., None]
            counts[target_y, target_x] += valid
        frontier = (~known) & (counts > 0)
        if not np.any(frontier):
            break
        rgb[frontier] = accum[frontier] / counts[frontier, None]
        known |= frontier
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


@lru_cache(maxsize=512)
def _load_rgba_cached(path: str) -> Image.Image:
    with Image.open(path) as source:
        return _bleed_transparent_rgb(source.convert("RGBA"))


def load_cutout(cutout: Cutout) -> Image.Image:
    if not cutout.image_path.is_file():
        raise FileNotFoundError(f"Asset cutout tidak ditemukan: {cutout.image_path}")
    image = _load_rgba_cached(str(cutout.image_path))
    if not np.any(np.asarray(image.getchannel("A")) >= 32):
        raise ValueError(f"Asset tidak memiliki foreground: {cutout.image_path}")
    return image


def _resize_rotate_premultiplied(
    image: Image.Image,
    size: tuple[int, int],
    angle: float,
) -> Image.Image:
    array = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = array[:, :, 3:4]
    premultiplied = np.concatenate((array[:, :, :3] * alpha, alpha), axis=2)
    working = Image.fromarray(
        np.clip(premultiplied * 255.0, 0, 255).astype(np.uint8), mode="RGBA"
    )
    working = working.resize(size, Image.Resampling.LANCZOS)
    working = working.rotate(
        angle, resample=Image.Resampling.BICUBIC, expand=True
    )
    transformed = np.asarray(working, dtype=np.float32) / 255.0
    out_alpha = transformed[:, :, 3:4]
    rgb = np.divide(
        transformed[:, :, :3],
        np.maximum(out_alpha, 1.0 / 255.0),
        out=np.zeros_like(transformed[:, :, :3]),
        where=out_alpha > 0,
    )
    output = np.concatenate((np.clip(rgb, 0, 1), out_alpha), axis=2)
    return Image.fromarray((output * 255.0).astype(np.uint8), mode="RGBA")


@lru_cache(maxsize=4096)
def _cutout_principal_geometry(path: str) -> tuple[float, float, float]:
    """Cache mask geometry; it is invariant to RGB reflectance jitter."""
    image = _load_rgba_cached(path)
    mask = np.asarray(image.getchannel("A"), dtype=np.uint8) >= 32
    return principal_mask_geometry(mask)


def _cutout_intrinsic_aspect(cutout: Cutout) -> float:
    if cutout.intrinsic_aspect_ratio is not None:
        return max(float(cutout.intrinsic_aspect_ratio), 1.0)
    major, minor, _ = _cutout_principal_geometry(str(cutout.image_path))
    aspect = major / max(minor, 1e-6)
    cutout.intrinsic_aspect_ratio = aspect
    return aspect


def _projected_ellipse_size(
    major: float,
    minor: float,
    axis_angle: float,
) -> tuple[float, float]:
    """Approximate the axis-aligned box of an oriented oval bean."""
    cosine = math.cos(axis_angle)
    sine = math.sin(axis_angle)
    width = math.sqrt((major * cosine) ** 2 + (minor * sine) ** 2)
    height = math.sqrt((major * sine) ** 2 + (minor * cosine) ** 2)
    return max(width, 1.0), max(height, 1.0)


def _rotation_and_projected_size(
    mask: np.ndarray,
    rng: random.Random,
    target_width_height_ratio: float | None,
    geometry: tuple[float, float, float] | None = None,
) -> tuple[float, tuple[float, float]]:
    """Plan a rotation, optionally matching a real bbox aspect/orientation.

    The calibration target is a signed width/height ratio.  Searching the
    projected principal ellipse is cheap and avoids rendering dozens of angle
    candidates.  A random 180-degree flip preserves appearance diversity
    without changing the axis-aligned target box.
    """
    major, minor, source_axis = geometry or principal_mask_geometry(mask)
    if target_width_height_ratio is None:
        rotation = rng.uniform(0.0, 360.0)
        # PIL's positive angle is counter-clockwise on screen while mask
        # coordinates use y-down.  In mask coordinates the resulting axis is
        # therefore source_axis - rotation.
        target_axis = source_axis - math.radians(rotation)
    else:
        target_ratio = max(float(target_width_height_ratio), 1e-6)
        scored: list[tuple[float, float]] = []
        for degrees in range(180):
            axis = math.radians(float(degrees))
            width, height = _projected_ellipse_size(major, minor, axis)
            score = abs(math.log((width / height) / target_ratio))
            scored.append((score, axis))
        best_score = min(score for score, _ in scored)
        # Symmetric angles are equally valid.  Keeping all near-optimal
        # candidates prevents a deterministic directional artifact.
        candidates = [
            axis for score, axis in scored if score <= best_score + 0.003
        ]
        target_axis = rng.choice(candidates) + math.radians(
            rng.uniform(-0.5, 0.5)
        )
        rotation = math.degrees(source_axis - target_axis)
        if rng.random() < 0.5:
            rotation += 180.0
    projected_size = _projected_ellipse_size(major, minor, target_axis)
    return rotation, projected_size


def _transform_cutout(
    cutout: Cutout,
    spec: CompositionSpec,
    rng: random.Random,
    *,
    scene_scale: float | None = None,
    calibration: SceneCalibration | None = None,
    target_width_height_ratio: float | None = None,
) -> TransformedCutout:
    image = load_cutout(cutout)
    alpha = image.getchannel("A")
    reflectance = rng.uniform(
        1.0 - spec.per_object_reflectance_jitter,
        1.0 + spec.per_object_reflectance_jitter,
    )
    rgb = ImageEnhance.Brightness(image.convert("RGB")).enhance(reflectance)
    rgb.putalpha(alpha)
    image = rgb
    canvas_long = max(spec.canvas_size)
    if calibration is not None and scene_scale is not None:
        ratio = rng.choice(calibration.within_scene_scale_ratios)
        fraction = scene_scale * ratio
        fraction = min(
            max(fraction, spec.calibrated_scale_limits[0]),
            spec.calibrated_scale_limits[1],
        )
    else:
        fraction = rng.uniform(spec.object_scale[0], spec.object_scale[1])
    target_long_side = max(6, int(round(canvas_long * fraction)))
    source_mask = np.asarray(image.getchannel("A"), dtype=np.uint8) >= 32
    target_ratio = target_width_height_ratio
    if target_ratio is None and calibration is not None:
        target_ratio = float(rng.choice(calibration.bbox_width_height_ratios))
    angle, projected_size = _rotation_and_projected_size(
        source_mask,
        rng,
        target_ratio,
        geometry=_cutout_principal_geometry(str(cutout.image_path)),
    )
    # Calibrate the final visible mask, not the transparent RGBA crop.  This
    # keeps long-side scale correct even when a bean is rotated diagonally.
    scale = target_long_side / max(projected_size)
    size = (
        max(3, int(round(image.width * scale))),
        max(3, int(round(image.height * scale))),
    )
    source_image = image
    image = _resize_rotate_premultiplied(source_image, size, angle)
    mask = np.asarray(image.getchannel("A"), dtype=np.uint8) >= 32
    transformed_box = mask_bbox(mask)
    if transformed_box is not None:
        actual_long_side = max(transformed_box[2], transformed_box[3])
        if actual_long_side > 0 and abs(actual_long_side - target_long_side) > 1:
            correction = target_long_side / actual_long_side
            corrected_size = (
                max(3, int(round(size[0] * correction))),
                max(3, int(round(size[1] * correction))),
            )
            image = _resize_rotate_premultiplied(
                source_image, corrected_size, angle
            )
            mask = np.asarray(image.getchannel("A"), dtype=np.uint8) >= 32
    if not np.any(mask):
        raise ValueError(f"Transformasi menghasilkan mask kosong: {cutout.asset_id}")
    achieved_box = mask_bbox(mask)
    achieved_ratio = (
        achieved_box[2] / achieved_box[3]
        if achieved_box is not None and achieved_box[3] > 0
        else None
    )
    return TransformedCutout(
        cutout=cutout,
        rgba=image,
        mask=mask,
        target_bbox_ratio=target_ratio,
        achieved_bbox_ratio=achieved_ratio,
    )


def _global_mask(
    local_mask: np.ndarray,
    x: int,
    y: int,
    canvas_size: tuple[int, int],
) -> np.ndarray:
    width, height = canvas_size
    result = np.zeros((height, width), dtype=bool)
    local_height, local_width = local_mask.shape
    result[y : y + local_height, x : x + local_width] = local_mask
    return result


def _random_position(
    transformed: TransformedCutout,
    canvas_size: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    width, height = canvas_size
    return (
        rng.randint(0, max(0, width - transformed.rgba.width)),
        rng.randint(0, max(0, height - transformed.rgba.height)),
    )


def _scene_centers(
    mode: str,
    canvas_size: tuple[int, int],
    rng: random.Random,
) -> tuple[tuple[float, float], ...]:
    if mode == "spread":
        return ()
    width, height = canvas_size
    count = rng.randint(2, 4) if mode == "cluster" else rng.randint(1, 2)
    margin_x, margin_y = width * 0.12, height * 0.12
    return tuple(
        (
            rng.uniform(margin_x, width - margin_x),
            rng.uniform(margin_y, height - margin_y),
        )
        for _ in range(count)
    )


def _candidate_position(
    item: TransformedCutout,
    canvas_size: tuple[int, int],
    mode: str,
    centers: tuple[tuple[float, float], ...],
    rng: random.Random,
) -> tuple[int, int]:
    if mode == "spread" or not centers:
        return _random_position(item, canvas_size, rng)
    width, height = canvas_size
    center_x, center_y = rng.choice(centers)
    sigma = (0.13 if mode == "cluster" else 0.075) * max(canvas_size)
    x = round(rng.gauss(center_x, sigma) - item.rgba.width / 2)
    y = round(rng.gauss(center_y, sigma) - item.rgba.height / 2)
    return (
        int(min(max(0, x), width - item.rgba.width)),
        int(min(max(0, y), height - item.rgba.height)),
    )


def _place_contact_constrained(
    item: TransformedCutout,
    occupied: np.ndarray,
    spec: CompositionSpec,
    mode: str,
    centers: tuple[tuple[float, float], ...],
    rng: random.Random,
    *,
    overlap_range: tuple[float, float] | None = None,
) -> tuple[int, int, float]:
    minimum, maximum = overlap_range or spec.support_overlap_ranges[mode]
    target = (minimum + maximum) / 2.0
    best: tuple[int, int, float] | None = None
    best_distance = float("inf")
    area = max(int(item.mask.sum()), 1)
    for _ in range(spec.max_position_attempts):
        x, y = _candidate_position(item, spec.canvas_size, mode, centers, rng)
        height, width = item.mask.shape
        overlap = float(np.logical_and(occupied[y : y + height, x : x + width], item.mask).sum()) / area
        distance = abs(overlap - target)
        if distance < best_distance:
            best = x, y, overlap
            best_distance = distance
        if minimum <= overlap <= maximum:
            return x, y, overlap
    if best is None:
        raise RuntimeError("Tidak dapat menempatkan objek")
    return best


def _select_target_bin(spec: CompositionSpec, rng: random.Random) -> VisibilityBin:
    bins = {item.name: item for item in spec.visibility_bins}
    names = list(spec.target_bin_weights)
    weights = [spec.target_bin_weights[name] for name in names]
    return bins[rng.choices(names, weights=weights, k=1)[0]]


def _find_occluder_position(
    transformed: TransformedCutout,
    focus_mask: np.ndarray,
    already_occluded: np.ndarray,
    desired_visibility: float,
    spec: CompositionSpec,
    rng: random.Random,
) -> tuple[int, int, float]:
    focus_box = mask_bbox(focus_mask)
    if focus_box is None:
        raise ValueError("Focus mask kosong")
    fx, fy, fw, fh = focus_box
    focus_area = max(int(focus_mask.sum()), 1)
    best: tuple[int, int, float] | None = None
    best_distance = float("inf")
    width, height = spec.canvas_size
    for _ in range(spec.max_position_attempts):
        center_x = rng.randint(fx, fx + max(fw - 1, 0))
        center_y = rng.randint(fy, fy + max(fh - 1, 0))
        x = min(
            max(0, center_x - rng.randint(0, max(transformed.rgba.width - 1, 0))),
            width - transformed.rgba.width,
        )
        y = min(
            max(0, center_y - rng.randint(0, max(transformed.rgba.height - 1, 0))),
            height - transformed.rgba.height,
        )
        local_height, local_width = transformed.mask.shape
        focus_region = focus_mask[y : y + local_height, x : x + local_width]
        previous_region = already_occluded[y : y + local_height, x : x + local_width]
        new_overlap = np.logical_and(focus_region, transformed.mask)
        new_pixels = np.logical_and(new_overlap, ~previous_region).sum()
        total_occluded = int(already_occluded.sum()) + int(new_pixels)
        visible = 1.0 - float(total_occluded) / focus_area
        distance = abs(visible - desired_visibility)
        if distance < best_distance:
            best = int(x), int(y), visible
            best_distance = distance
        if distance <= 0.02:
            return int(x), int(y), visible
    if best is None:
        raise RuntimeError("Tidak dapat menempatkan occluder")
    return best


def _visibility_bin(value: float, spec: CompositionSpec) -> str:
    if value < spec.minimum_visibility:
        return "ignored"
    for index, item in enumerate(spec.visibility_bins):
        if item.contains(value, is_last=index == len(spec.visibility_bins) - 1):
            return item.name
    return "ignored"


def _alpha_composite_clipped(
    canvas: Image.Image,
    overlay: Image.Image,
    x: int,
    y: int,
) -> None:
    left, top = max(0, x), max(0, y)
    right, bottom = min(canvas.width, x + overlay.width), min(canvas.height, y + overlay.height)
    if left >= right or top >= bottom:
        return
    crop = overlay.crop((left - x, top - y, right - x, bottom - y))
    canvas.alpha_composite(crop, (left, top))


def _shadow_layer(alpha: Image.Image, blur: float, opacity: float) -> Image.Image:
    filtered = alpha.filter(ImageFilter.GaussianBlur(radius=max(0.1, blur)))
    layer = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
    layer.putalpha(filtered.point(lambda value: int(value * opacity)))
    return layer


def _apply_camera_pipeline(
    image: Image.Image,
    lighting: _SceneLighting,
    rng: random.Random,
) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gains = np.asarray(
        [lighting.red_gain, 1.0, lighting.blue_gain], dtype=np.float32
    ).reshape(1, 1, 3)
    array = np.clip(array * gains * lighting.exposure, 0.0, 1.0)
    array = np.clip((array - 0.5) * lighting.contrast + 0.5, 0.0, 1.0)
    array = np.power(array, lighting.gamma)
    numpy_rng = np.random.default_rng(rng.getrandbits(64))
    noise = numpy_rng.normal(
        0.0,
        lighting.sensor_noise_std / 255.0,
        size=array.shape[:2] + (1,),
    )
    array = np.clip(array + noise, 0.0, 1.0)
    result = Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")
    if lighting.defocus_radius > 0.05:
        result = result.filter(ImageFilter.GaussianBlur(lighting.defocus_radius))
    return result


def _render_scene(
    background: Image.Image,
    transformed: list[tuple[TransformedCutout, int, int, bool]],
    spec: CompositionSpec,
    lighting: _SceneLighting,
    camera_rng: random.Random,
) -> tuple[Image.Image, list[PlacedInstance]]:
    canvas = background.convert("RGBA")
    if canvas.size != spec.canvas_size:
        canvas = canvas.resize(spec.canvas_size, Image.Resampling.LANCZOS)
    instances: list[PlacedInstance] = []
    for index, (item, x, y, is_focus) in enumerate(transformed):
        if spec.use_shadows:
            alpha = item.rgba.getchannel("A")
            cast = _shadow_layer(
                alpha, lighting.shadow_blur, lighting.shadow_opacity
            )
            contact = _shadow_layer(
                alpha, lighting.contact_blur, lighting.contact_opacity
            )
            _alpha_composite_clipped(
                canvas, cast, x + lighting.shadow_dx, y + lighting.shadow_dy
            )
            _alpha_composite_clipped(canvas, contact, x, y + 1)
        _alpha_composite_clipped(canvas, item.rgba, x, y)
        instances.append(
            PlacedInstance(
                instance_id=index + 1,
                cutout=item.cutout,
                x=x,
                y=y,
                z_order=index,
                full_mask=_global_mask(item.mask, x, y, spec.canvas_size),
                is_focus=is_focus,
                target_bbox_ratio=item.target_bbox_ratio,
                achieved_bbox_ratio=item.achieved_bbox_ratio,
            )
        )

    occlusion = np.zeros((spec.canvas_size[1], spec.canvas_size[0]), dtype=bool)
    for instance in reversed(instances):
        instance.visible_mask = instance.full_mask & ~occlusion
        full_area = int(instance.full_mask.sum())
        instance.visibility_ratio = float(instance.visible_mask.sum()) / max(full_area, 1)
        instance.visibility_bin = _visibility_bin(instance.visibility_ratio, spec)
        occlusion |= instance.full_mask
    return _apply_camera_pipeline(canvas.convert("RGB"), lighting, camera_rng), instances


def _choose_unique_cutouts(
    cutouts: list[Cutout],
    count: int,
    spec: CompositionSpec,
    rng: random.Random,
    *,
    calibration: SceneCalibration | None = None,
    geometry_rng: random.Random | None = None,
) -> tuple[list[Cutout], list[float | None], int, int]:
    """Choose class-balanced assets that can realize sampled bbox geometry.

    Requested ratios are sampled per class from real train.  Hardest ratios are
    allocated first so elongated assets are not consumed by easy targets.  No
    anisotropic warping is used; a fallback is recorded when no remaining real
    cutout can physically realize the requested aspect.
    """
    by_class: dict[int, list[Cutout]] = {}
    for cutout in cutouts:
        by_class.setdefault(cutout.class_id, []).append(cutout)
    for values in by_class.values():
        rng.shuffle(values)
    class_counts = {class_id: 0 for class_id in by_class}
    class_slots: list[int] = []
    for _ in range(count):
        class_ids = list(by_class)
        if spec.class_balanced:
            minimum = min(class_counts.values())
            class_ids = [key for key in class_ids if class_counts[key] == minimum]
        class_id = rng.choice(class_ids)
        class_slots.append(class_id)
        class_counts[class_id] += 1

    ratio_rng = geometry_rng or rng
    target_ratios: list[float | None] = []
    for class_id in class_slots:
        if calibration is None:
            target_ratios.append(None)
            continue
        distribution = (
            calibration.bbox_width_height_ratios_by_class.get(class_id)
            or calibration.bbox_width_height_ratios
        )
        target_ratios.append(float(ratio_rng.choice(distribution)))

    chosen: list[Cutout | None] = [None] * count
    used_assets: set[str] = set()
    used_parents: set[str] = set()
    repeated = 0
    geometry_fallbacks = 0
    allocation_order = sorted(
        range(count),
        key=lambda index: (
            max(
                float(target_ratios[index]),
                1.0 / float(target_ratios[index]),
            )
            if target_ratios[index] is not None
            else 1.0
        ),
        reverse=True,
    )
    for index in allocation_order:
        class_id = class_slots[index]
        candidates = [
            item
            for item in by_class[class_id]
            if item.asset_id not in used_assets
        ]
        if not candidates:
            candidates = list(by_class[class_id])
            repeated += 1
        target_ratio = target_ratios[index]
        if target_ratio is not None:
            target_aspect = max(target_ratio, 1.0 / target_ratio)
            capable = [
                item
                for item in candidates
                if _cutout_intrinsic_aspect(item) >= target_aspect * 0.97
            ]
            if capable:
                gaps = {
                    item.asset_id: _cutout_intrinsic_aspect(item) - target_aspect
                    for item in capable
                }
                best_gap = min(gaps.values())
                candidates = [
                    item
                    for item in capable
                    if gaps[item.asset_id] <= best_gap + 0.05
                ]
            else:
                best = max(_cutout_intrinsic_aspect(item) for item in candidates)
                candidates = [
                    item
                    for item in candidates
                    if _cutout_intrinsic_aspect(item) >= best * 0.995
                ]
                geometry_fallbacks += 1
        unused_parent_candidates = [
            item
            for item in candidates
            if item.source_parent_id is None
            or item.source_parent_id not in used_parents
        ]
        if unused_parent_candidates:
            candidates = unused_parent_candidates
        candidate = rng.choice(candidates)
        chosen[index] = candidate
        used_assets.add(candidate.asset_id)
        if candidate.source_parent_id is not None:
            used_parents.add(candidate.source_parent_id)
    selected = [item for item in chosen if item is not None]
    if len(selected) != count:
        raise RuntimeError("Pemilihan geometry-aware menghasilkan slot kosong")
    return selected, target_ratios, repeated, geometry_fallbacks


def _sample_lighting(
    calibration: SceneCalibration | None,
    median_object_pixels: float,
    rng: random.Random,
) -> _SceneLighting:
    angle = rng.uniform(0.0, math.tau)
    shadow_length = rng.uniform(0.015, 0.04) * median_object_pixels
    sensor_std = (
        rng.choice(calibration.background_sensor_std)
        if calibration is not None and calibration.background_sensor_std
        else rng.uniform(0.8, 1.8)
    )
    return _SceneLighting(
        exposure=rng.uniform(0.98, 1.02),
        contrast=rng.uniform(0.985, 1.015),
        gamma=rng.uniform(0.985, 1.015),
        red_gain=rng.uniform(0.985, 1.015),
        blue_gain=rng.uniform(0.985, 1.015),
        shadow_dx=int(round(math.cos(angle) * shadow_length)),
        shadow_dy=int(round(math.sin(angle) * shadow_length)),
        shadow_blur=max(0.6, median_object_pixels * rng.uniform(0.025, 0.055)),
        shadow_opacity=rng.uniform(0.08, 0.16),
        contact_blur=max(0.4, median_object_pixels * rng.uniform(0.008, 0.018)),
        contact_opacity=rng.uniform(0.14, 0.24),
        sensor_noise_std=min(max(float(sensor_std), 0.4), 2.5),
        defocus_radius=rng.uniform(0.0, 0.25),
    )


def compose_scene(
    background: Image.Image,
    cutouts: list[Cutout],
    spec: CompositionSpec,
    rng: random.Random,
    calibration: SceneCalibration | None = None,
) -> SyntheticScene:
    """Create a deterministic physics-informed 2.5D projected scene.

    Named RNG streams and pre-transformed assets keep A1/A2 paired. Support
    objects use contact-constrained projected packing. A2 then creates several
    explicit focus/occluder layers in the requested visibility band.
    """
    if not cutouts:
        raise ValueError("Pustaka cutout kosong")
    streams = [random.Random(rng.getrandbits(64)) for _ in range(7)]
    (
        selection_rng,
        geometry_plan_rng,
        transform_rng,
        placement_rng,
        lighting_rng,
        camera_rng,
        mode_rng,
    ) = streams
    if calibration is not None:
        if calibration.scene_count_scale_pairs:
            sampled_count, scene_scale = selection_rng.choice(
                calibration.scene_count_scale_pairs
            )
            sampled_count = int(sampled_count)
            scene_scale = float(scene_scale)
        else:
            sampled_count = int(selection_rng.choice(calibration.scene_counts))
            scene_scale = float(
                geometry_plan_rng.choice(calibration.scene_scale_medians)
            )
        count = min(max(sampled_count, spec.object_range[0]), spec.object_range[1])
    else:
        count = selection_rng.randint(spec.object_range[0], spec.object_range[1])
        scene_scale = None
    chosen, target_ratios, repeated_assets, geometry_fallbacks = _choose_unique_cutouts(
        cutouts,
        count,
        spec,
        selection_rng,
        calibration=calibration,
        geometry_rng=geometry_plan_rng,
    )
    transformed_items = [
        _transform_cutout(
            item,
            spec,
            transform_rng,
            scene_scale=scene_scale,
            calibration=calibration,
            target_width_height_ratio=target_ratio,
        )
        for item, target_ratio in zip(chosen, target_ratios)
    ]
    geometry_targets = sum(item.target_bbox_ratio is not None for item in transformed_items)
    geometry_hits = sum(
        item.target_bbox_ratio is not None
        and item.achieved_bbox_ratio is not None
        and abs(
            math.log(item.achieved_bbox_ratio / item.target_bbox_ratio)
        )
        <= math.log(1.10)
        for item in transformed_items
    )
    median_pixels = float(np.median([max(item.rgba.size) for item in transformed_items]))
    lighting = _sample_lighting(calibration, median_pixels, lighting_rng)
    mode_names = list(spec.scene_mode_weights)
    scene_mode = mode_rng.choices(
        mode_names,
        weights=[spec.scene_mode_weights[name] for name in mode_names],
        k=1,
    )[0]
    centers = _scene_centers(scene_mode, spec.canvas_size, mode_rng)
    occupied = np.zeros((spec.canvas_size[1], spec.canvas_size[0]), dtype=bool)
    placed: list[tuple[TransformedCutout, int, int, bool]] = []
    target: VisibilityBin | None = None
    controlled_count = 0

    def place_support(item: TransformedCutout) -> None:
        x, y, _ = _place_contact_constrained(
            item, occupied, spec, scene_mode, centers, placement_rng
        )
        placed.append((item, x, y, False))
        occupied[:] |= _global_mask(item.mask, x, y, spec.canvas_size)

    if spec.mode == "naive" or count == 1:
        for item in transformed_items:
            place_support(item)
    else:
        target = _select_target_bin(spec, placement_rng)
        occluders_per_focus = {
            "clear": 0,
            "mild": 1,
            "severe": 2,
            "extreme": 2,
        }[target.name]
        occluders_per_focus = min(occluders_per_focus, max(0, count - 1))
        desired_focuses = max(2, int(math.ceil(count * spec.controlled_fraction)))
        if occluders_per_focus:
            maximum_focuses = max(1, count // (1 + occluders_per_focus))
            controlled_count = min(desired_focuses, maximum_focuses)
        else:
            controlled_count = min(desired_focuses, count)
        support_count = count - controlled_count * (1 + occluders_per_focus)
        cursor = 0
        for item in transformed_items[:support_count]:
            place_support(item)
            cursor += 1

        target_mid = (target.minimum + target.maximum) / 2.0
        for _ in range(controlled_count):
            focus = transformed_items[cursor]
            cursor += 1
            focus_x, focus_y, _ = _place_contact_constrained(
                focus,
                occupied,
                spec,
                scene_mode,
                centers,
                placement_rng,
                overlap_range=(0.0, 0.04),
            )
            focus_mask = _global_mask(
                focus.mask, focus_x, focus_y, spec.canvas_size
            )
            placed.append((focus, focus_x, focus_y, True))
            occupied[:] |= focus_mask
            already_occluded = np.zeros_like(occupied)
            current_visibility = 1.0
            for occluder_index in range(occluders_per_focus):
                occluder = transformed_items[cursor]
                cursor += 1
                remaining = occluders_per_focus - occluder_index
                desired = target_mid + (current_visibility - target_mid) * (
                    (remaining - 1) / remaining
                )
                x, y, current_visibility = _find_occluder_position(
                    occluder,
                    focus_mask,
                    already_occluded,
                    desired,
                    spec,
                    placement_rng,
                )
                occluder_mask = _global_mask(
                    occluder.mask, x, y, spec.canvas_size
                )
                already_occluded |= occluder_mask & focus_mask
                occupied |= occluder_mask
                placed.append((occluder, x, y, False))

    image, instances = _render_scene(
        background, placed, spec, lighting, camera_rng
    )
    focus_instances = [item for item in instances if item.is_focus]
    controlled_hits = 0
    if target is not None:
        controlled_hits = sum(
            target.contains(
                item.visibility_ratio,
                is_last=target.maximum == 1.0,
            )
            for item in focus_instances
        )
    return SyntheticScene(
        image=image,
        instances=instances,
        target_visibility_bin=target.name if target else None,
        target_visibility_hit=(
            controlled_hits == len(focus_instances) if focus_instances else False
        ),
        scene_mode=scene_mode,
        controlled_instances=len(focus_instances),
        controlled_hits=controlled_hits,
        repeated_assets=repeated_assets,
        geometry_targets=geometry_targets,
        geometry_hits=geometry_hits,
        geometry_fallbacks=geometry_fallbacks,
    )


def load_background(
    path: Path | None,
    size: tuple[int, int],
    rng: random.Random,
    calibration: SceneCalibration | None = None,
) -> Image.Image:
    if path is None:
        width, height = size
        if calibration is not None and calibration.background_colors:
            base_color = np.asarray(
                rng.choice(calibration.background_colors), dtype=np.float32
            )
            gradient_std = rng.choice(calibration.background_gradient_std)
        else:
            base = rng.randint(210, 240)
            base_color = np.asarray(
                [base + rng.randint(-3, 3) for _ in range(3)], dtype=np.float32
            )
            gradient_std = rng.uniform(2.0, 6.0)
        gradient_std = min(max(float(gradient_std), 1.0), 10.0)
        yy, xx = np.mgrid[-0.5:0.5:complex(height), -0.5:0.5:complex(width)]
        angle = rng.uniform(0.0, math.tau)
        plane = xx * math.cos(angle) + yy * math.sin(angle)
        plane = plane / max(float(plane.std()), 1e-6) * gradient_std
        numpy_rng = np.random.default_rng(rng.getrandbits(64))
        coarse = numpy_rng.normal(0.0, gradient_std * 0.30, size=(16, 16))
        texture = np.asarray(
            Image.fromarray(coarse.astype(np.float32), mode="F").resize(
                (width, height), Image.Resampling.BICUBIC
            ),
            dtype=np.float32,
        )
        radial = xx**2 + yy**2
        vignette = radial * rng.uniform(0.0, 3.0)
        pixels = base_color.reshape(1, 1, 3) + plane[..., None] + texture[..., None]
        pixels -= vignette[..., None]
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
    image = Image.open(path).convert("RGB")
    target_width, target_height = size
    scale = max(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (
            max(target_width, int(round(image.width * scale))),
            max(target_height, int(round(image.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    max_x, max_y = resized.width - target_width, resized.height - target_height
    left = rng.randint(0, max_x) if max_x else 0
    top = rng.randint(0, max_y) if max_y else 0
    return resized.crop((left, top, left + target_width, top + target_height))
