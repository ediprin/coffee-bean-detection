from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .masks import mask_bbox
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
    object_scale: tuple[float, float] = (0.08, 0.18)
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
    use_shadows: bool = True
    color_jitter: float = 0.08
    class_balanced: bool = True
    max_position_attempts: int = 400

    def __post_init__(self) -> None:
        width, height = self.canvas_size
        if width <= 0 or height <= 0:
            raise ValueError("Ukuran canvas harus positif")
        if self.object_range[0] <= 0 or self.object_range[0] > self.object_range[1]:
            raise ValueError("object_range tidak valid")
        if not (0 < self.object_scale[0] <= self.object_scale[1] < 1):
            raise ValueError("object_scale harus berupa fraksi canvas antara 0 dan 1")
        if self.mode not in {"naive", "visibility"}:
            raise ValueError("mode harus naive atau visibility")
        if not (0.0 <= self.minimum_visibility < 1.0):
            raise ValueError("minimum_visibility harus di antara 0 dan 1")
        known = {item.name for item in self.visibility_bins}
        if set(self.target_bin_weights) - known:
            raise ValueError("target_bin_weights memuat visibility bin yang tidak dikenal")
        if sum(self.target_bin_weights.values()) <= 0:
            raise ValueError("Bobot target visibility harus positif")


def load_cutout(cutout: Cutout) -> Image.Image:
    if not cutout.image_path.is_file():
        raise FileNotFoundError(f"Asset cutout tidak ditemukan: {cutout.image_path}")
    image = Image.open(cutout.image_path).convert("RGBA")
    if not np.any(np.asarray(image.getchannel("A")) >= 32):
        raise ValueError(f"Asset tidak memiliki foreground: {cutout.image_path}")
    return image


def _transform_cutout(
    cutout: Cutout,
    spec: CompositionSpec,
    rng: random.Random,
    target_long_side: int | None = None,
) -> TransformedCutout:
    image = load_cutout(cutout)
    canvas_long = max(spec.canvas_size)
    if target_long_side is None:
        target_long_side = int(
            round(canvas_long * rng.uniform(spec.object_scale[0], spec.object_scale[1]))
        )
    scale = target_long_side / max(image.size)
    size = (
        max(3, int(round(image.width * scale))),
        max(3, int(round(image.height * scale))),
    )
    image = image.resize(size, Image.Resampling.LANCZOS)
    if spec.color_jitter > 0:
        low, high = 1.0 - spec.color_jitter, 1.0 + spec.color_jitter
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(low, high))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(low, high))
        image = ImageEnhance.Color(image).enhance(rng.uniform(low, high))
    image = image.rotate(
        rng.uniform(0.0, 360.0),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    mask = np.asarray(image.getchannel("A"), dtype=np.uint8) >= 32
    if not np.any(mask):
        raise ValueError(f"Transformasi menghasilkan mask kosong: {cutout.asset_id}")
    return TransformedCutout(cutout=cutout, rgba=image, mask=mask)


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


def _focus_position(
    transformed: TransformedCutout,
    canvas_size: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    width, height = canvas_size
    center_x = width // 2 + rng.randint(-width // 8, width // 8)
    center_y = height // 2 + rng.randint(-height // 8, height // 8)
    x = min(max(0, center_x - transformed.rgba.width // 2), width - transformed.rgba.width)
    y = min(max(0, center_y - transformed.rgba.height // 2), height - transformed.rgba.height)
    return int(x), int(y)


def _select_target_bin(spec: CompositionSpec, rng: random.Random) -> VisibilityBin:
    bins = {item.name: item for item in spec.visibility_bins}
    names = list(spec.target_bin_weights)
    weights = [spec.target_bin_weights[name] for name in names]
    return bins[rng.choices(names, weights=weights, k=1)[0]]


def _find_occluder_position(
    transformed: TransformedCutout,
    focus_mask: np.ndarray,
    target: VisibilityBin,
    spec: CompositionSpec,
    rng: random.Random,
) -> tuple[int, int, float, bool]:
    focus_box = mask_bbox(focus_mask)
    if focus_box is None:
        raise ValueError("Focus mask kosong")
    fx, fy, fw, fh = focus_box
    focus_area = int(focus_mask.sum())
    target_mid = (target.minimum + target.maximum) / 2.0
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
        focus_region = focus_mask[
            int(y) : int(y) + local_height,
            int(x) : int(x) + local_width,
        ]
        intersection = np.logical_and(focus_region, transformed.mask).sum()
        visible = 1.0 - float(intersection) / max(focus_area, 1)
        distance = abs(visible - target_mid)
        if distance < best_distance:
            best = int(x), int(y), visible
            best_distance = distance
        if target.contains(visible, is_last=target.maximum == 1.0):
            return int(x), int(y), visible, True
    if best is None:
        raise RuntimeError("Tidak dapat menempatkan occluder")
    return best[0], best[1], best[2], False


def _visibility_bin(value: float, spec: CompositionSpec) -> str:
    if value < spec.minimum_visibility:
        return "ignored"
    for index, item in enumerate(spec.visibility_bins):
        if item.contains(value, is_last=index == len(spec.visibility_bins) - 1):
            return item.name
    return "ignored"


def _render_scene(
    background: Image.Image,
    transformed: list[tuple[TransformedCutout, int, int, bool]],
    spec: CompositionSpec,
) -> tuple[Image.Image, list[PlacedInstance]]:
    canvas = background.convert("RGBA")
    if canvas.size != spec.canvas_size:
        canvas = canvas.resize(spec.canvas_size, Image.Resampling.LANCZOS)
    instances: list[PlacedInstance] = []
    for index, (item, x, y, is_focus) in enumerate(transformed):
        if spec.use_shadows:
            alpha = item.rgba.getchannel("A").filter(ImageFilter.GaussianBlur(radius=2.0))
            shadow = Image.new("RGBA", item.rgba.size, (0, 0, 0, 0))
            shadow.putalpha(alpha.point(lambda value: int(value * 0.18)))
            canvas.alpha_composite(shadow, (min(x + 2, canvas.width - shadow.width), min(y + 3, canvas.height - shadow.height)))
        canvas.alpha_composite(item.rgba, (x, y))
        instances.append(
            PlacedInstance(
                instance_id=index + 1,
                cutout=item.cutout,
                x=x,
                y=y,
                z_order=index,
                full_mask=_global_mask(item.mask, x, y, spec.canvas_size),
                is_focus=is_focus,
            )
        )

    occlusion = np.zeros((spec.canvas_size[1], spec.canvas_size[0]), dtype=bool)
    for instance in reversed(instances):
        instance.visible_mask = instance.full_mask & ~occlusion
        full_area = int(instance.full_mask.sum())
        instance.visibility_ratio = float(instance.visible_mask.sum()) / max(full_area, 1)
        instance.visibility_bin = _visibility_bin(instance.visibility_ratio, spec)
        occlusion |= instance.full_mask
    return canvas.convert("RGB"), instances


def compose_scene(
    background: Image.Image,
    cutouts: list[Cutout],
    spec: CompositionSpec,
    rng: random.Random,
) -> SyntheticScene:
    """Compose one deterministic scene using the supplied ``rng``.

    Visibility-aware mode guarantees one focus instance and attempts to place a
    single foreground occluder in the requested visibility band.  Other support
    objects are placed behind the focus, so they cannot silently change its
    final visibility ratio.
    """
    if not cutouts:
        raise ValueError("Pustaka cutout kosong")
    count = rng.randint(spec.object_range[0], spec.object_range[1])
    if spec.class_balanced:
        by_class: dict[int, list[Cutout]] = {}
        for cutout in cutouts:
            by_class.setdefault(cutout.class_id, []).append(cutout)
        class_ids = sorted(by_class)
        chosen = [rng.choice(by_class[rng.choice(class_ids)]) for _ in range(count)]
    else:
        chosen = [rng.choice(cutouts) for _ in range(count)]
    placed: list[tuple[TransformedCutout, int, int, bool]] = []
    target: VisibilityBin | None = None
    target_hit = False

    if spec.mode == "naive" or count == 1:
        for cutout in chosen:
            item = _transform_cutout(cutout, spec, rng)
            x, y = _random_position(item, spec.canvas_size, rng)
            placed.append((item, x, y, False))
    else:
        target = _select_target_bin(spec, rng)
        needs_occluder = target.name != "clear"
        support_count = max(0, count - 1 - int(needs_occluder))
        for cutout in chosen[:support_count]:
            item = _transform_cutout(cutout, spec, rng)
            x, y = _random_position(item, spec.canvas_size, rng)
            placed.append((item, x, y, False))

        focus_cutout = chosen[support_count]
        focus = _transform_cutout(focus_cutout, spec, rng)
        focus_x, focus_y = _focus_position(focus, spec.canvas_size, rng)
        focus_mask = _global_mask(focus.mask, focus_x, focus_y, spec.canvas_size)
        placed.append((focus, focus_x, focus_y, True))

        if needs_occluder:
            occluder_cutout = chosen[support_count + 1]
            focus_long = max(focus.rgba.size)
            best_item: TransformedCutout | None = None
            best_position: tuple[int, int, float, bool] | None = None
            best_distance = float("inf")
            target_mid = (target.minimum + target.maximum) / 2.0
            for _ in range(8):
                target_size = max(8, int(round(focus_long * rng.uniform(0.80, 1.35))))
                candidate = _transform_cutout(
                    occluder_cutout, spec, rng, target_long_side=target_size
                )
                position = _find_occluder_position(
                    candidate, focus_mask, target, spec, rng
                )
                distance = abs(position[2] - target_mid)
                if distance < best_distance:
                    best_item, best_position = candidate, position
                    best_distance = distance
                if position[3]:
                    break
            assert best_item is not None and best_position is not None
            placed.append((best_item, best_position[0], best_position[1], False))
            target_hit = best_position[3]
        else:
            target_hit = True

    image, instances = _render_scene(background, placed, spec)
    focus_instance = next((item for item in instances if item.is_focus), None)
    if target is not None and focus_instance is not None:
        target_hit = target.contains(
            focus_instance.visibility_ratio,
            is_last=target.maximum == 1.0,
        )
    return SyntheticScene(
        image=image,
        instances=instances,
        target_visibility_bin=target.name if target else None,
        target_visibility_hit=target_hit,
    )


def load_background(path: Path | None, size: tuple[int, int], rng: random.Random) -> Image.Image:
    if path is None:
        base = rng.randint(205, 242)
        height, width = size[1], size[0]
        numpy_rng = np.random.default_rng(rng.getrandbits(64))
        noise = numpy_rng.integers(
            -5, 6, size=(height, width, 1), dtype=np.int16
        )
        color = np.asarray(
            [base + rng.randint(-3, 3) for _ in range(3)], dtype=np.int16
        ).reshape(1, 1, 3)
        pixels = np.clip(color + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(pixels, mode="RGB")
    image = Image.open(path).convert("RGB")
    target_width, target_height = size
    scale = max(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (max(target_width, int(round(image.width * scale))), max(target_height, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    max_x, max_y = resized.width - target_width, resized.height - target_height
    left = rng.randint(0, max_x) if max_x else 0
    top = rng.randint(0, max_y) if max_y else 0
    return resized.crop((left, top, left + target_width, top + target_height))
