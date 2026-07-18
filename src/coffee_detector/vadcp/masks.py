from __future__ import annotations

import math
from collections import deque

import numpy as np
from PIL import Image, ImageFilter


def largest_component(
    mask: np.ndarray,
    preferred_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Select an 8-connected component.

    Classification images use the largest component. YOLO crops can provide
    the annotated box centre so a neighbouring bean is not silently selected
    merely because it has a slightly larger segmented area.
    """
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for start_y, start_x in zip(*np.nonzero(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        queue = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        while queue:
            y, x = queue.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        components.append(component)
    result = np.zeros_like(mask, dtype=bool)
    if not components:
        return result
    if preferred_point is None:
        best = max(components, key=len)
    else:
        point_x, point_y = preferred_point
        best = min(
            components,
            key=lambda component: min(
                (x - point_x) ** 2 + (y - point_y) ** 2
                for y, x in component
            ),
        )
    if best:
        ys, xs = zip(*best)
        result[np.asarray(ys), np.asarray(xs)] = True
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that are not connected to the image border."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if not mask[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if not mask[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < height
                and 0 <= nx < width
                and not mask[ny, nx]
                and not exterior[ny, nx]
            ):
                exterior[ny, nx] = True
                queue.append((ny, nx))
    return mask | (~mask & ~exterior)


def estimate_foreground_mask(
    image: Image.Image,
    threshold: float = 24.0,
    border_fraction: float = 0.06,
    preferred_point: tuple[float, float] | None = None,
) -> np.ndarray:
    """Estimate a bean mask from a mostly uniform border background.

    This is deliberately conservative.  The resulting object-library audit
    records failed or implausible masks so they can be corrected manually.
    """
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    if np.any(alpha < 250):
        mask = alpha >= 32
    else:
        rgb = np.asarray(rgba.convert("RGB"), dtype=np.float32)
        height, width = rgb.shape[:2]
        border = max(1, int(round(min(height, width) * border_fraction)))
        pixels = np.concatenate(
            [
                rgb[:border].reshape(-1, 3),
                rgb[-border:].reshape(-1, 3),
                rgb[:, :border].reshape(-1, 3),
                rgb[:, -border:].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(pixels, axis=0)
        distance = np.sqrt(np.sum((rgb - background) ** 2, axis=2))
        mask = distance >= float(threshold)

    # A neutral 3x3 close suppresses isolated dust without expanding the
    # foreground into the bright crop background.  The previous 5x5 dilate
    # followed by a 3x3 erode had a net outward growth and retained a visible
    # white rim around some beans.
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(3))
    mask_image = mask_image.filter(ImageFilter.MinFilter(3))
    mask = np.asarray(mask_image, dtype=np.uint8) >= 128
    return fill_holes(largest_component(mask, preferred_point=preferred_point))


def crop_to_mask(
    image: Image.Image,
    mask: np.ndarray,
    padding: int = 2,
) -> tuple[Image.Image, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Mask objek kosong")
    left = max(0, int(xs.min()) - padding)
    top = max(0, int(ys.min()) - padding)
    right = min(mask.shape[1], int(xs.max()) + padding + 1)
    bottom = min(mask.shape[0], int(ys.max()) + padding + 1)
    cropped_image = image.convert("RGBA").crop((left, top, right, bottom))
    cropped_mask = mask[top:bottom, left:right]
    # Transparent pixels retain their original background RGB.  Resampling
    # straight-alpha RGBA can pull that hidden color into the silhouette and
    # create a white/black halo.  Propagate nearby foreground RGB into the
    # narrow transparent padding before writing alpha.
    rgba = np.asarray(cropped_image, dtype=np.uint8).copy()
    rgb = rgba[:, :, :3].astype(np.float32)
    hard_mask_image = Image.fromarray(cropped_mask.astype(np.uint8) * 255, mode="L")
    core = np.asarray(
        hard_mask_image.filter(ImageFilter.MinFilter(3)), dtype=np.uint8
    ) >= 128
    # Tiny masks can disappear under a 3x3 erosion; retain them unchanged.
    if not np.any(core):
        core = cropped_mask.copy()
    # Boundary pixels are deliberately not trusted as RGB sources.  Flooding
    # outward from the eroded bean core replaces any segmented white rim as
    # well as the hidden RGB in transparent padding.
    known = core.copy()
    crop_height, crop_width = known.shape
    for _ in range(max(2, padding + 2)):
        accum = np.zeros_like(rgb)
        counts = np.zeros((crop_height, crop_width), dtype=np.float32)
        for dy, dx in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            source_y = slice(max(0, -dy), min(crop_height, crop_height - dy))
            source_x = slice(max(0, -dx), min(crop_width, crop_width - dx))
            target_y = slice(max(0, dy), min(crop_height, crop_height + dy))
            target_x = slice(max(0, dx), min(crop_width, crop_width + dx))
            valid = known[source_y, source_x]
            accum[target_y, target_x] += rgb[source_y, source_x] * valid[..., None]
            counts[target_y, target_x] += valid
        frontier = (~known) & (counts > 0)
        if not np.any(frontier):
            break
        rgb[frontier] = accum[frontier] / counts[frontier, None]
        known |= frontier
    rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    cropped_image = Image.fromarray(rgba, mode="RGBA")
    # Feather inward only: antialias the last foreground pixel but never add
    # alpha outside the binary annotation mask.  Thus visual blending improves
    # while the mask used to derive boxes/visibility stays unchanged.
    hard_alpha = cropped_mask.astype(np.uint8) * 255
    softened = np.asarray(
        Image.fromarray(core.astype(np.uint8) * 255, mode="L").filter(
            ImageFilter.GaussianBlur(radius=0.70)
        ),
        dtype=np.uint8,
    )
    alpha = Image.fromarray(np.minimum(softened, hard_alpha), mode="L")
    cropped_image.putalpha(alpha)
    return cropped_image, cropped_mask


def binary_mask_rle(mask: np.ndarray) -> dict:
    """Return an uncompressed COCO RLE in column-major order."""
    mask = np.asarray(mask, dtype=np.uint8)
    flattened = mask.flatten(order="F")
    if flattened.size == 0:
        counts = []
    else:
        changes = np.flatnonzero(flattened[1:] != flattened[:-1]) + 1
        boundaries = np.concatenate(
            (np.asarray([0]), changes, np.asarray([flattened.size]))
        )
        runs = np.diff(boundaries).astype(int).tolist()
        counts = ([0] + runs) if int(flattened[0]) else runs
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    return left, top, right - left, bottom - top


def principal_mask_geometry(mask: np.ndarray) -> tuple[float, float, float]:
    """Return silhouette major/minor extent and major-axis angle.

    Unlike an axis-aligned crop ratio, this geometry is invariant to the
    cutout's original rotation and therefore represents the aspect ratios a
    rigid rotation can physically realize without stretching pixels.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        box = mask_bbox(mask)
        if box is None:
            raise ValueError("Mask objek kosong")
        _, _, width, height = box
        if width >= height:
            return float(width), float(height), 0.0
        return float(height), float(width), math.pi / 2.0
    coordinates = np.column_stack((xs, ys)).astype(np.float64)
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    covariance = np.cov(coordinates, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    major_vector = eigenvectors[:, -1]
    minor_vector = np.asarray((-major_vector[1], major_vector[0]))
    major = float(np.ptp(coordinates @ major_vector) + 1.0)
    minor = float(np.ptp(coordinates @ minor_vector) + 1.0)
    if major < minor:
        major, minor = minor, major
        major_vector = minor_vector
    angle = math.atan2(float(major_vector[1]), float(major_vector[0]))
    return max(major, 1.0), max(minor, 1.0), angle
