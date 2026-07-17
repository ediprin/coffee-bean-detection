from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image, ImageFilter


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected foreground component."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
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
        if len(component) > len(best):
            best = component
    result = np.zeros_like(mask, dtype=bool)
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

    # A small close/open sequence suppresses isolated dust while preserving
    # narrow bean cracks and tips.
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(5))
    mask_image = mask_image.filter(ImageFilter.MinFilter(3))
    mask = np.asarray(mask_image, dtype=np.uint8) >= 128
    return fill_holes(largest_component(mask))


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
    alpha = Image.fromarray(cropped_mask.astype(np.uint8) * 255, mode="L")
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
