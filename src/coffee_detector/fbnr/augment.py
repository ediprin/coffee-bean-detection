from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FBNRConfig:
    mode: str = "stochastic_decoupled"
    sigma_ratio: float = 3.0
    conceal_radius_min: float = 0.15
    conceal_radius_max: float = 0.35
    original_probability: float = 0.34
    foreground_probability: float = 0.33
    background_probability: float = 0.33

    @classmethod
    def from_mapping(cls, payload: "FBNRConfig | dict[str, Any] | None") -> "FBNRConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        if result.mode not in {"foreground_only", "stochastic_decoupled"}:
            raise ValueError("mode harus foreground_only atau stochastic_decoupled")
        if result.sigma_ratio <= 0:
            raise ValueError("sigma_ratio harus positif")
        if not 0 < result.conceal_radius_min <= result.conceal_radius_max <= 1:
            raise ValueError("rentang conceal radius tidak valid")
        probs = (
            result.original_probability,
            result.foreground_probability,
            result.background_probability,
        )
        if any(value < 0 for value in probs):
            raise ValueError("probabilitas tidak boleh negatif")
        if result.mode == "stochastic_decoupled" and abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("probabilitas stochastic_decoupled harus berjumlah 1")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_grid(height: int, width: int, *, device, dtype):
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def build_foreground_soft_mask(
    images: torch.Tensor,
    bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    sigma_ratio: float,
) -> torch.Tensor:
    """Gaussian soft foreground mask corresponding to DSRDet Eqs. (1)-(3).

    Boxes are normalized xywh. Multiple instance Gaussians are combined by a
    pixel-wise maximum, matching the paper's foreground/background decoupling.
    """
    if images.ndim != 4:
        raise ValueError("images harus [B,C,H,W]")
    batch, _, height, width = images.shape
    mask = images.new_zeros((batch, 1, height, width))
    if not len(bboxes):
        return mask
    xx, yy = _normalized_grid(height, width, device=images.device, dtype=images.dtype)
    boxes = bboxes.to(device=images.device, dtype=images.dtype)
    indices = batch_idx.to(device=images.device, dtype=torch.long).reshape(-1)
    for row, image_id in zip(boxes, indices):
        cx, cy, bw, bh = row[:4]
        radius = torch.maximum(bw, bh).clamp_min(1.0 / max(height, width))
        sigma = (radius / float(sigma_ratio)).clamp_min(1e-4)
        gaussian = torch.exp(-((xx - cx).square() + (yy - cy).square()) / (2.0 * sigma.square()))
        mask[image_id, 0] = torch.maximum(mask[image_id, 0], gaussian)
    return mask.clamp_(0.0, 1.0)


def foreground_random_conceal(
    images: torch.Tensor,
    bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    radius_min: float,
    radius_max: float,
) -> torch.Tensor:
    """Structure-agnostic Gaussian foreground concealment.

    DSRDet samples concealment centers using an aircraft cross-shape prior.
    Coffee beans do not justify that prior, so two centers are sampled inside
    each annotated box instead. The operator remains soft Gaussian masking.
    """
    batch, _, height, width = images.shape
    erase = images.new_zeros((batch, 1, height, width))
    if not len(bboxes):
        return images.clone()
    xx, yy = _normalized_grid(height, width, device=images.device, dtype=images.dtype)
    boxes = bboxes.to(device=images.device, dtype=images.dtype)
    indices = batch_idx.to(device=images.device, dtype=torch.long).reshape(-1)
    for row, image_id in zip(boxes, indices):
        cx, cy, bw, bh = row[:4]
        short_side = torch.minimum(bw, bh).clamp_min(1.0 / max(height, width))
        for _ in range(2):
            offset = torch.rand(2, device=images.device, dtype=images.dtype) - 0.5
            center_x = (cx + offset[0] * 0.6 * bw).clamp(0.0, 1.0)
            center_y = (cy + offset[1] * 0.6 * bh).clamp(0.0, 1.0)
            k = torch.empty((), device=images.device, dtype=images.dtype).uniform_(
                float(radius_min), float(radius_max)
            )
            sigma = (k * short_side).clamp_min(1e-4)
            gaussian = torch.exp(
                -((xx - center_x).square() + (yy - center_y).square())
                / (2.0 * sigma.square())
            )
            erase[image_id, 0] = torch.maximum(erase[image_id, 0], gaussian)
    return images * (1.0 - erase.clamp(0.0, 1.0))


def background_soft_replace(
    images: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> torch.Tensor:
    """Structure-agnostic background regularization.

    The paper uses Sobel-gradient selection and Poisson reconstruction. This
    screening transfer instead performs soft background substitution while
    explicitly suppressing donor foreground via its Gaussian mask. It tests the
    foreground/background-decoupling hypothesis without claiming the paper's
    gradient-domain reconstruction.
    """
    batch = images.shape[0]
    if batch < 2:
        return images.clone()
    permutation = torch.randperm(batch, device=images.device)
    donor = images[permutation]
    donor_mask = foreground_mask[permutation]
    source_mask = foreground_mask
    donor_background = (1.0 - donor_mask) * donor
    fallback = donor_mask * images
    mixed_background = donor_background + fallback
    return source_mask * images + (1.0 - source_mask) * mixed_background


def apply_fbnr_transfer(
    images: torch.Tensor,
    bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    config: FBNRConfig | dict[str, Any] | None,
) -> torch.Tensor:
    frozen = FBNRConfig.from_mapping(config)
    foreground_mask = build_foreground_soft_mask(
        images, bboxes, batch_idx, sigma_ratio=frozen.sigma_ratio
    )
    concealed = foreground_random_conceal(
        images,
        bboxes,
        batch_idx,
        radius_min=frozen.conceal_radius_min,
        radius_max=frozen.conceal_radius_max,
    )
    if frozen.mode == "foreground_only":
        return concealed
    background = background_soft_replace(images, foreground_mask)
    selector = torch.rand(images.shape[0], device=images.device)
    first = frozen.original_probability
    second = first + frozen.foreground_probability
    output = images.clone()
    fg_rows = (selector >= first) & (selector < second)
    bg_rows = selector >= second
    output[fg_rows] = concealed[fg_rows]
    output[bg_rows] = background[bg_rows]
    return output.clamp_(0.0, 1.0)
