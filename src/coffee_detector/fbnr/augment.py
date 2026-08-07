from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FBNRConfig:
    """Coffee-transfer configuration for Xu et al. DSRDet FBNR.

    `background_gradient` implements the paper's BRBB idea: Sobel gradients,
    point-wise selection of the stronger source/donor-background gradient, then
    Poisson reconstruction in the Fourier domain. `foreground_only` keeps the
    paper's Gaussian concealment but replaces the aircraft-specific oriented
    cross prior with two centers sampled along the horizontal/vertical central
    axes of each coffee-bean box. `stochastic_decoupled` is a matched-update
    approximation of the paper's three parallel input spaces: original,
    foreground-regularized, and background-regularized samples are selected
    per image with frozen probabilities instead of tripling the batch size.
    """

    mode: str = "stochastic_decoupled"
    sigma_ratio: float = 3.0
    conceal_radius_min: float = 0.5
    conceal_radius_max: float = 0.8
    original_probability: float = 0.34
    foreground_probability: float = 0.33
    background_probability: float = 0.33

    @classmethod
    def from_mapping(cls, payload: "FBNRConfig | dict[str, Any] | None") -> "FBNRConfig":
        result = payload if isinstance(payload, cls) else cls(**dict(payload or {}))
        valid_modes = {
            "foreground_only",
            "background_linear",
            "background_gradient",
            "stochastic_decoupled",
        }
        if result.mode not in valid_modes:
            raise ValueError(f"mode harus salah satu {sorted(valid_modes)}")
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
    """DSRDet Eqs. (1)-(3): Gaussian foreground/background decoupling mask."""
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
    sigma_ratio: float = 3.0,
) -> torch.Tensor:
    """Coffee-adapted DSRDet FGRC, corresponding to Eqs. (9)-(10).

    DSRDet uses an aircraft-specific oriented cross prior. We retain its two-axis
    sampling principle but use the horizontal and vertical central axes of each
    axis-aligned coffee box. The paper-selected dynamic radius range [0.5, 0.8]
    is preserved by default.
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
        k = torch.empty((), device=images.device, dtype=images.dtype).uniform_(
            float(radius_min), float(radius_max)
        )
        radius = k * short_side
        sigma = (radius / float(sigma_ratio)).clamp_min(1e-4)
        # One point on each central axis. This removes the aircraft geometry
        # prior while retaining the paper's two-axis concealment logic.
        dx = (torch.rand((), device=images.device, dtype=images.dtype) - 0.5) * 0.7 * bw
        dy = (torch.rand((), device=images.device, dtype=images.dtype) - 0.5) * 0.7 * bh
        centers = ((cx + dx, cy), (cx, cy + dy))
        for center_x, center_y in centers:
            gaussian = torch.exp(
                -((xx - center_x).square() + (yy - center_y).square())
                / (2.0 * sigma.square())
            )
            erase[image_id, 0] = torch.maximum(erase[image_id, 0], gaussian)
    return (images * (1.0 - erase.clamp(0.0, 1.0))).clamp_(0.0, 1.0)


def background_linear_blend(images: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    """Spatial-domain control corresponding to the linear-blending ablation."""
    batch = images.shape[0]
    if batch < 2:
        return images.clone()
    permutation = torch.roll(torch.arange(batch, device=images.device), shifts=1)
    donor = images[permutation]
    donor_mask = foreground_mask[permutation]
    donor_background = (1.0 - donor_mask) * donor
    return (foreground_mask * images + (1.0 - foreground_mask) * donor_background).clamp_(0.0, 1.0)


def _sobel_gradients(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    channels = images.shape[1]
    kx = images.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    ky = kx.t()
    gx = F.conv2d(images, kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), padding=1, groups=channels)
    gy = F.conv2d(images, ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), padding=1, groups=channels)
    return gx, gy


def _divergence(gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    dx = gx - torch.roll(gx, shifts=1, dims=-1)
    dy = gy - torch.roll(gy, shifts=1, dims=-2)
    # Remove periodic wrap contributions at the image boundaries.
    dx[..., 0] = gx[..., 0]
    dy[..., 0, :] = gy[..., 0, :]
    return dx + dy


def _poisson_reconstruct(divergence: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """FFT Poisson solver for DSRDet Eq. (8), using a discrete Laplacian."""
    _, _, height, width = divergence.shape
    fy = torch.fft.fftfreq(height, device=divergence.device, dtype=divergence.dtype)
    fx = torch.fft.fftfreq(width, device=divergence.device, dtype=divergence.dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    eigen = 2.0 * torch.cos(2.0 * torch.pi * xx) + 2.0 * torch.cos(2.0 * torch.pi * yy) - 4.0
    div_hat = torch.fft.fft2(divergence)
    safe = eigen.clone()
    safe[0, 0] = 1.0
    out_hat = div_hat / safe
    # Poisson reconstruction leaves the DC term undefined; preserve source mean.
    source_mean = source.mean(dim=(-2, -1))
    out_hat[..., 0, 0] = source_mean * float(height * width)
    output = torch.fft.ifft2(out_hat).real
    return output.clamp(0.0, 1.0)


def background_gradient_blend(images: torch.Tensor, foreground_mask: torch.Tensor) -> torch.Tensor:
    """Gradient-domain BRBB transfer following DSRDet Eqs. (4)-(8)."""
    batch = images.shape[0]
    if batch < 2:
        return images.clone()
    permutation = torch.roll(torch.arange(batch, device=images.device), shifts=1)
    donor = images[permutation]
    donor_mask = foreground_mask[permutation]
    donor_background = (1.0 - donor_mask) * donor
    source_gx, source_gy = _sobel_gradients(images)
    donor_gx, donor_gy = _sobel_gradients(donor_background)
    source_mag = torch.sqrt(source_gx.square() + source_gy.square() + 1e-12)
    donor_mag = torch.sqrt(donor_gx.square() + donor_gy.square() + 1e-12)
    choose_source = source_mag > donor_mag
    blend_gx = torch.where(choose_source, source_gx, donor_gx)
    blend_gy = torch.where(choose_source, source_gy, donor_gy)
    reconstructed = _poisson_reconstruct(_divergence(blend_gx, blend_gy), images)
    # Preserve foreground semantics explicitly, as required by BRBB.
    return (foreground_mask * images + (1.0 - foreground_mask) * reconstructed).clamp_(0.0, 1.0)


def apply_fbnr_transfer(
    images: torch.Tensor,
    bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    config: FBNRConfig | dict[str, Any] | None,
) -> torch.Tensor:
    frozen = FBNRConfig.from_mapping(config)
    foreground_mask = build_foreground_soft_mask(images, bboxes, batch_idx, sigma_ratio=frozen.sigma_ratio)
    if frozen.mode == "foreground_only":
        return foreground_random_conceal(
            images,
            bboxes,
            batch_idx,
            radius_min=frozen.conceal_radius_min,
            radius_max=frozen.conceal_radius_max,
            sigma_ratio=frozen.sigma_ratio,
        )
    if frozen.mode == "background_linear":
        return background_linear_blend(images, foreground_mask)
    if frozen.mode == "background_gradient":
        return background_gradient_blend(images, foreground_mask)

    concealed = foreground_random_conceal(
        images,
        bboxes,
        batch_idx,
        radius_min=frozen.conceal_radius_min,
        radius_max=frozen.conceal_radius_max,
        sigma_ratio=frozen.sigma_ratio,
    )
    background = background_gradient_blend(images, foreground_mask)
    selector = torch.rand(images.shape[0], device=images.device)
    first = frozen.original_probability
    second = first + frozen.foreground_probability
    output = images.clone()
    fg_rows = (selector >= first) & (selector < second)
    bg_rows = selector >= second
    output[fg_rows] = concealed[fg_rows]
    output[bg_rows] = background[bg_rows]
    return output.clamp_(0.0, 1.0)
