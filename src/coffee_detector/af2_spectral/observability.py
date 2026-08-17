from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffee_detector.afab.operator import af2_entropy_threshold, minmax_spatial
from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary

from .config import frozen_arm_config
from .operator import SpectralInputEnhancer


def _tensor(path: Path, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"mean": float("nan"), "median": float("nan"), "q05": float("nan"), "q95": float("nan")}
    def quantile(fraction: float) -> float:
        return ordered[min(int(round((len(ordered) - 1) * fraction)), len(ordered) - 1)]
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "q05": quantile(0.05),
        "q95": quantile(0.95),
    }


def _conditions(value: torch.Tensor) -> dict[str, torch.Tensor]:
    mean = value.mean(dim=(-2, -1), keepdim=True)
    warm = value * value.new_tensor((1.12, 1.0, 0.88)).view(1, 3, 1, 1)
    cool = value * value.new_tensor((0.88, 1.0, 1.12)).view(1, 3, 1, 1)
    shadow = torch.ones_like(value)
    shadow[..., :, : value.shape[-1] // 2] = 0.55
    return {
        "dark_ev10": (value * 0.5).clamp(0, 1),
        "bright_ev10": (value * 1.5).clamp(0, 1),
        "contrast075": ((value - mean) * 0.75 + mean).clamp(0, 1),
        "contrast125": ((value - mean) * 1.25 + mean).clamp(0, 1),
        "warm": warm.clamp(0, 1),
        "cool": cool.clamp(0, 1),
        "shadow55": value * shadow,
    }


def run_spectral_observability_audit(
    data_root: str | Path,
    grouped_summary: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 256,
    patches_per_image: int = 16,
    stability_images: int = 128,
) -> dict:
    data_root = Path(data_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    layout = discover_layout(data_root)
    if "test" in layout.splits or (data_root / "test").exists():
        raise RuntimeError("Observability audit tidak boleh mengekspos test")
    image_root = layout.splits["train"][0]
    images = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(images) != 1665:
        raise RuntimeError(f"Kontrak train Faruq-v3 berubah: {len(images)} gambar")
    torch_device = torch.device(device)
    polar = SpectralInputEnhancer(frozen_arm_config("AF2SOFT")).to(torch_device)
    luminance = SpectralInputEnhancer(frozen_arm_config("AF2LUM")).to(torch_device)
    window = polar.analysis_window.to(torch_device)
    entropy_values: list[float] = []
    threshold_values: list[float] = []
    retained_mass: list[float] = []
    rectangle_concentration: list[float] = []
    hann_concentration: list[float] = []
    channel_disagreement: list[float] = []
    radial_energy = [[] for _ in range(3)]
    for image_index, path in enumerate(images, 1):
        value = _tensor(path, image_size, torch_device)
        columns = F.unfold(value, kernel_size=32, stride=16)
        count = columns.shape[-1]
        indices = torch.linspace(0, count - 1, patches_per_image, device=torch_device).long().unique()
        patches = columns[:, :, indices].transpose(1, 2).reshape(-1, 3, 32, 32)
        with torch.inference_mode(), torch.autocast(device_type=torch_device.type, enabled=False):
            rectangle = torch.fft.fftshift(torch.fft.fft2(patches.float(), norm="ortho"), dim=(-2, -1))
            tapered = torch.fft.fftshift(
                torch.fft.fft2(patches.float() * window, norm="ortho"), dim=(-2, -1)
            )
            for spectrum, target in ((rectangle, rectangle_concentration), (tapered, hann_concentration)):
                magnitude = spectrum.abs().flatten(2)
                top = max(1, int(magnitude.shape[-1] * 0.10))
                concentration = magnitude.topk(top, dim=-1).values.sum(-1) / magnitude.sum(-1).clamp_min(1e-8)
                target.extend(concentration.flatten().cpu().tolist())
            magnitude = tapered.abs().flatten(2)
            angle = polar.angle_bin.reshape(1, 1, -1).expand(magnitude.shape[0], 3, -1)
            density = magnitude.new_zeros((magnitude.shape[0], 3, 16))
            density.scatter_add_(-1, angle, magnitude)
            probability = density / density.sum(-1, keepdim=True).clamp_min(1e-8)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1)
            threshold = af2_entropy_threshold(probability, 0.1)
            normalized = density / density.amax(-1, keepdim=True).clamp_min(1e-8)
            hard = torch.where(normalized <= threshold.unsqueeze(-1), 0.0, normalized)
            retained = (density * (hard > 0)).sum(-1) / density.sum(-1).clamp_min(1e-8)
            entropy_values.extend(entropy.flatten().cpu().tolist())
            threshold_values.extend(threshold.flatten().cpu().tolist())
            retained_mass.extend(retained.flatten().cpu().tolist())
            channel_weight = polar._direction_weight(tapered)
            shared_weight = luminance._direction_weight(tapered)
            channel_disagreement.extend(
                (channel_weight - shared_weight).abs().mean((-2, -1)).flatten().cpu().tolist()
            )
            radial = polar.radial_bin.flatten()
            total = magnitude.sum(-1).clamp_min(1e-8)
            for band in range(3):
                fraction = magnitude[..., radial == band].sum(-1) / total
                radial_energy[band].extend(fraction.flatten().cpu().tolist())
        if image_index % 100 == 0 or image_index == len(images):
            print(f"OBSERVABILITY {image_index}/{len(images)}", flush=True)

    stability_paths = [images[index] for index in np.linspace(0, len(images) - 1, stability_images, dtype=int)]
    stability = {
        arm: {name: [] for name in _conditions(torch.zeros(1, 3, 8, 8)).keys()}
        for arm in ("AF2C", "AF2LUM")
    }
    frontends = {
        arm: SpectralInputEnhancer(frozen_arm_config(arm)).to(torch_device)
        for arm in stability
    }
    for path in stability_paths:
        value = _tensor(path, 128, torch_device)
        variants = _conditions(value)
        with torch.inference_mode():
            for arm, frontend in frontends.items():
                clean = minmax_spatial(frontend.recover(value)).flatten(1)
                for condition, transformed in variants.items():
                    cue = minmax_spatial(frontend.recover(transformed)).flatten(1)
                    score = F.cosine_similarity(clean, cue, dim=1).mean()
                    stability[arm][condition].append(float(score))

    result = {
        "format": "coffee_detector.af2_spectral.observability.v1",
        "split": "train",
        "images": len(images),
        "patches_per_image": patches_per_image,
        "hyperparameter_tuning_performed": False,
        "angular_occupancy": {
            "af2_360_occupied": int(torch.unique(frontends["AF2C"].legacy.angle_bin).numel()),
            "af2_360_empty": 360 - int(torch.unique(frontends["AF2C"].legacy.angle_bin).numel()),
            "orientation16_occupied": int(torch.unique(polar.angle_bin).numel()),
        },
        "entropy": _summary(entropy_values),
        "threshold": _summary(threshold_values),
        "retained_spectral_mass": _summary(retained_mass),
        "spectral_concentration": {
            "rectangular": _summary(rectangle_concentration),
            "sqrt_hann": _summary(hann_concentration),
        },
        "radial_energy": {f"band_{index}": _summary(values) for index, values in enumerate(radial_energy)},
        "rgb_vs_luminance_gate_disagreement": _summary(channel_disagreement),
        "cue_cosine_stability": {
            arm: {condition: _summary(values) for condition, values in conditions.items()}
            for arm, conditions in stability.items()
        },
        "test_images_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only AF2 spectral observability audit")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patches-per-image", type=int, default=16)
    parser.add_argument("--stability-images", type=int, default=128)
    args = parser.parse_args()
    result = run_spectral_observability_audit(
        args.data_root,
        args.grouped_summary,
        args.output,
        device=args.device,
        image_size=args.image_size,
        patches_per_image=args.patches_per_image,
        stability_images=args.stability_images,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
