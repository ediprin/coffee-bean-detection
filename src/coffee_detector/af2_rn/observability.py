from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffee_detector.afab import AFABConfig, AFABInputEnhancer
from coffee_detector.afab.operator import af2_entropy_threshold, minmax_spatial
from coffee_detector.experiments.run_faruq_v3_baseline import load_faruq_grouped_summary
from coffee_detector.prepare_sni_fullscene import SNI21_CLASSES

from .audit import sha256
from .operator import AF2RNInputEnhancer


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "q05": float(np.quantile(ordered, 0.05)),
        "q95": float(np.quantile(ordered, 0.95)),
    }


def _tensor(path: Path, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


def _tv(value: torch.Tensor) -> torch.Tensor:
    return (value[..., 1:, :] - value[..., :-1, :]).abs().mean() + (value[..., :, 1:] - value[..., :, :-1]).abs().mean()


def _label_strength(label: Path, cue: torch.Tensor, by_class: dict[int, list[float]]) -> None:
    height, width = cue.shape[-2:]
    for raw in label.read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) != 5:
            continue
        class_id = int(fields[0]); x, y, w, h = map(float, fields[1:])
        x1 = max(0, int((x - w / 2) * width)); x2 = min(width, max(x1 + 1, int((x + w / 2) * width)))
        y1 = max(0, int((y - h / 2) * height)); y2 = min(height, max(y1 + 1, int((y + h / 2) * height)))
        by_class[class_id].append(float(cue[..., y1:y2, x1:x2].mean()))


def run_af2rn_observability_audit(
    data_root: str | Path,
    grouped_summary: str | Path,
    static_audit: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    image_size: int = 128,
    patches_per_image: int = 16,
) -> dict:
    data_root = Path(data_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    static_path = Path(static_audit).expanduser().resolve()
    load_faruq_grouped_summary(grouped_summary, data_root)
    static = json.loads(static_path.read_text(encoding="utf-8"))
    if static.get("format") != "coffee_detector.af2rn.static_audit.v1" or static.get("decision") != "PASS":
        raise RuntimeError("Static audit AF2RN belum PASS")
    if (data_root / "test").exists():
        raise RuntimeError("Test tidak boleh tersedia")
    images = sorted(path for path in (data_root / "train/images").glob("*") if path.is_file())
    if len(images) != 1665:
        raise RuntimeError(f"Kontrak train berubah: {len(images)} != 1665")
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    legacy = AFABInputEnhancer(AFABConfig(mode="af2")).to(torch_device).eval()
    candidate = AF2RNInputEnhancer().to(torch_device).eval()
    values: dict[str, list[float]] = defaultdict(list)
    radial: dict[str, list[float]] = defaultdict(list)
    class_strength: dict[int, list[float]] = defaultdict(list)
    nondegenerate = 0; different = 0
    radius = candidate.frequency_radius.to(torch_device)
    maximum = float(radius.max()); bands = {
        "low": radius <= maximum / 3,
        "mid": (radius > maximum / 3) & (radius <= 2 * maximum / 3),
        "high": radius > 2 * maximum / 3,
    }
    for index, image_path in enumerate(images, 1):
        raw = _tensor(image_path, image_size, torch_device)
        with torch.inference_mode():
            legacy_recovered = legacy.recover(raw)
            rn_recovered = candidate.recover(raw)
            legacy_output = raw + raw * minmax_spatial(legacy_recovered)
            rn_output = raw + raw * minmax_spatial(rn_recovered)
            cue = minmax_spatial(rn_recovered)
            nondegenerate += int(float(rn_recovered.std()) > 1e-6)
            different += int(float((legacy_output - rn_output).abs().max()) > 1e-6)
            values["spatial_ringing_ratio"].append(float(_tv(rn_recovered) / _tv(raw).clamp_min(1e-8)))
            _label_strength(data_root / "train/labels" / image_path.with_suffix(".txt").name, cue, class_strength)

            columns = F.unfold(raw, kernel_size=32, stride=16)
            count = columns.shape[-1]
            chosen = torch.linspace(0, count - 1, min(patches_per_image, count), device=torch_device).round().long().unique()
            patches = columns[:, :, chosen].transpose(1, 2).reshape(-1, 3, 32, 32)
            frequency = torch.fft.fftshift(torch.fft.fft2(patches.float(), norm="ortho"), dim=(-2, -1))
            magnitude = frequency.abs()
            legacy_weight = legacy._af2_weight(frequency)
            rn_weight = candidate._af2_weight(frequency)
            total = magnitude.sum(dim=(-2, -1)).clamp_min(1e-8)
            values["retained_spectral_mass"].append(float(((magnitude * rn_weight).sum(dim=(-2, -1)) / total).median()))
            legacy_mask, rn_mask = legacy_weight > 0, rn_weight > 0
            intersection = (legacy_mask & rn_mask).sum(dim=(-2, -1)).float()
            union = (legacy_mask | rn_mask).sum(dim=(-2, -1)).float().clamp_min(1)
            values["af2_mask_jaccard"].append(float((intersection / union).median()))

            density = candidate.angular_density(frequency)
            probability = density / density.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1)
            threshold = af2_entropy_threshold(probability, gamma=0.1)
            normalized = density / density.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            values["angular_occupancy"].append(float((normalized > threshold.unsqueeze(-1)).float().mean()))
            values["entropy"].append(float(entropy.median()))
            values["threshold"].append(float(threshold.median()))
            for name, mask in bands.items():
                band_mass = magnitude[..., mask].sum(dim=-1).clamp_min(1e-8)
                kept = (magnitude * rn_weight)[..., mask].sum(dim=-1)
                radial[name].append(float((kept / band_mass).median()))
        if index % 100 == 0 or index == len(images):
            print(f"AF2RN OBSERVABILITY {index}/{len(images)}", flush=True)

    nondegenerate_fraction = nondegenerate / len(images)
    difference_fraction = different / len(images)
    retained_median = statistics.median(values["retained_spectral_mass"])
    gates = {
        "all_1665_train_images_observed": len(images) == 1665,
        "nondegenerate_fraction_at_least_95_percent": nondegenerate_fraction >= 0.95,
        "differs_from_af2_at_least_95_percent": difference_fraction >= 0.95,
        "median_retained_mass_above_2_percent": retained_median > 0.02,
        "median_retained_mass_below_98_percent": retained_median < 0.98,
        "all_21_classes_have_box_strength": set(class_strength) == set(range(21)),
        "validation_not_read": True,
        "test_accessed": False,
    }
    decision = "PASS" if all(value for key, value in gates.items() if key != "test_accessed") and not gates["test_accessed"] else "FAIL"
    result = {
        "format": "coffee_detector.af2rn.observability.v1",
        "decision": decision,
        "static_audit": str(static_path),
        "static_audit_sha256": sha256(static_path),
        "images": len(images),
        "image_size": image_size,
        "patches_per_image": patches_per_image,
        "nondegenerate_fraction": nondegenerate_fraction,
        "different_from_af2_fraction": difference_fraction,
        "distributions": {key: _summary(rows) for key, rows in values.items()},
        "radial_retention": {key: _summary(rows) for key, rows in radial.items()},
        "per_class_gt_box_cue_strength": {
            SNI21_CLASSES[index]: {"boxes": len(class_strength[index]), **_summary(class_strength[index])}
            for index in range(21)
        },
        "gates": gates,
        "training_authorized": decision == "PASS",
        "validation_images_accessed": False,
        "test_images_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only AF2RN observability")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--grouped-summary", required=True)
    parser.add_argument("--static-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_af2rn_observability_audit(args.data_root, args.grouped_summary, args.static_audit, args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
