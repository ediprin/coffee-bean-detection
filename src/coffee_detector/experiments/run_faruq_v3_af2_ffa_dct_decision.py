"""Frozen accuracy/efficiency decision for selected-DCT FFAB2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SEEDS = (42, 123, 2026)
METRICS = ("macro_map50_95", "bottom3_class_map50_95", "worst_class_map50_95")


def _read(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def _benchmark(checkpoint: str | Path, device: torch.device, *, iterations: int, adapters_only: bool) -> dict:
    from ultralytics import YOLO
    from coffee_detector.af2_ffa import AF2FFADetectHead

    model = YOLO(str(checkpoint)).model.eval().to(device)
    if adapters_only:
        head = model.model[-1]
        if not isinstance(head, AF2FFADetectHead):
            raise TypeError(type(head).__name__)
        modules = list(head.adapters)
        samples = [
            torch.rand(1, adapter.channels, size, size, device=device)
            for adapter, size in zip(modules, (80, 40, 20))
        ]
        def forward():
            for module, sample in zip(modules, samples):
                module(sample)
    else:
        sample = torch.rand(1, 3, 640, 640, device=device)
        def forward():
            model(sample)

    values: list[float] = []
    with torch.inference_mode():
        for _ in range(10):
            forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iterations):
            started = time.perf_counter()
            forward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            values.append((time.perf_counter() - started) * 1000.0)
    peak = float(torch.cuda.max_memory_allocated(device)) / 1048576.0 if device.type == "cuda" else None
    return {
        "median_ms": _median(values),
        "peak_memory_mb": peak,
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def run_dct_decision(rfft_paths, dct_paths, output, *, device="0") -> dict:
    if len(rfft_paths) != 3 or len(dct_paths) != 3:
        raise ValueError("Perlu tiga seed untuk rFFT dan DCT")
    rfft = [_read(path) for path in rfft_paths]
    dct = [_read(path) for path in dct_paths]
    for seed, left, right in zip(SEEDS, rfft, dct):
        if left.get("arm") != "AF2FFAB2FS" or right.get("arm") != "AF2FFADCTFS":
            raise RuntimeError("Arm comparison salah")
        if int(left.get("seed")) != seed or int(right.get("seed")) != seed:
            raise RuntimeError("Seed comparison salah")
        if left.get("initial_d0_checkpoint_sha256") != right.get("initial_d0_checkpoint_sha256"):
            raise RuntimeError("D0 pair berbeda")
        if left.get("test_images_accessed") is not False or right.get("test_images_accessed") is not False:
            raise RuntimeError("Test lock dilanggar")

    aggregate = {}
    for metric in METRICS:
        lv = [float(item["metrics"][metric]) for item in rfft]
        rv = [float(item["metrics"][metric]) for item in dct]
        delta = [b - a for a, b in zip(lv, rv)]
        aggregate[metric] = {
            "rfft_mean": sum(lv) / 3.0,
            "dct_mean": sum(rv) / 3.0,
            "delta_mean": sum(delta) / 3.0,
            "deltas": {str(seed): value for seed, value in zip(SEEDS, delta)},
        }

    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    rfft_ckpt, dct_ckpt = rfft[0]["checkpoint"], dct[0]["checkpoint"]
    efficiency = {
        "rfft_adapter": _benchmark(rfft_ckpt, torch_device, iterations=200, adapters_only=True),
        "dct_adapter": _benchmark(dct_ckpt, torch_device, iterations=200, adapters_only=True),
        "rfft_full": _benchmark(rfft_ckpt, torch_device, iterations=60, adapters_only=False),
        "dct_full": _benchmark(dct_ckpt, torch_device, iterations=60, adapters_only=False),
    }
    criteria = {
        "macro_drop_no_more_than_0_20pp": aggregate["macro_map50_95"]["delta_mean"] >= -0.002,
        "bottom3_drop_no_more_than_0_50pp": aggregate["bottom3_class_map50_95"]["delta_mean"] >= -0.005,
        "worst_drop_no_more_than_1_00pp": aggregate["worst_class_map50_95"]["delta_mean"] >= -0.010,
        "adapter_at_least_20pct_faster": efficiency["dct_adapter"]["median_ms"] <= 0.80 * efficiency["rfft_adapter"]["median_ms"],
        "full_model_not_slower": efficiency["dct_full"]["median_ms"] <= efficiency["rfft_full"]["median_ms"],
        "same_parameters": efficiency["dct_full"]["parameters"] == efficiency["rfft_full"]["parameters"],
    }
    passed = all(criteria.values())
    result = {
        "format": "coffee_detector.af2_ffa.dct_efficiency_decision.v1",
        "comparison": "AF2FFADCTFS_vs_AF2FFAB2FS",
        "aggregate": aggregate,
        "efficiency": efficiency,
        "criteria": criteria,
        "decision": "PASS" if passed else "REJECT",
        "next": "RETAIN_DCT_EFFICIENT_REPLACEMENT" if passed else "RETAIN_RFFT_FFAB2",
        "test_opened": False,
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfft", nargs=3, required=True)
    parser.add_argument("--dct", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    run_dct_decision(args.rfft, args.dct, args.output, device=args.device)


if __name__ == "__main__":
    main()
