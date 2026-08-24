from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def benchmark_checkpoint(
    checkpoint: str | Path,
    *,
    image_size: int = 640,
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 100,
    device: str = "cpu",
) -> dict:
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup tidak boleh negatif dan iterations harus positif")

    torch_device = (
        f"cuda:{device}"
        if str(device).isdigit()
        else str(device)
    )
    if torch_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")
    yolo = YOLO(str(checkpoint))
    model = yolo.model.to(torch_device).eval()
    tensor = torch.randn(batch_size, 3, image_size, image_size, device=torch_device)
    synchronize = torch.cuda.synchronize if torch_device.startswith("cuda") else lambda: None

    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        synchronize()
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(tensor)
            synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    fp32_bytes = sum(parameter.numel() * 4 for parameter in model.parameters())
    latency = float(np.mean(samples))
    return {
        "checkpoint": str(checkpoint),
        "device": torch_device,
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "parameters": int(parameters),
        "trainable_parameters": int(trainable),
        "model_size_fp32_mb": float(fp32_bytes / (1024**2)),
        "latency_ms_per_batch": latency,
        "latency_ms_per_image": latency / batch_size,
        "throughput_images_per_second": float(batch_size * 1000.0 / latency),
        "warmup": int(warmup),
        "iterations": int(iterations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark checkpoint CoffeeFG pada perangkat dan ukuran input yang sama."
    )
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows = [
        benchmark_checkpoint(
            checkpoint,
            image_size=args.image_size,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            device=args.device,
        )
        for checkpoint in args.checkpoint
    ]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(
            Path(row["checkpoint"]).parent.parent.name,
            f"params={row['parameters']:,}",
            f"size={row['model_size_fp32_mb']:.2f} MB",
            f"latency={row['latency_ms_per_image']:.3f} ms/image",
            f"throughput={row['throughput_images_per_second']:.1f} image/s",
        )
    print("SAVED:", output)


if __name__ == "__main__":
    main()
