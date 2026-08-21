from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path


SEEDS = (42, 123, 2026)
MODELS = ("D0FT", "AF2")
IMAGE_SIZE = 640
BATCH_SIZE = 1
WARMUP_ITERATIONS = 30
MEASURED_ITERATIONS = 100


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Daftar latency kosong")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _cuda_device(device: str):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Audit efisiensi memerlukan satu GPU CUDA")
    value = str(device)
    if value.isdigit():
        value = f"cuda:{value}"
    resolved = torch.device(value)
    if resolved.type != "cuda":
        raise RuntimeError("Audit efisiensi hanya sah pada GPU CUDA")
    return resolved


def _environment(device) -> dict:
    import torch
    import ultralytics

    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_capability": list(torch.cuda.get_device_capability(index)),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "ultralytics_version": str(ultralytics.__version__),
    }


def _benchmark_checkpoint(
    checkpoint: Path,
    model_name: str,
    seed: int,
    device,
    *,
    warmup: int,
    iterations: int,
) -> dict:
    import torch
    from ultralytics import YOLO

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize(device)
    allocator_before = int(torch.cuda.memory_allocated(device))

    network = YOLO(str(checkpoint)).model.to(device).float().eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(20260821)
    sample = torch.rand(
        BATCH_SIZE,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    parameter_count = int(sum(parameter.numel() for parameter in network.parameters()))
    trainable_parameter_count = int(
        sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
    )
    parameter_bytes = int(
        sum(parameter.numel() * parameter.element_size() for parameter in network.parameters())
    )
    buffer_bytes = int(
        sum(buffer.numel() * buffer.element_size() for buffer in network.buffers())
    )
    state_tensor_bytes = int(
        sum(
            tensor.numel() * tensor.element_size()
            for tensor in network.state_dict().values()
            if torch.is_tensor(tensor)
        )
    )

    with torch.inference_mode():
        for _ in range(warmup):
            network(sample)
        torch.cuda.synchronize(device)
        steady_allocated = int(torch.cuda.memory_allocated(device))
        steady_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for _ in range(iterations):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            network(sample)
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    mean_ms = float(statistics.mean(latencies))
    result = {
        "format": "coffee_detector.af2.efficiency_model.v1",
        "model": model_name,
        "seed": int(seed),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_file_bytes": int(checkpoint.stat().st_size),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "state_tensor_bytes": state_tensor_bytes,
        "latency_mean_ms": mean_ms,
        "latency_median_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "throughput_images_per_second": float(BATCH_SIZE * 1000.0 / mean_ms),
        "allocator_before_bytes": allocator_before,
        "steady_allocated_bytes": steady_allocated,
        "steady_reserved_bytes": steady_reserved,
        "model_sample_resident_bytes": int(steady_allocated - allocator_before),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_inference_peak_bytes": int(max(0, peak_allocated - steady_allocated)),
        "training_executed": False,
        "test_images_accessed": False,
    }
    del network, sample
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return result


def _load_pair_report(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "coffee_detector.af2.efficiency_pair.v1":
        raise RuntimeError(f"Format pair report tidak dikenal: {path}")
    if payload.get("training_executed") is not False:
        raise RuntimeError(f"Audit efisiensi tidak boleh training: {path}")
    if payload.get("test_images_accessed") is not False:
        raise RuntimeError(f"Audit efisiensi tidak boleh membuka test: {path}")
    if set(payload.get("models", {})) != set(MODELS):
        raise RuntimeError(f"Pair report tidak lengkap: {path}")
    for model in MODELS:
        if payload["models"][model].get("test_images_accessed") is not False:
            raise RuntimeError(f"Model report membuka test: {path}")
    return payload


def _aggregate_metric(pairs: dict[int, dict], metric: str) -> dict:
    controls = [float(pairs[seed]["models"]["D0FT"][metric]) for seed in SEEDS]
    candidates = [float(pairs[seed]["models"]["AF2"][metric]) for seed in SEEDS]
    deltas = [candidate - control for control, candidate in zip(controls, candidates)]
    ratios = [candidate / control if control else float("nan") for control, candidate in zip(controls, candidates)]
    return {
        "d0ft_mean": float(statistics.mean(controls)),
        "d0ft_std": float(statistics.pstdev(controls)),
        "af2_mean": float(statistics.mean(candidates)),
        "af2_std": float(statistics.pstdev(candidates)),
        "delta_mean": float(statistics.mean(deltas)),
        "ratio_mean": float(statistics.mean(ratios)),
        "paired": {
            str(seed): {
                "D0FT": controls[index],
                "AF2": candidates[index],
                "delta": deltas[index],
                "ratio": ratios[index],
            }
            for index, seed in enumerate(SEEDS)
        },
    }


def build_af2_efficiency_summary(
    pair_reports: list[str | Path], output: str | Path
) -> dict:
    if len(pair_reports) != len(SEEDS):
        raise ValueError("Audit efisiensi memerlukan tepat tiga pair report")
    pairs = {}
    for path in pair_reports:
        payload = _load_pair_report(path)
        seed = int(payload["seed"])
        if seed not in SEEDS or seed in pairs:
            raise RuntimeError(f"Seed pair report tidak sah/duplikat: {seed}")
        pairs[seed] = payload
    if set(pairs) != set(SEEDS):
        raise RuntimeError("Pair report tidak mencakup seed 42/123/2026")

    metrics = (
        "parameter_count",
        "trainable_parameter_count",
        "parameter_bytes",
        "buffer_bytes",
        "state_tensor_bytes",
        "checkpoint_file_bytes",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p95_ms",
        "throughput_images_per_second",
        "model_sample_resident_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "incremental_inference_peak_bytes",
    )
    aggregate = {metric: _aggregate_metric(pairs, metric) for metric in metrics}
    parameter_equal = all(
        pairs[seed]["models"]["D0FT"]["parameter_count"]
        == pairs[seed]["models"]["AF2"]["parameter_count"]
        for seed in SEEDS
    )
    environment_pairs_match = all(
        pairs[seed]["environment_before"] == pairs[seed]["environment_after"]
        for seed in SEEDS
    )
    payload = {
        "format": "coffee_detector.af2.efficiency_summary.v1",
        "protocol": "faruq-v3-af2-efficiency-audit-v1",
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "settings": {
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "dtype": "float32",
            "timing_scope": "synchronized_gpu_tensor_forward",
            "input_transfer_included": False,
            "postprocessing_included": False,
            "warmup_iterations": WARMUP_ITERATIONS,
            "measured_iterations": MEASURED_ITERATIONS,
        },
        "aggregate": aggregate,
        "gates": {
            "paired_environment_unchanged": environment_pairs_match,
            "detector_parameter_count_equal": parameter_equal,
            "all_reports_training_false": True,
            "all_reports_test_false": True,
        },
        "parameter_free_frontend_supported": bool(parameter_equal),
        "training_executed": False,
        "test_images_accessed": False,
        "decision": "DESCRIPTIVE_DEPLOYMENT_AUDIT",
        "claim_limit": (
            "same-device tensor-forward comparison; checkpoint size is not a parameter-count proxy; "
            "standard YOLO FLOPs are omitted because they do not count the FFT frontend"
        ),
    }
    if not all(payload["gates"].values()):
        raise RuntimeError(f"Audit efisiensi tidak valid: {payload['gates']}")
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload["summary"] = str(destination)
    return payload


def run_af2_efficiency_audit(
    d0ft_checkpoints: list[str | Path],
    af2_checkpoints: list[str | Path],
    output_root: str | Path,
    *,
    device: str = "0",
) -> dict:
    if len(d0ft_checkpoints) != len(SEEDS) or len(af2_checkpoints) != len(SEEDS):
        raise ValueError("D0FT dan AF2 masing-masing memerlukan tiga checkpoint")
    checkpoints = {
        seed: {
            "D0FT": Path(d0ft_checkpoints[index]).expanduser().resolve(),
            "AF2": Path(af2_checkpoints[index]).expanduser().resolve(),
        }
        for index, seed in enumerate(SEEDS)
    }
    for seed in SEEDS:
        for model in MODELS:
            if not checkpoints[seed][model].is_file():
                raise FileNotFoundError(checkpoints[seed][model])

    destination = Path(output_root).expanduser().resolve()
    reports_root = destination / "pair_reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "coffee_detector.af2.efficiency_contract.v1",
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "settings": {
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "dtype": "float32",
            "warmup_iterations": WARMUP_ITERATIONS,
            "measured_iterations": MEASURED_ITERATIONS,
        },
        "checkpoints": {
            str(seed): {
                model: {
                    "sha256": _sha256(checkpoints[seed][model]),
                    "bytes": checkpoints[seed][model].stat().st_size,
                }
                for model in MODELS
            }
            for seed in SEEDS
        },
        "training_executed": False,
        "test_images_accessed": False,
    }
    contract_path = destination / "input_contract.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError("Kontrak audit berubah; gunakan output root baru")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")

    torch_device = _cuda_device(device)
    pair_paths = []
    for index, seed in enumerate(SEEDS):
        pair_path = reports_root / f"D0FT_vs_AF2_seed{seed}.json"
        if pair_path.is_file():
            pair = _load_pair_report(pair_path)
            expected = contract["checkpoints"][str(seed)]
            actual = {
                model: {
                    "sha256": pair["models"][model]["checkpoint_sha256"],
                    "bytes": pair["models"][model]["checkpoint_file_bytes"],
                }
                for model in MODELS
            }
            if actual != expected:
                raise RuntimeError(f"Pair report seed {seed} tidak cocok dengan kontrak")
            print(f"REUSE pair seed {seed}: {pair_path}", flush=True)
        else:
            order = MODELS if index % 2 == 0 else tuple(reversed(MODELS))
            before = _environment(torch_device)
            measured = {}
            for model in order:
                print(f"BENCHMARK {model} seed {seed}", flush=True)
                measured[model] = _benchmark_checkpoint(
                    checkpoints[seed][model],
                    model,
                    seed,
                    torch_device,
                    warmup=WARMUP_ITERATIONS,
                    iterations=MEASURED_ITERATIONS,
                )
            pair = {
                "format": "coffee_detector.af2.efficiency_pair.v1",
                "seed": seed,
                "measurement_order": list(order),
                "environment_before": before,
                "environment_after": _environment(torch_device),
                "models": measured,
                "training_executed": False,
                "test_images_accessed": False,
            }
            pair_path.write_text(
                json.dumps(pair, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            _load_pair_report(pair_path)
            print(f"READY pair seed {seed}: {pair_path}", flush=True)
        pair_paths.append(pair_path)

    return build_af2_efficiency_summary(
        pair_paths, destination / "af2_efficiency_summary.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Same-GPU paired D0FT versus AF2 efficiency audit"
    )
    parser.add_argument("--d0ft-checkpoints", nargs=3, required=True)
    parser.add_argument("--af2-checkpoints", nargs=3, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run_af2_efficiency_audit(
        args.d0ft_checkpoints,
        args.af2_checkpoints,
        args.output_root,
        device=args.device,
    )
    headline = {
        metric: {
            key: result["aggregate"][metric][key]
            for key in ("d0ft_mean", "af2_mean", "delta_mean", "ratio_mean")
        }
        for metric in (
            "parameter_count",
            "latency_median_ms",
            "latency_p95_ms",
            "throughput_images_per_second",
            "peak_allocated_bytes",
        )
    }
    print(json.dumps(headline, indent=2))
    print("GATES:", result["gates"])
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
