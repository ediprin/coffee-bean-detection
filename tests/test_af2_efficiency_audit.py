import json
from pathlib import Path

import pytest

from coffee_detector.experiments.run_faruq_v3_af2_efficiency_audit import (
    build_af2_efficiency_summary,
)


def _model(model: str, seed: int, latency: float, parameters: int = 100) -> dict:
    return {
        "format": "coffee_detector.af2.efficiency_model.v1",
        "model": model,
        "seed": seed,
        "checkpoint_sha256": f"{model}-{seed}",
        "checkpoint_file_bytes": 1000 + seed,
        "parameter_count": parameters,
        "trainable_parameter_count": parameters,
        "parameter_bytes": parameters * 4,
        "buffer_bytes": 40,
        "state_tensor_bytes": parameters * 4 + 40,
        "latency_mean_ms": latency,
        "latency_median_ms": latency - 1.0,
        "latency_p95_ms": latency + 2.0,
        "throughput_images_per_second": 1000.0 / latency,
        "allocator_before_bytes": 0,
        "steady_allocated_bytes": 1000,
        "steady_reserved_bytes": 1200,
        "model_sample_resident_bytes": 1000,
        "peak_allocated_bytes": 1100,
        "peak_reserved_bytes": 1200,
        "incremental_inference_peak_bytes": 100,
        "training_executed": False,
        "test_images_accessed": False,
    }


def _pair(path: Path, seed: int, *, af2_parameters: int = 100) -> Path:
    environment = {
        "device": "cuda:0",
        "gpu_name": "example",
        "gpu_capability": [7, 5],
        "torch_version": "test",
        "cuda_version": "test",
        "ultralytics_version": "test",
    }
    payload = {
        "format": "coffee_detector.af2.efficiency_pair.v1",
        "seed": seed,
        "measurement_order": ["D0FT", "AF2"],
        "environment_before": environment,
        "environment_after": environment,
        "models": {
            "D0FT": _model("D0FT", seed, 10.0),
            "AF2": _model("AF2", seed, 20.0, af2_parameters),
        },
        "training_executed": False,
        "test_images_accessed": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_summary_reports_paired_efficiency_and_parameter_equality(
    tmp_path: Path,
) -> None:
    reports = [_pair(tmp_path / f"pair-{seed}.json", seed) for seed in (42, 123, 2026)]

    result = build_af2_efficiency_summary(reports, tmp_path / "summary.json")

    latency = result["aggregate"]["latency_mean_ms"]
    assert latency["d0ft_mean"] == pytest.approx(10.0)
    assert latency["af2_mean"] == pytest.approx(20.0)
    assert latency["ratio_mean"] == pytest.approx(2.0)
    assert result["parameter_free_frontend_supported"]
    assert result["gates"]["paired_environment_unchanged"]
    assert result["training_executed"] is False
    assert result["test_images_accessed"] is False


def test_parameter_mismatch_invalidates_audit(tmp_path: Path) -> None:
    reports = [
        _pair(
            tmp_path / f"pair-{seed}.json",
            seed,
            af2_parameters=101 if seed == 123 else 100,
        )
        for seed in (42, 123, 2026)
    ]

    with pytest.raises(RuntimeError, match="tidak valid"):
        build_af2_efficiency_summary(reports, tmp_path / "summary.json")


def test_test_access_is_rejected(tmp_path: Path) -> None:
    reports = [_pair(tmp_path / f"pair-{seed}.json", seed) for seed in (42, 123, 2026)]
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    payload["test_images_accessed"] = True
    reports[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="membuka test"):
        build_af2_efficiency_summary(reports, tmp_path / "summary.json")


def test_requires_exactly_three_pair_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tiga pair report"):
        build_af2_efficiency_summary([], tmp_path / "summary.json")


def test_protocol_and_notebook_freeze_no_training_same_device_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "docs/FARUQ_V3_AF2_EFFICIENCY_AUDIT_PROTOCOL_2026-08-21.md").read_text(
        encoding="utf-8"
    )
    notebook = (root / "notebooks/Faruq_V3_AF2_Efficiency_Audit_Colab.ipynb").read_text(
        encoding="utf-8"
    )

    assert "frozen before measurement" in protocol
    assert "Warmup: 30" in protocol
    assert "Measurement: 100" in protocol
    assert "Standard YOLO FLOPs are deliberately omitted" in protocol
    assert "run_faruq_v3_af2_efficiency_audit" in notebook
    assert "D0FT_seed42/weights/best.pt" in notebook
    assert "AF2_seed2026/weights/best.pt" in notebook
    assert "pair_reports" in notebook
    assert "--authorize-training" not in notebook
    assert "test" not in " ".join(
        line.strip() for line in notebook.splitlines() if "require_project_artifact" in line
    ).lower()
