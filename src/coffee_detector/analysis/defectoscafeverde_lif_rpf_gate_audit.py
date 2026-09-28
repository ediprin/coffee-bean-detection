"""Validation-only attribution audit for the DefectosCafeVerde LIF-RPF arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from coffee_detector.dataset import discover_layout
from coffee_detector.evaluate import _classwise_summary


METRICS = (
    "macro_map50_95",
    "bottom3_class_map50_95",
    "worst_class_map50_95",
)


def _read_json(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _headline(payload: Mapping) -> dict[str, float]:
    metrics = payload.get("metrics", payload)
    return {name: float(metrics[name]) for name in METRICS}


def classify_attribution(
    baseline: Mapping[str, float],
    active: Mapping[str, float],
    zero_gate: Mapping[str, float],
    *,
    tolerance: float = 0.001,
) -> str:
    """Separate active inference-gate harm from training-path harm."""

    zero_minus_active = {
        name: float(zero_gate[name]) - float(active[name]) for name in METRICS
    }
    zero_minus_baseline = {
        name: float(zero_gate[name]) - float(baseline[name]) for name in METRICS
    }
    if all(value > tolerance for value in zero_minus_active.values()):
        return "ACTIVE_CUE_HARMS_INFERENCE"
    if all(abs(value) <= tolerance for value in zero_minus_active.values()) and all(
        value < -tolerance for value in zero_minus_baseline.values()
    ):
        return "TRAINING_PATH_DOMINANT_CUE_NEARLY_IGNORED"
    if all(value < -tolerance for value in zero_minus_active.values()):
        return "ACTIVE_CUE_PARTIALLY_COMPENSATES_TRAINING_PATH"
    return "MIXED_GATE_AND_TRAINING_PATH_EFFECT"


def run_audit(
    checkpoint: str | Path,
    data_root: str | Path,
    d0_result: str | Path,
    lif_result: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
) -> dict:
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    baseline_payload = _read_json(d0_result)
    active_payload = _read_json(lif_result)
    if baseline_payload.get("test_images_accessed") is not False:
        raise RuntimeError("Baseline bukan artefak validation-only")
    if active_payload.get("test_images_accessed") is not False:
        raise RuntimeError("LIF-RPF bukan artefak validation-only")

    layout = discover_layout(data_root)
    wrapper = YOLO(str(checkpoint))
    head = wrapper.model.model[-1]
    if type(head).__name__ != "LIFRPFDetectHead":
        raise TypeError(f"Checkpoint bukan LIFRPFDetectHead: {type(head).__name__}")

    raw_gates = [float(adapter.gate.detach().cpu()) for adapter in head.adapters]
    effective_gains = [
        float((adapter.maximum_gain * torch.tanh(adapter.gate)).detach().cpu())
        for adapter in head.adapters
    ]
    with torch.no_grad():
        for adapter in head.adapters:
            adapter.gate.zero_()

    result = wrapper.val(
        data=str(layout.yaml_path),
        split="val",
        device=device,
        plots=False,
        verbose=False,
    )
    zero_metrics = {
        key: float(value) for key, value in result.results_dict.items()
    }
    zero_metrics.update(_classwise_summary(result.box, layout.names))
    if zero_metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")

    baseline = _headline(baseline_payload)
    active = _headline(active_payload)
    zero = {name: float(zero_metrics[name]) for name in METRICS}
    zero_minus_active = {name: zero[name] - active[name] for name in METRICS}
    zero_minus_baseline = {name: zero[name] - baseline[name] for name in METRICS}
    active_classwise = active_payload["map50_95_by_class"]
    baseline_classwise = baseline_payload["map50_95_by_class"]
    classwise = {
        name: {
            "D0DIRECT": float(baseline_classwise[name]),
            "LIFRPF1_ACTIVE": float(active_classwise[name]),
            "LIFRPF1_ZERO_GATE": float(zero_metrics["map50_95_by_class"][name]),
            "active_minus_d0": float(active_classwise[name])
            - float(baseline_classwise[name]),
            "zero_minus_active": float(zero_metrics["map50_95_by_class"][name])
            - float(active_classwise[name]),
        }
        for name in sorted(baseline_classwise)
    }
    payload = {
        "format": "coffee_detector.defectoscafeverde.lif_rpf.gate_audit.v1",
        "evaluation_split": "val",
        "training_executed": False,
        "test_images_accessed": False,
        "raw_gates": raw_gates,
        "effective_gains": effective_gains,
        "values": {
            "D0DIRECT": baseline,
            "LIFRPF1_ACTIVE": active,
            "LIFRPF1_ZERO_GATE": zero,
        },
        "zero_minus_active": zero_minus_active,
        "zero_minus_baseline": zero_minus_baseline,
        "classwise": classwise,
        "attribution": classify_attribution(baseline, active, zero),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--d0-result", required=True)
    parser.add_argument("--lif-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = run_audit(
        args.checkpoint,
        args.data_root,
        args.d0_result,
        args.lif_result,
        args.output,
        device=args.device,
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
