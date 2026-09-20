"""Validation-only gate ablation for the quarantined J25 CWCF2 checkpoint."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from coffee_detector.data.prepare_coffee_standard_primary import J25_CLASSES
from coffee_detector.evaluate import evaluate
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    METRICS,
    _json,
    validate_j25_development,
)
from coffee_detector.experiments.run_coffee_standard_j25_cwcf import (
    PROTOCOL as CWCF1_PROTOCOL,
)
from coffee_detector.experiments.run_coffee_standard_j25_cwcf2 import (
    PROTOCOL as CWCF2_PROTOCOL,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import _sha256
from coffee_detector.j25_cwcf import ExplicitCompositionDetectHead


TARGET = "Biji Hitam Pecah"
STATUS = "QUARANTINED_CONCURRENT_WRITER_DIAGNOSTIC_ONLY"


def classify_gate_attribution(
    active: dict[str, float],
    zero: dict[str, float],
    cwcf1: dict[str, float],
    effective_gates: list[float],
) -> str:
    gate_effect = {metric: zero[metric] - active[metric] for metric in METRICS}
    if max((abs(value) for value in effective_gates), default=0.0) < 1e-4:
        return "TRAINING_PATH_DOMINANT_GATE_EFFECT_NEGLIGIBLE"
    if all(gate_effect[metric] >= 0.0 for metric in METRICS) and any(
        gate_effect[metric] > 0.0 for metric in METRICS
    ):
        return "INFERENCE_COMPOSITION_HARMFUL"
    if all(zero[metric] < cwcf1[metric] for metric in METRICS):
        return "TRAINING_PATH_DOMINANT_ZERO_GATE_STILL_BELOW_CWCF1"
    return "MIXED_GATE_AND_TRAINING_EFFECT"


def _metrics(payload: dict) -> dict[str, float]:
    return {metric: float(payload["metrics"][metric]) for metric in METRICS}


def run_cwcf2_gate_audit(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
    cwcf2_checkpoint: str | Path,
    quarantine_result: str | Path,
    cwcf1_result: str | Path,
    output: str | Path,
    *,
    device: str = "0",
    authorize_diagnostic: bool = False,
) -> dict:
    if not authorize_diagnostic:
        raise RuntimeError("Audit gate harus diotorisasi eksplisit")
    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("Audit hanya untuk J25 train-siblings v2")

    checkpoint = Path(cwcf2_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    quarantine = _json(quarantine_result, "CWCF2 quarantine result")
    cwcf1 = _json(cwcf1_result, "CWCF1 result")
    if (
        quarantine.get("protocol") != CWCF2_PROTOCOL
        or quarantine.get("status") != STATUS
        or quarantine.get("valid_for_claims") is not False
        or quarantine.get("test_images_accessed") is not False
        or quarantine.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise RuntimeError("Kontrak quarantine CWCF2 tidak valid")
    if (
        cwcf1.get("protocol") != CWCF1_PROTOCOL
        or cwcf1.get("test_images_accessed") is not False
    ):
        raise RuntimeError("Kontrak CWCF1 tidak valid")

    from ultralytics import YOLO

    wrapper = YOLO(str(checkpoint))
    head = wrapper.model.model[-1]
    if not isinstance(head, ExplicitCompositionDetectHead):
        raise TypeError(f"Checkpoint bukan CWCF2 explicit composition: {type(head).__name__}")
    raw_gates = [float(value.detach().cpu()) for value in head.composition_gates]
    effective_gates = [
        float(head.config.composition_gain_max * torch.tanh(value).detach().cpu())
        for value in head.composition_gates
    ]

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    active_report = destination.with_name(destination.stem + "_active_val.json")
    active_eval = evaluate(checkpoint, root, active_report, split="val", device=device)
    active_metrics = _metrics(active_eval)
    quarantine_metrics = _metrics(quarantine)
    endpoint_delta = {
        metric: active_metrics[metric] - quarantine_metrics[metric] for metric in METRICS
    }
    if any(abs(value) > 1e-6 for value in endpoint_delta.values()):
        raise RuntimeError(f"Endpoint aktif tidak mereproduksi quarantine: {endpoint_delta}")

    with tempfile.TemporaryDirectory(prefix="cwcf2-zero-gate-") as temporary:
        zero_checkpoint = Path(temporary) / "cwcf2_zero_gate.pt"
        with torch.no_grad():
            for gate in head.composition_gates:
                gate.zero_()
        wrapper.save(str(zero_checkpoint))
        zero_report = destination.with_name(destination.stem + "_zero_gate_val.json")
        zero_eval = evaluate(zero_checkpoint, root, zero_report, split="val", device=device)
    zero_metrics = _metrics(zero_eval)
    cwcf1_metrics = _metrics(cwcf1)

    active_by_class = {
        name: float(active_eval["metrics"]["map50_95_by_class"][name])
        for name in J25_CLASSES
    }
    zero_by_class = {
        name: float(zero_eval["metrics"]["map50_95_by_class"][name])
        for name in J25_CLASSES
    }
    classwise = [
        {
            "class_name": name,
            "active": active_by_class[name],
            "zero_gate": zero_by_class[name],
            "zero_minus_active": zero_by_class[name] - active_by_class[name],
        }
        for name in J25_CLASSES
    ]
    ordered = sorted(classwise, key=lambda row: row["zero_minus_active"], reverse=True)
    attribution = classify_gate_attribution(
        active_metrics, zero_metrics, cwcf1_metrics, effective_gates
    )
    result = {
        "format": "coffee_detector.coffee_standard_j25.cwcf2.gate_audit.v1",
        "scope": "validation-only quarantine diagnostic",
        "checkpoint_sha256": _sha256(checkpoint),
        "composition_gates": {
            "raw": raw_gates,
            "effective": effective_gates,
            "maximum_absolute_effective": max(abs(value) for value in effective_gates),
        },
        "values": {
            "CWCF1": cwcf1_metrics,
            "CWCF2_ACTIVE": active_metrics,
            "CWCF2_ZERO_GATE": zero_metrics,
        },
        "zero_minus_active": {
            metric: zero_metrics[metric] - active_metrics[metric] for metric in METRICS
        },
        "zero_minus_cwcf1": {
            metric: zero_metrics[metric] - cwcf1_metrics[metric] for metric in METRICS
        },
        "target_class": TARGET,
        "target_values": {
            "CWCF2_ACTIVE": active_by_class[TARGET],
            "CWCF2_ZERO_GATE": zero_by_class[TARGET],
            "zero_minus_active": zero_by_class[TARGET] - active_by_class[TARGET],
        },
        "largest_zero_gate_gains": ordered[:5],
        "largest_zero_gate_drops": list(reversed(ordered[-5:])),
        "classwise": classwise,
        "attribution": attribution,
        "endpoint_reproduced": True,
        "training_executed": False,
        "test_opened": False,
        "valid_for_claims": False,
        "next": "STOP_CWCF2_AND_RETAIN_CWCF1",
    }
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--development-contract", required=True)
    parser.add_argument("--provenance-summary", required=True)
    parser.add_argument("--cwcf2-checkpoint", required=True)
    parser.add_argument("--quarantine-result", required=True)
    parser.add_argument("--cwcf1-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-diagnostic", action="store_true")
    args = parser.parse_args()
    run_cwcf2_gate_audit(
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.cwcf2_checkpoint,
        args.quarantine_result,
        args.cwcf1_result,
        args.output,
        device=args.device,
        authorize_diagnostic=args.authorize_diagnostic,
    )


if __name__ == "__main__":
    main()
