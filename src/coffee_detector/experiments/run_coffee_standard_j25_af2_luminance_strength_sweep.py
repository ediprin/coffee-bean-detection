"""Validation-only inference-strength diagnostic for the frozen AF2LUMSAFE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.dataset import discover_layout
from coffee_detector.evaluate import _classwise_summary
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    METRICS,
    _json,
    validate_j25_development,
)
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    _sha256,
)
from coffee_detector.experiments.run_coffee_standard_j25_af2_luminance_safe import (
    ARM,
    PROTOCOL as CANDIDATE_PROTOCOL,
)


PROTOCOL = "coffee-standard-j25-af2-luminance-strength-sweep-seed42-v1"
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGET_CLASS = "Biji Hitam Pecah"


def _strength_code(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _dominates(left: dict, right: dict) -> bool:
    pairs = [(float(left[key]), float(right[key])) for key in METRICS]
    return all(a >= b for a, b in pairs) and any(a > b for a, b in pairs)


def summarize_sweep(rows: dict[str, dict], endpoint: dict) -> dict:
    expected = {_strength_code(value) for value in STRENGTHS}
    if set(rows) != expected:
        raise RuntimeError(f"Strength rows tidak lengkap: {sorted(rows)}")
    endpoint_row = rows[_strength_code(1.0)]
    endpoint_deltas = {
        metric: float(endpoint_row[metric]) - float(endpoint[metric]) for metric in METRICS
    }
    endpoint_exact = all(abs(value) <= 1e-6 for value in endpoint_deltas.values())
    if not endpoint_exact:
        raise RuntimeError(f"Endpoint lambda=1 tidak mereproduksi hasil historis: {endpoint_deltas}")
    frontier = []
    for code, row in rows.items():
        if not any(
            other != code and _dominates(other_row, row)
            for other, other_row in rows.items()
        ):
            frontier.append(code)
    winners = {
        metric: max(rows, key=lambda code: float(rows[code][metric])) for metric in METRICS
    }
    target_winner = max(
        rows, key=lambda code: float(rows[code]["map50_95_by_class"][TARGET_CLASS])
    )
    rescued = [
        code
        for code, row in rows.items()
        if code != _strength_code(1.0)
        and float(row["map50_95_by_class"][TARGET_CLASS]) > 0.0
    ]
    return {
        "endpoint_deltas": endpoint_deltas,
        "endpoint_exact": endpoint_exact,
        "pareto_frontier": sorted(frontier, key=lambda code: float(code.replace("p", "."))),
        "metric_winners": winners,
        "target_class": TARGET_CLASS,
        "target_class_winner": target_winner,
        "target_class_rescued_strengths": rescued,
        "interpretation": (
            "INFERENCE_STRENGTH_CAN_RESCUE_TARGET"
            if rescued
            else "TARGET_FAILURE_NOT_RESCUED_BY_STRENGTH"
        ),
    }


def _evaluate_strength(checkpoint: Path, root: Path, strength: float, device: str) -> dict:
    from ultralytics import YOLO

    layout = discover_layout(root)
    model = YOLO(str(checkpoint))
    detector = model.model
    enhancer = getattr(detector, "af2_luminance_safe", None)
    if enhancer is None or not hasattr(enhancer, "set_inference_strength"):
        raise RuntimeError("Checkpoint bukan AF2LUMSAFE yang mendukung strength diagnostic")
    if any(parameter.requires_grad for parameter in enhancer.parameters()):
        raise RuntimeError("Frontend diagnostic tidak lagi parameter-free")
    enhancer.set_inference_strength(strength)
    detector.eval()
    metrics = model.val(
        data=str(layout.yaml_path),
        split="val",
        plots=False,
        verbose=False,
        device=device,
    )
    results = {key: float(value) for key, value in metrics.results_dict.items()}
    results.update(_classwise_summary(metrics.box, layout.names))
    if results.get("classes_without_ground_truth"):
        raise RuntimeError("Validation kehilangan kelas")
    return results


def run_strength_sweep(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
    checkpoint: str | Path,
    candidate_result: str | Path,
    output_root: str | Path,
    *,
    device: str = "0",
    authorize_diagnostic: bool = False,
) -> dict:
    if not authorize_diagnostic:
        raise RuntimeError("Validation diagnostic harus diotorisasi eksplisit")
    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("Strength sweep hanya untuk J25 train-siblings v2")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    candidate = _json(candidate_result, "AF2LUMSAFE result")
    if candidate.get("protocol") != CANDIDATE_PROTOCOL or candidate.get("arm") != ARM:
        raise RuntimeError("Candidate result bukan AF2LUMSAFE seed-42")
    if candidate.get("seed") != 42 or candidate.get("test_images_accessed") is not False:
        raise RuntimeError("Candidate seed/test contract salah")
    if candidate.get("checkpoint_sha256") != _sha256(checkpoint):
        raise RuntimeError("Checkpoint SHA berbeda dari candidate result")

    destination = Path(output_root).expanduser().resolve()
    report_root = destination / "strength_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    for strength in STRENGTHS:
        code = _strength_code(strength)
        path = report_root / f"lambda_{code}.json"
        if path.is_file():
            cached = _json(path, f"lambda={strength}")
            if (
                cached.get("checkpoint_sha256") != _sha256(checkpoint)
                or cached.get("strength") != strength
                or cached.get("test_images_accessed") is not False
            ):
                raise RuntimeError(f"Cached report lambda={strength} berbeda kontrak")
            metrics = cached["metrics"]
        else:
            metrics = _evaluate_strength(checkpoint, root, strength, device)
            path.write_text(
                json.dumps(
                    {
                        "format": "coffee_detector.j25.af2_luminance_strength.report.v1",
                        "protocol": PROTOCOL,
                        "strength": strength,
                        "checkpoint_sha256": _sha256(checkpoint),
                        "metrics": metrics,
                        "training_executed": False,
                        "evaluation_split": "val",
                        "test_images_accessed": False,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        rows[code] = metrics
        print(
            f"lambda={strength:.2f}: "
            + ", ".join(f"{metric}={metrics[metric]:.6f}" for metric in METRICS)
            + f", {TARGET_CLASS}={metrics['map50_95_by_class'][TARGET_CLASS]:.6f}",
            flush=True,
        )

    diagnosis = summarize_sweep(rows, candidate["metrics"])
    payload = {
        "format": "coffee_detector.j25.af2_luminance_strength.sweep.v1",
        "protocol": PROTOCOL,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "strengths": list(STRENGTHS),
        "values": {
            code: {
                **{metric: float(row[metric]) for metric in METRICS},
                "target_class_map50_95": float(row["map50_95_by_class"][TARGET_CLASS]),
            }
            for code, row in rows.items()
        },
        **diagnosis,
        "claim_boundary": "post-screen validation-only diagnostic; no tuned test claim",
        "training_executed": False,
        "test_opened": False,
    }
    summary = destination / "af2_luminance_strength_sweep.json"
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--development-contract", required=True)
    parser.add_argument("--provenance-summary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-diagnostic", action="store_true")
    args = parser.parse_args()
    run_strength_sweep(
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.checkpoint,
        args.candidate_result,
        args.output_root,
        device=args.device,
        authorize_diagnostic=args.authorize_diagnostic,
    )


if __name__ == "__main__":
    main()
