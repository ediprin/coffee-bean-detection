from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import discover_layout


def _classwise_summary(metric_object, names: dict[int, str]) -> dict:
    """Summarize only classes that actually have evaluation targets.

    Ultralytics ``Metric.maps`` fills classes without targets with the overall
    mAP. Using it directly would make an absent rare class look average rather
    than exposing the missing coverage.
    """

    class_indices = [
        int(value) for value in np.asarray(metric_object.ap_class_index).reshape(-1)
    ]
    average_precision = np.asarray(metric_object.ap, dtype=np.float64)
    if average_precision.ndim == 1:
        average_precision = average_precision[:, None]
    by_class = {
        names[class_id]: float(average_precision[row].mean())
        for row, class_id in enumerate(class_indices)
        if class_id in names and row < len(average_precision)
    }
    missing = [names[class_id] for class_id in sorted(set(names) - set(class_indices))]
    if not by_class:
        return {
            "map50_95_by_class": {},
            "classes_without_ground_truth": missing,
        }
    values = np.asarray(list(by_class.values()), dtype=np.float64)
    return {
        "map50_95_by_class": by_class,
        "classes_without_ground_truth": missing,
        "macro_map50_95": float(values.mean()),
        "worst_class_map50_95": float(values.min()),
        "bottom3_class_map50_95": float(
            np.sort(values)[: min(3, len(values))].mean()
        ),
    }


def evaluate(
    checkpoint: str | Path,
    data_root: str | Path,
    output: str | Path,
    split: str = "test",
    device: str | None = None,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Ultralytics belum terpasang. Jalankan `pip install -e .`.") from error

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint}")
    layout = discover_layout(data_root)
    kwargs = {"data": str(layout.yaml_path), "split": split, "plots": True, "verbose": True}
    if device is not None:
        kwargs["device"] = device
    metrics = YOLO(str(checkpoint)).val(**kwargs)
    results = {key: float(value) for key, value in metrics.results_dict.items()}
    for metric_name, metric_object in (
        ("box", getattr(metrics, "box", None)),
        ("mask", getattr(metrics, "seg", None)),
    ):
        if metric_object is None or getattr(metric_object, "ap", None) is None:
            continue
        prefix = "" if metric_name == "box" else "mask_"
        for key, value in _classwise_summary(metric_object, layout.names).items():
            results[f"{prefix}{key}"] = value
    payload = {
        "checkpoint": str(checkpoint),
        "data": str(layout.root),
        "split": split,
        "metrics": results,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi checkpoint detector pada split terkunci.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device")
    args = parser.parse_args()
    payload = evaluate(args.checkpoint, args.data_root, args.output, args.split, args.device)
    print(json.dumps(payload["metrics"], indent=2, ensure_ascii=False))
    print(f"SAVED: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
