from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import discover_layout


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
    box = getattr(metrics, "box", None)
    if box is not None and getattr(box, "maps", None) is not None:
        maps = list(box.maps)
        results["map50_95_by_class"] = {
            layout.names[index]: float(maps[index])
            for index in sorted(layout.names)
            if index < len(maps)
        }
        if results["map50_95_by_class"]:
            results["worst_class_map50_95"] = min(results["map50_95_by_class"].values())
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
