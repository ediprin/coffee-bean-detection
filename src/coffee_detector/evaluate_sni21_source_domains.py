from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .evaluate import evaluate
from .separate_sni21_sources import SOURCES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric(metrics: dict, *keys: str) -> float | None:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return None


def evaluate_sni21_source_domains(
    checkpoint: str | Path,
    separated_root: str | Path,
    output_root: str | Path,
    *,
    device: str | None = "0",
) -> dict:
    checkpoint = Path(checkpoint).expanduser().resolve()
    separated_root = Path(separated_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint A0 tidak ditemukan: {checkpoint}")
    separation_path = separated_root / "source_separation_summary.json"
    separation = json.loads(separation_path.read_text(encoding="utf-8"))
    if separation.get("status") != "complete":
        raise RuntimeError(f"Pemisahan sumber belum complete: {separation_path}")
    if separation.get("test_images_accessed") is not False:
        raise RuntimeError("Pemisahan sumber pernah mengakses test")

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256(checkpoint)
    reports = {}
    rows = []
    separation_rows = {
        row["source_dataset"]: row for row in separation.get("rows", [])
    }
    for index, source in enumerate(SOURCES, 1):
        data_root = separated_root / source
        if (data_root / "test").exists():
            raise RuntimeError(f"Test tidak boleh tersedia pada {source}: {data_root}")
        report_path = output_root / f"{source}_val.json"
        cached = None
        if report_path.is_file():
            candidate = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                candidate.get("checkpoint_sha256") == checkpoint_hash
                and Path(candidate.get("data", "")).resolve() == data_root
                and candidate.get("split") == "val"
                and candidate.get("complete") is True
            ):
                cached = candidate
        if cached is None:
            print(f"[{index}/{len(SOURCES)}] EVALUATE {source} validation", flush=True)
            payload = evaluate(
                checkpoint, data_root, report_path, split="val", device=device
            )
            payload.update(
                {
                    "format": "coffee_detector.sni21_source_domain_evaluation.v1",
                    "source_dataset": source,
                    "checkpoint_sha256": checkpoint_hash,
                    "training_executed": False,
                    "test_images_accessed": False,
                    "development_only": True,
                    "complete": True,
                }
            )
            report_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            print(f"[{index}/{len(SOURCES)}] REUSE {source} validation", flush=True)
            payload = cached
        reports[source] = str(report_path)
        metrics = payload["metrics"]
        stats = separation_rows.get(source, {})
        val_images = int(stats.get("val_images", 0))
        val_boxes = int(stats.get("val_boxes", 0))
        rows.append(
            {
                "source_dataset": source,
                "val_images": val_images,
                "val_boxes": val_boxes,
                "boxes_per_image": val_boxes / val_images if val_images else None,
                "map50_95": _metric(metrics, "metrics/mAP50-95(B)"),
                "map50": _metric(metrics, "metrics/mAP50(B)"),
                "precision": _metric(metrics, "metrics/precision(B)"),
                "recall": _metric(metrics, "metrics/recall(B)"),
                "macro_map50_95": _metric(metrics, "macro_map50_95"),
                "bottom3_map50_95": _metric(metrics, "bottom3_class_map50_95"),
                "worst_map50_95": _metric(metrics, "worst_class_map50_95"),
                "classes_without_ground_truth": metrics.get(
                    "classes_without_ground_truth", []
                ),
            }
        )

    summary = {
        "format": "coffee_detector.sni21_source_domain_summary.v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "separated_root": str(separated_root),
        "reports": reports,
        "rows": rows,
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "interpretation_rule": (
            "Bandingkan domain secara deskriptif; ukuran/density validation berbeda, "
            "sehingga selisih bukan efek kausal source dataset."
        ),
    }
    summary_path = output_root / "source_domain_evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluasi checkpoint A0 beku pada validation Adrian dan Faruq secara "
            "terpisah tanpa training/test."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--separated-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = evaluate_sni21_source_domains(
        args.checkpoint,
        args.separated_root,
        args.output_root,
        device=args.device,
    )
    print("\n=== FROZEN A0 PER SOURCE ===")
    for row in result["rows"]:
        print(
            f"{row['source_dataset']}: mAP50-95={row['map50_95']:.2%} | "
            f"Macro={row['macro_map50_95']:.2%} | "
            f"Worst={row['worst_map50_95']:.2%} | "
            f"density={row['boxes_per_image']:.2f} box/image"
        )
    print(f"TRAINING: {result['training_executed']}")
    print(f"TEST ACCESSED: {result['test_images_accessed']}")
    print(f"SAVED: {result['summary']}")


if __name__ == "__main__":
    main()
