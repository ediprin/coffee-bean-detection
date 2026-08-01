from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import IMAGE_SUFFIXES, discover_layout, parse_label
from .separate_sni21_sources import SOURCES


def _validation_counts(data_root: Path) -> tuple[Counter[int], Counter[int]]:
    layout = discover_layout(data_root)
    if "test" in layout.splits or (data_root / "test").exists():
        raise RuntimeError(f"Test tidak boleh tersedia: {data_root}")
    image_root, label_root = layout.splits["val"]
    instance_counts: Counter[int] = Counter()
    image_counts: Counter[int] = Counter()
    for image_path in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image_path.relative_to(image_root)
        boxes = parse_label(
            (label_root / relative).with_suffix(".txt"), set(layout.names)
        )
        per_image = Counter(box.class_id for box in boxes)
        instance_counts.update(per_image)
        image_counts.update(per_image.keys())
    return instance_counts, image_counts


def _support_ap_correlation(rows: list[dict]) -> float | None:
    observed = [
        row
        for row in rows
        if row["val_instances"] > 0 and row["map50_95"] is not None
    ]
    if len(observed) < 2:
        return None
    support = np.log1p([row["val_instances"] for row in observed])
    performance = np.asarray([row["map50_95"] for row in observed], dtype=float)
    if np.std(support) == 0 or np.std(performance) == 0:
        return None
    return float(np.corrcoef(support, performance)[0, 1])


def analyze_sni21_source_classes(
    separated_root: str | Path,
    evaluation_summary: str | Path,
    output_root: str | Path,
) -> dict:
    separated_root = Path(separated_root).expanduser().resolve()
    evaluation_summary = Path(evaluation_summary).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    evaluation = json.loads(evaluation_summary.read_text(encoding="utf-8"))
    if evaluation.get("training_executed") is not False:
        raise RuntimeError("Summary evaluasi mencatat training")
    if evaluation.get("test_images_accessed") is not False:
        raise RuntimeError("Summary evaluasi pernah mengakses test")

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    source_summaries = {}
    hard_sets = {}
    for source in SOURCES:
        data_root = separated_root / source
        layout = discover_layout(data_root)
        instances, images = _validation_counts(data_root)
        report_path = Path(evaluation["reports"][source]).expanduser().resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("split") != "val":
            raise RuntimeError(f"Laporan bukan validation: {report_path}")
        if report.get("test_images_accessed") is not False:
            raise RuntimeError(f"Laporan source mengakses test: {report_path}")
        ap_by_class = report["metrics"].get("map50_95_by_class", {})
        source_rows = []
        for class_id, class_name in layout.names.items():
            row = {
                "source_dataset": source,
                "class_id": class_id,
                "class_name": class_name,
                "val_images_with_class": int(images[class_id]),
                "val_instances": int(instances[class_id]),
                "map50_95": (
                    float(ap_by_class[class_name])
                    if class_name in ap_by_class
                    else None
                ),
                "has_ground_truth": instances[class_id] > 0,
            }
            rows.append(row)
            source_rows.append(row)
        observed = sorted(
            (row for row in source_rows if row["map50_95"] is not None),
            key=lambda row: (row["map50_95"], row["val_instances"], row["class_id"]),
        )
        hard = observed[: min(5, len(observed))]
        hard_sets[source] = {row["class_name"] for row in hard}
        source_summaries[source] = {
            "classes_with_ground_truth": sum(
                row["has_ground_truth"] for row in source_rows
            ),
            "classes_without_ground_truth": [
                row["class_name"] for row in source_rows if not row["has_ground_truth"]
            ],
            "zero_ap_classes": [
                row["class_name"]
                for row in source_rows
                if row["map50_95"] is not None and row["map50_95"] <= 0
            ],
            "bottom5": hard,
            "log_support_ap_correlation": _support_ap_correlation(source_rows),
        }

    overlap = sorted(set.intersection(*(hard_sets[source] for source in SOURCES)))
    csv_path = output_root / "source_class_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format": "coffee_detector.sni21_source_class_audit.v1",
        "separated_root": str(separated_root),
        "evaluation_summary": str(evaluation_summary),
        "sources": source_summaries,
        "shared_bottom5_classes": overlap,
        "rows": rows,
        "table": str(csv_path),
        "training_executed": False,
        "test_images_accessed": False,
        "development_only": True,
        "interpretation_rule": (
            "AP tanpa GT tidak diisi nol; kelas tanpa GT dilaporkan sebagai coverage gap. "
            "Korelasi support-AP bersifat deskriptif, bukan bukti kausal."
        ),
    }
    summary_path = output_root / "source_class_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit support dan AP per kelas untuk validation Adrian/Faruq."
    )
    parser.add_argument("--separated-root", required=True)
    parser.add_argument("--evaluation-summary", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = analyze_sni21_source_classes(
        args.separated_root, args.evaluation_summary, args.output_root
    )
    print("\n=== BOTTOM-5 PER SOURCE ===")
    for source, summary in result["sources"].items():
        print(f"\n{source}")
        for row in summary["bottom5"]:
            print(
                f"  {row['class_name']}: AP={row['map50_95']:.2%} | "
                f"n={row['val_instances']} | images={row['val_images_with_class']}"
            )
        print("  Missing GT:", summary["classes_without_ground_truth"])
        print("  Corr log-support/AP:", summary["log_support_ap_correlation"])
    print("\nSHARED BOTTOM-5:", result["shared_bottom5_classes"])
    print("TRAINING:", result["training_executed"])
    print("TEST ACCESSED:", result["test_images_accessed"])
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
