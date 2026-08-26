from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from coffee_detector.dataset import discover_layout


MINIMUM_CLASS_GAIN = 0.02
MINIMUM_SELECTED_CLASSES = 2
MAXIMUM_SELECTED_CLASSES = 10


def _load_report(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} tidak ditemukan: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("split") != "train":
        raise RuntimeError(f"{label} wajib berasal dari split train")
    return payload


def _class_ap(report: Mapping[str, Any], names: Mapping[int, str]) -> dict[str, float]:
    metrics = report.get("metrics", {})
    values = metrics.get("map50_95_by_class", {})
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError("Train report kehilangan kelas")
    missing = sorted(set(names.values()) - set(values))
    if missing:
        raise RuntimeError(f"Train report tidak lengkap: {missing}")
    return {name: float(values[name]) for name in names.values()}


def build_af2_dlrbc_complementarity_audit(
    data_root: str | Path,
    af2_train_report: str | Path,
    dlrbc_train_report: str | Path,
    output: str | Path,
    *,
    minimum_class_gain: float = MINIMUM_CLASS_GAIN,
    maximum_selected_classes: int = MAXIMUM_SELECTED_CLASSES,
) -> dict[str, Any]:
    """Freeze class selection using train-only AP complementarity.

    Validation and test are intentionally absent from this API. A class is
    eligible only when the independently trained DLRBC arm is at least two AP
    points better than AF2 on train. The cap prevents the residual from
    degenerating into another global DLRBC head.
    """

    layout = discover_layout(data_root)
    if "test" in layout.splits:
        raise RuntimeError("Development root tidak boleh mengekspos test")
    af2_report = _load_report(af2_train_report, "AF2 train report")
    dlrbc_report = _load_report(dlrbc_train_report, "DLRBC train report")
    af2_values = _class_ap(af2_report, layout.names)
    dlrbc_values = _class_ap(dlrbc_report, layout.names)
    rows = []
    for class_id, class_name in sorted(layout.names.items()):
        delta = dlrbc_values[class_name] - af2_values[class_name]
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "af2_train_map50_95": af2_values[class_name],
                "dlrbc_train_map50_95": dlrbc_values[class_name],
                "delta": delta,
                "eligible": delta >= float(minimum_class_gain),
            }
        )
    eligible = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: (-row["delta"], row["class_id"]),
    )[: int(maximum_selected_classes)]
    selected_ids = sorted(int(row["class_id"]) for row in eligible)
    gates = {
        "selection_uses_train_only": True,
        "all_21_classes_observed": len(rows) == 21,
        "minimum_selected_classes": len(selected_ids) >= MINIMUM_SELECTED_CLASSES,
        "maximum_selected_classes": len(selected_ids) <= int(maximum_selected_classes),
        "minimum_gain_satisfied": bool(eligible)
        and min(float(row["delta"]) for row in eligible) >= float(minimum_class_gain),
        "test_not_accessed": True,
    }
    decision = "AUTHORIZE_AF2CSD1" if all(gates.values()) else "STOP_NO_COMPLEMENTARITY"
    payload = {
        "format": "coffee_detector.af2_selective_dlrbc.complementarity.v1",
        "protocol": "faruq-v3-af2-class-selective-dlrbc-seed42-v1",
        "selection_split": "train",
        "minimum_class_gain": float(minimum_class_gain),
        "maximum_selected_classes": int(maximum_selected_classes),
        "selected_class_ids": selected_ids,
        "selected_class_names": [layout.names[value] for value in selected_ids],
        "classes": rows,
        "gates": gates,
        "decision": decision,
        "training_authorized": decision == "AUTHORIZE_AF2CSD1",
        "validation_accessed": False,
        "test_images_accessed": False,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only AF2/DLRBC class complementarity audit")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--af2-train-report", required=True)
    parser.add_argument("--dlrbc-train-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_af2_dlrbc_complementarity_audit(
        args.data_root, args.af2_train_report, args.dlrbc_train_report, args.output
    )
    print(json.dumps({key: result[key] for key in ("selected_class_ids", "selected_class_names", "gates", "decision")}, indent=2))


if __name__ == "__main__":
    main()
