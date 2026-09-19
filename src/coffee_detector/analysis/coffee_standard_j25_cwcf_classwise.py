"""Validation-only classwise attribution for the J25 CWCF1 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coffee_detector.data.prepare_coffee_standard_primary import J25_CLASSES
from coffee_detector.j25_cwcf.model import ATTRIBUTE_NAMES, build_j25_attribute_matrix


MODELS = ("D0DIRECT", "SAFEAUG0", "CWCF1")
TARGET = "Biji Hitam Pecah"


def _load_report(path: str | Path, label: str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val":
        raise RuntimeError(f"{label} bukan validation report")
    metrics = payload.get("metrics", {})
    if metrics.get("classes_without_ground_truth"):
        raise RuntimeError(f"{label} kehilangan kelas")
    by_class = metrics.get("map50_95_by_class", {})
    if tuple(by_class) != J25_CLASSES:
        if set(by_class) != set(J25_CLASSES):
            raise RuntimeError(f"{label} tidak memiliki tepat ontologi J25")
    return payload


def run_cwcf_classwise_audit(
    d0_report: str | Path,
    safeaug_report: str | Path,
    cwcf_report: str | Path,
    output: str | Path,
) -> dict:
    payloads = {
        "D0DIRECT": _load_report(d0_report, "D0DIRECT"),
        "SAFEAUG0": _load_report(safeaug_report, "SAFEAUG0"),
        "CWCF1": _load_report(cwcf_report, "CWCF1"),
    }
    values = {
        model: {
            name: float(payloads[model]["metrics"]["map50_95_by_class"][name])
            for name in J25_CLASSES
        }
        for model in MODELS
    }
    classwise = []
    for name in J25_CLASSES:
        row = {"class_name": name, **{model: values[model][name] for model in MODELS}}
        row["cwcf_minus_d0"] = row["CWCF1"] - row["D0DIRECT"]
        row["cwcf_minus_safeaug"] = row["CWCF1"] - row["SAFEAUG0"]
        classwise.append(row)

    bottom3 = {
        model: [
            name
            for name, _ in sorted(values[model].items(), key=lambda item: (item[1], item[0]))[:3]
        ]
        for model in MODELS
    }
    matrix = build_j25_attribute_matrix().numpy()
    attribute_groups = []
    for attribute_index, attribute in enumerate(ATTRIBUTE_NAMES):
        indices = np.flatnonzero(matrix[:, attribute_index] > 0.5).tolist()
        names = [J25_CLASSES[index] for index in indices]
        row = {"attribute": attribute, "classes": names}
        for model in MODELS:
            row[model] = float(np.mean([values[model][name] for name in names]))
        row["cwcf_minus_d0"] = row["CWCF1"] - row["D0DIRECT"]
        row["cwcf_minus_safeaug"] = row["CWCF1"] - row["SAFEAUG0"]
        attribute_groups.append(row)

    black_only = ("Biji Hitam Penuh", "Biji Hitam Sebagian")
    broken_only = ("Biji Pecah",)
    conjunction = {}
    for reference in ("D0DIRECT", "SAFEAUG0"):
        target_delta = values["CWCF1"][TARGET] - values[reference][TARGET]
        primitive_names = black_only + broken_only
        primitive_delta = float(
            np.mean(
                [values["CWCF1"][name] - values[reference][name] for name in primitive_names]
            )
        )
        conjunction[reference] = {
            "target_delta": target_delta,
            "primitive_mean_delta": primitive_delta,
            "target_minus_primitive_delta": target_delta - primitive_delta,
        }

    gains = sorted(classwise, key=lambda row: row["cwcf_minus_safeaug"], reverse=True)
    result = {
        "format": "coffee_detector.coffee_standard_j25.cwcf.classwise.v1",
        "scope": "validation-only post-screen diagnostic",
        "classwise": classwise,
        "bottom3_classes": bottom3,
        "largest_gains_vs_safeaug": gains[:5],
        "largest_drops_vs_safeaug": list(reversed(gains[-5:])),
        "attribute_groups": attribute_groups,
        "black_broken_conjunction": conjunction,
        "target_class": TARGET,
        "target_values": {model: values[model][TARGET] for model in MODELS},
        "training_executed": False,
        "test_opened": False,
        "claim_boundary": (
            "descriptive validation attribution; attribute groups overlap and do not prove causality"
        ),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0-report", required=True)
    parser.add_argument("--safeaug-report", required=True)
    parser.add_argument("--cwcf-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_cwcf_classwise_audit(
        args.d0_report, args.safeaug_report, args.cwcf_report, args.output
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
