from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROUTE_A = {"macro": 0.005, "bottom3": -0.005, "worst": -0.005}
ROUTE_B = {"macro": -0.002, "bottom3": 0.010, "worst": 0.010}


def _load_eval(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "precision" not in payload:
        raise RuntimeError(f"Bukan COCOeval eval dictionary: {source}")
    return payload


def _load_categories(annotation_json: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(annotation_json).expanduser().resolve().read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    if not categories:
        raise RuntimeError("COCO annotation tidak memiliki categories")
    return categories


def _class_ap(eval_payload: dict[str, Any], categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # COCOeval precision shape: [IoU, Recall, Category, Area, MaxDet].
    precision = np.asarray(eval_payload["precision"])
    if precision.ndim != 5:
        raise RuntimeError(f"Unexpected precision shape: {precision.shape}")
    if precision.shape[2] != len(categories):
        raise RuntimeError(
            f"Category count mismatch: eval K={precision.shape[2]} vs annotations={len(categories)}"
        )

    rows: list[dict[str, Any]] = []
    for index, category in enumerate(categories):
        values = precision[:, :, index, 0, -1]
        valid = values[values > -1]
        ap = float(valid.mean()) if valid.size else float("nan")
        rows.append(
            {
                "index": index,
                "category_id": int(category["id"]),
                "name": str(category["name"]),
                "ap50_95": ap,
            }
        )
    return rows


def summarize(eval_path: str | Path, annotation_json: str | Path) -> dict[str, Any]:
    payload = _load_eval(eval_path)
    categories = _load_categories(annotation_json)
    rows = _class_ap(payload, categories)
    finite = [row["ap50_95"] for row in rows if np.isfinite(row["ap50_95"])]
    if len(finite) != len(rows):
        raise RuntimeError("Ada kelas validation tanpa AP terdefinisi; screen dihentikan")
    ordered = sorted(rows, key=lambda row: row["ap50_95"])
    macro = float(np.mean(finite))
    bottom3 = float(np.mean([row["ap50_95"] for row in ordered[:3]]))
    worst = float(ordered[0]["ap50_95"])

    # Global COCO AP uses all categories jointly. It is not mathematically required
    # to equal the arithmetic Macro class AP, so keep both.
    precision = np.asarray(payload["precision"])
    valid = precision[:, :, :, 0, -1]
    valid = valid[valid > -1]
    global_coco_ap = float(valid.mean()) if valid.size else float("nan")
    return {
        "eval_path": str(Path(eval_path).expanduser().resolve()),
        "annotation_json": str(Path(annotation_json).expanduser().resolve()),
        "global_coco_ap_from_precision": global_coco_ap,
        "macro_class_ap50_95": macro,
        "bottom3_class_ap50_95": bottom3,
        "worst_class_ap50_95": worst,
        "bottom3_classes": ordered[:3],
        "per_class": rows,
    }


def compare(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        "macro": candidate["macro_class_ap50_95"] - control["macro_class_ap50_95"],
        "bottom3": candidate["bottom3_class_ap50_95"] - control["bottom3_class_ap50_95"],
        "worst": candidate["worst_class_ap50_95"] - control["worst_class_ap50_95"],
        "global_coco_ap": (
            candidate["global_coco_ap_from_precision"] - control["global_coco_ap_from_precision"]
        ),
    }
    route_a = (
        deltas["macro"] >= ROUTE_A["macro"]
        and deltas["bottom3"] >= ROUTE_A["bottom3"]
        and deltas["worst"] >= ROUTE_A["worst"]
    )
    route_b = (
        deltas["macro"] >= ROUTE_B["macro"]
        and deltas["bottom3"] >= ROUTE_B["bottom3"]
        and deltas["worst"] >= ROUTE_B["worst"]
    )

    by_name_control = {row["name"]: row for row in control["per_class"]}
    by_name_candidate = {row["name"]: row for row in candidate["per_class"]}
    if set(by_name_control) != set(by_name_candidate):
        raise RuntimeError("Class names differ between paired arms")
    paired_classes = []
    for name in by_name_control:
        c0 = by_name_control[name]["ap50_95"]
        c1 = by_name_candidate[name]["ap50_95"]
        paired_classes.append(
            {"name": name, "DFN0": c0, "DFN_AF2": c1, "delta": c1 - c0}
        )
    paired_classes.sort(key=lambda row: row["delta"], reverse=True)

    return {
        "format": "coffee_detector.af2_dfine_transfer.seed42_summary.v1",
        "protocol": "faruq-v3-af2-dfine-n-transfer-seed42-v1",
        "seed": 42,
        "test_images_accessed": False,
        "control": control,
        "candidate": candidate,
        "deltas_af2_minus_native": deltas,
        "paired_per_class": paired_classes,
        "screen": {
            "route_a_overall_gain": route_a,
            "route_b_lower_tail_pareto": route_b,
            "decision": "PROMOTE_TO_3_SEED" if (route_a or route_b) else "STOP",
            "thresholds": {"route_a": ROUTE_A, "route_b": ROUTE_B},
            "localization_gate": "NOT_DEFINED_FOR_DFINE_SCREEN",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze paired D-FINE-N vs AF2+D-FINE-N COCOeval outputs")
    p.add_argument("--control-eval", required=True)
    p.add_argument("--candidate-eval", required=True)
    p.add_argument("--val-annotations", required=True)
    p.add_argument("--output", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    control = summarize(args.control_eval, args.val_annotations)
    candidate = summarize(args.candidate_eval, args.val_annotations)
    result = compare(control, candidate)
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
