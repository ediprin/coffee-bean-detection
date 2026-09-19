"""Validation-only root-cause audit for J25 ``Biji Hitam Pecah``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coffee_detector.analysis.coffee_fg_diagnostics import diagnose_checkpoint
from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout, parse_label
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    _json,
    validate_j25_development,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import _sha256


PROTOCOL = "coffee-standard-j25-black-broken-root-cause-v1"
TARGET_CLASS = "Biji Hitam Pecah"
D0_PROTOCOL = "coffee-standard-j25-train-siblings-af2-direct-seed42-v2"
SAFE_PROTOCOL = "coffee-standard-j25-af2-luminance-safe-seed42-v1"


def _target_support(root: Path) -> dict:
    layout = discover_layout(root)
    target_ids = [class_id for class_id, name in layout.names.items() if name == TARGET_CLASS]
    if len(target_ids) != 1:
        raise RuntimeError(f"Target class ambigu/tidak ada: {target_ids}")
    target_id = target_ids[0]
    image_root, label_root = layout.splits["val"]
    images = 0
    instances = 0
    for image in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        annotations = parse_label(
            (label_root / image.relative_to(image_root)).with_suffix(".txt"),
            set(layout.names),
        )
        count = sum(item.class_id == target_id for item in annotations)
        images += int(count > 0)
        instances += count
    return {
        "class_id": target_id,
        "class_name": TARGET_CLASS,
        "validation_images": images,
        "validation_instances": instances,
    }


def _target_view(diagnostic: dict) -> dict:
    raw_branch = diagnostic["branches"]["one2one"]["500"]
    final_branch = diagnostic["final_detections"]
    raw = raw_branch["per_class"][TARGET_CLASS]
    final = final_branch["per_class"][TARGET_CLASS]

    def wrong_destinations(branch: dict) -> dict:
        return {
            name: int(count)
            for name, count in branch.get("confusion", {}).get(TARGET_CLASS, {}).items()
            if name != TARGET_CLASS
        }

    return {
        "raw_top500": {
            **raw,
            "wrong_class": int(raw["matched"] - raw["correct_class"]),
            "wrong_destinations": wrong_destinations(raw_branch),
        },
        "final_conf0001": {
            **final,
            "wrong_class": int(final["matched"] - final["correct_class"]),
            "wrong_destinations": wrong_destinations(final_branch),
        },
    }


def attribute_target(view: dict) -> str:
    raw = view["raw_top500"]
    final = view["final_conf0001"]
    if raw["accessible"] == 0:
        return "NO_RAW_LOCALIZATION"
    if raw["matched"] > 0 and raw["correct_class"] == 0:
        return "RAW_LOCALIZED_BUT_WRONG_CLASS"
    if raw["matched"] > final["matched"]:
        return "FINAL_SELECTION_OR_RANKING_LOSS"
    if final["matched"] > 0 and final["correct_class"] == 0:
        return "FINAL_LOCALIZED_BUT_WRONG_CLASS"
    if final["correct_class"] > 0:
        return "CORRECT_DETECTIONS_EXIST_BUT_AP_RANKING_FAILS"
    return "MIXED_OR_SPARSE_SUPPORT"


def run_black_broken_audit(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
    d0_checkpoint: str | Path,
    d0_result: str | Path,
    safe_checkpoint: str | Path,
    safe_result: str | Path,
    output: str | Path,
    *,
    device: str = "0",
    authorize_diagnostic: bool = False,
) -> dict:
    if not authorize_diagnostic:
        raise RuntimeError("Audit harus diotorisasi eksplisit")
    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(root, development_contract, provenance_summary)
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("Audit hanya untuk J25 train-siblings v2")
    checkpoints = {
        "D0DIRECT": Path(d0_checkpoint).expanduser().resolve(),
        "AF2LUMSAFE_lambda0": Path(safe_checkpoint).expanduser().resolve(),
    }
    results = {
        "D0DIRECT": _json(d0_result, "D0 result"),
        "AF2LUMSAFE_lambda0": _json(safe_result, "SAFE result"),
    }
    contracts = {
        "D0DIRECT": D0_PROTOCOL,
        "AF2LUMSAFE_lambda0": SAFE_PROTOCOL,
    }
    for name in checkpoints:
        if not checkpoints[name].is_file():
            raise FileNotFoundError(checkpoints[name])
        result = results[name]
        if result.get("protocol") != contracts[name] or result.get("test_images_accessed") is not False:
            raise RuntimeError(f"Result contract salah: {name}")
        if result.get("checkpoint_sha256") != _sha256(checkpoints[name]):
            raise RuntimeError(f"Checkpoint SHA berbeda: {name}")

    support = _target_support(root)
    diagnostics = {}
    views = {}
    for name, checkpoint in checkpoints.items():
        diagnostic = diagnose_checkpoint(
            checkpoint,
            root,
            split="val",
            image_size=640,
            candidate_counts=(500,),
            iou_threshold=0.5,
            confidence_threshold=0.001,
            nms_iou=0.7,
            max_det=500,
            device=device,
            inference_strength=0.0 if name == "AF2LUMSAFE_lambda0" else None,
        )
        diagnostics[name] = diagnostic
        views[name] = _target_view(diagnostic)
        print(name, json.dumps(views[name], ensure_ascii=False), flush=True)

    attribution = {name: attribute_target(view) for name, view in views.items()}
    payload = {
        "format": "coffee_detector.j25.black_broken_root_cause.v1",
        "protocol": PROTOCOL,
        "target_support": support,
        "models": views,
        "attribution": attribution,
        "diagnostic_settings": {
            "raw_candidates": 500,
            "iou_threshold": 0.5,
            "final_confidence": 0.001,
            "safe_inference_strength": 0.0,
        },
        "training_executed": False,
        "test_opened": False,
    }
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raw_output = output.with_name(output.stem + "_full.json")
    raw_output.write_text(
        json.dumps(
            {"summary": payload, "full_diagnostics": diagnostics},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--development-contract", required=True)
    parser.add_argument("--provenance-summary", required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--d0-result", required=True)
    parser.add_argument("--safe-checkpoint", required=True)
    parser.add_argument("--safe-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-diagnostic", action="store_true")
    args = parser.parse_args()
    run_black_broken_audit(
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.d0_checkpoint,
        args.d0_result,
        args.safe_checkpoint,
        args.safe_result,
        args.output,
        device=args.device,
        authorize_diagnostic=args.authorize_diagnostic,
    )


if __name__ == "__main__":
    main()
