"""Compare finished HAAR and HAAR_MASK2 checkpoints on two shared validation grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import ultralytics
from ultralytics import YOLO

from coffee_detector.analysis.auliya_robusta_preflight import REUSED_OBJECT_CLASSES, _sha256
from coffee_detector.auliya_mask2 import MASK2_ARM, MASK2_PROTOCOL, CommonMaskGridValidator
from coffee_detector.experiments.prepare_auliya_official_fold import PROTOCOL, validate_official_fold
from coffee_detector.experiments.run_auliya_official_frequency import (
    SCHEDULE, ULTRALYTICS_VERSION, WEIGHTS_SHA256, _schedule_for,
)
from coffee_detector.experiments.run_auliya_robusta_frequency import _summarize_masks, _write_json

ARMS = ("HAAR", MASK2_ARM)


def _report(folder: Path, arm: str, development_digest: str) -> tuple[Path, dict]:
    path = folder / "validation_result.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = folder / "weights" / "best.pt"
    contract = report["run_contract"]
    if (report["arm"] != arm or contract["arm"] != arm or contract["fold"] != "A"
            or contract["protocol"] != (MASK2_PROTOCOL if arm == MASK2_ARM else PROTOCOL)
            or contract["development_contract_sha256"] != development_digest
            or contract["pretrained_sha256"] != WEIGHTS_SHA256 or contract["ultralytics"] != ULTRALYTICS_VERSION
            or contract["schedule"] != _schedule_for(arm) or report["evaluation_split"] != "val"
            or report["test_images_accessed"] is not False or report["checkpoint_sha256"] != _sha256(checkpoint)):
        raise ValueError(f"Report or checkpoint contract failed: {path}")
    return checkpoint, report


def _metrics(evaluation) -> dict:
    seg = evaluation.seg
    clean_indices = [i for i, idx in enumerate(seg.ap_class_index) if int(idx) not in REUSED_OBJECT_CLASSES]
    target_idx = next(i for i, idx in enumerate(seg.ap_class_index)
        if evaluation.names[int(idx)] == "biji_hitam_pecah")
    native = _summarize_masks(seg, evaluation.names)
    return {**native,
        "mask_ap90_18": float(np.mean(seg.all_ap[clean_indices, 8])),
        "mask_ap95_18": float(np.mean(seg.all_ap[clean_indices, 9])),
        "biji_hitam_pecah_mask_ap90": float(seg.all_ap[target_idx, 8]),
        "biji_hitam_pecah_mask_ap95": float(seg.all_ap[target_idx, 9]),
        "box_map50_95_21": float(evaluation.box.map),
        "runtime_ms_per_image": {k: float(v) for k, v in evaluation.speed.items()},
    }


def compare(development: Path, haar: Path, mask2: Path, device: str) -> dict:
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected Ultralytics {ULTRALYTICS_VERSION}")
    data = validate_official_fold(development.resolve())
    digest = _sha256(data.parent / "official_fold_contract.json")
    inputs = {arm: _report(folder, arm, digest) for arm, folder in zip(ARMS, (haar, mask2), strict=True)}
    if inputs[MASK2_ARM][1]["run_contract"].get("haar_reference_checkpoint_sha256") != inputs["HAAR"][1]["checkpoint_sha256"]:
        raise ValueError("Treatment was not paired with this exact HAAR checkpoint")
    ignore = {"arm", "protocol", "schedule", "haar_reference_checkpoint_sha256"}
    contracts = [{k:v for k,v in r["run_contract"].items() if k not in ignore} for _,r in inputs.values()]
    if contracts[0] != contracts[1]:
        raise ValueError("Training contracts differ outside the mask-grid treatment")
    grids = {}
    with TemporaryDirectory(prefix="auliya-mask2-compare-") as tmp:
        for stride, batch in ((4, SCHEDULE["batch"]), (2, 4)):
            values = {}
            for arm, (checkpoint, report) in inputs.items():
                print(f"VALIDATION ONLY: {arm}, shared stride {stride}", flush=True)
                model = YOLO(str(checkpoint), task="segment")
                if model.model.frequency_arm != arm:
                    raise ValueError(f"Unexpected checkpoint arm: {checkpoint}")
                evaluation = model.val(validator=CommonMaskGridValidator, data=str(data), split="val",
                    mask_ratio=stride, imgsz=SCHEDULE["imgsz"], batch=batch, workers=0, device=device,
                    plots=False, save_json=False, save_txt=False, project=tmp, name=f"{arm}_s{stride}")
                values[arm] = _metrics(evaluation)
                if stride == 4:
                    expected = report["metrics"]["mask_map50_95_18"]
                    if abs(values[arm]["mask_map50_95_18"] - expected) > .001:
                        raise RuntimeError(f"Common stride-4 endpoint did not reproduce the saved {arm} report")
                del model
            grids[str(stride)] = {"batch": batch, "values": values,
                "mask2_minus_haar_pp": {k:100*(v-values["HAAR"][k])
                    for k,v in values[MASK2_ARM].items() if isinstance(v, (int,float))}}
    return {"protocol": MASK2_PROTOCOL, "evaluation_split": "val", "grids": grids,
        "checkpoint_sha256": {arm: r["checkpoint_sha256"] for arm, (_,r) in inputs.items()},
        "training_executed": False, "test_images_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--haar", type=Path, required=True)
    parser.add_argument("--mask2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    value = compare(args.development, args.haar, args.mask2, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
