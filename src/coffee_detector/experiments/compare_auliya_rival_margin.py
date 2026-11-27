"""Compare native RGB and RGBMARGIN1 on the same localized fold-A objects."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import ultralytics
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import box_iou

from coffee_detector.analysis.auliya_robusta_preflight import REUSED_OBJECT_CLASSES, _sha256
from coffee_detector.auliya_mask2 import CommonMaskGridValidator
from coffee_detector.auliya_rival_margin import RIVAL_ARM, RIVAL_CONTRACT, RIVAL_PROTOCOL
from coffee_detector.experiments.compare_auliya_mask2 import _metrics, _report
from coffee_detector.experiments.prepare_auliya_official_fold import validate_official_fold
from coffee_detector.experiments.run_auliya_official_frequency import SCHEDULE, ULTRALYTICS_VERSION
from coffee_detector.experiments.run_auliya_robusta_frequency import _write_json

ARMS = ("RGB", RIVAL_ARM)
FOCUS = {
    "biji_hitam", "biji_hitam_pecah", "biji_hitam_sebagian", "biji_pecah",
    "biji_berlubang_satu", "biji_berlubang_lebih_satu",
}


def class_blind_match(ious: np.ndarray, classes: np.ndarray, confidence: np.ndarray,
                      *, threshold: float = .50, conf: float = .25) -> dict[int, int]:
    """Maximize valid one-to-one box pairs, then summed IoU; class never affects matching."""
    keep = np.flatnonzero(confidence >= conf)
    overlap = ious[:, keep]
    if not overlap.size:
        return {}
    valid = overlap >= threshold
    row, col = linear_sum_assignment(np.where(valid, (len(overlap) + 1) + overlap, 0), maximize=True)
    return {int(i): int(classes[keep[j]]) for i, j in zip(row, col, strict=True) if valid[i, j]}


def paired_objects(truth: dict, predictions: dict, names: dict, all_images: list[str], *, arms=ARMS) -> dict:
    common = sorted(set.intersection(*(set(predictions[arm]) for arm in arms)))
    if not common:
        raise ValueError("No common localized objects")
    y = np.array([truth[k] for k in common])
    counts = Counter(y)
    correct, table = {}, {}
    for arm in arms:
        p = np.array([predictions[arm][k] for k in common])
        ok = (p == y).astype(int)
        correct[arm] = ok
        by_class = Counter(y[ok == 1])
        transitions = Counter((names[i], names[j]) for i, j in zip(y, p, strict=True) if i != j)
        class_accuracy = sorted(by_class[c] / n for c, n in counts.items())
        table[arm] = {
            "matched": len(predictions[arm]), "correct_common": int(ok.sum()),
            "accuracy_common": float(ok.mean()), "macro_accuracy_common": float(np.mean(class_accuracy)),
            "bottom3_accuracy_common": float(np.mean(class_accuracy[:3])),
            "worst_accuracy_common": float(class_accuracy[0]),
            "per_class": {names[c]: {"support": n, "correct": by_class[c]} for c, n in sorted(counts.items())},
            "confusions": [[a, b, n] for (a, b), n in transitions.most_common()],
        }
    diff = correct[arms[1]] - correct[arms[0]]
    index = {name: i for i, name in enumerate(all_images)}
    img = np.array([index[k[0]] for k in common])
    draws = np.random.default_rng(422026).integers(len(all_images), size=(10000, len(all_images)))
    denom = np.bincount(img, minlength=len(all_images))[draws].sum(axis=1)
    num = np.bincount(img, weights=diff, minlength=len(all_images))[draws].sum(axis=1)
    ci = np.quantile(100 * num[denom > 0] / denom[denom > 0], [.025, .975])
    return {
        "eligible_objects": len(truth), "common_objects": len(common), "common_images": len(set(img)),
        "arms": table, "accuracy_delta_pp": 100 * float(diff.mean()),
        "rescues": int((diff == 1).sum()), "regressions": int((diff == -1).sum()),
        "paired_bootstrap_by_image_ci95_pp": ci.tolist(),
    }


def compare(development: Path, rgb: Path, candidate: Path, device: str, *, arm: str = RIVAL_ARM) -> dict:
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f"Expected Ultralytics {ULTRALYTICS_VERSION}")
    if arm != RIVAL_ARM:
        raise ValueError(f"Unknown fold-A classification treatment: {arm}")
    arms = ("RGB", arm)
    protocol = RIVAL_PROTOCOL
    data = validate_official_fold(development.resolve())
    digest = _sha256(data.parent / "official_fold_contract.json")
    paths = sorted({str((data.parent / line).resolve()) for line in
                    (data.parent / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip()})
    valset = set(paths)
    baseline, record = _report(rgb, "RGB", digest)
    cand = candidate / "weights" / "best.pt"
    result = json.loads((candidate / "validation_result.json").read_text(encoding="utf-8"))
    contract = result["run_contract"]
    expected = {**record["run_contract"], "arm": arm, "protocol": protocol,
                "rgb_reference_checkpoint_sha256": record["checkpoint_sha256"]}
    if arm == RIVAL_ARM:
        expected["classification_objective"] = RIVAL_CONTRACT
    if (contract != expected or result.get("arm") != arm or result.get("evaluation_split") != "val"
            or result.get("test_images_accessed") is not False
            or result.get("checkpoint_sha256") != _sha256(cand)):
        raise ValueError("The candidate result or exact paired checkpoint contract does not match")

    class Capture(CommonMaskGridValidator):
        def update_metrics(self, preds, batch):
            for i, p in enumerate(preds):
                key = str(Path(batch["im_file"][i]).resolve())
                if key not in valset or key in current:
                    raise ValueError("Validator emitted an unknown or repeated image")
                g = DetectionValidator._prepare_batch(self, i, batch)
                current[key] = (
                    g["cls"].cpu().numpy().astype(int),
                    g["bboxes"].cpu().numpy(),
                    box_iou(g["bboxes"], p["bboxes"]).cpu().numpy(),
                    p["cls"].cpu().numpy().astype(int),
                    p["conf"].cpu().numpy(),
                )
            return super().update_metrics(preds, batch)

    runs, endpoints = {}, {}
    with TemporaryDirectory(prefix="auliya-rgbmargin1-paired-") as temp:
        for key, checkpoint, saved in zip(arms, (baseline, cand), (record, result), strict=True):
            current = {}
            model = YOLO(str(checkpoint), task="segment")
            if getattr(model.model, "frequency_arm", "RGB") != key:
                raise ValueError(f"Unexpected model arm: {key}")
            evaluated = model.val(
                validator=Capture, data=str(data), split="val", imgsz=SCHEDULE["imgsz"],
                batch=SCHEDULE["batch"], device=device, workers=0, mask_ratio=4,
                conf=.001, iou=.7, max_det=300, plots=False, save_json=False, save_txt=False,
                project=temp, name=key,
            )
            m = _metrics(evaluated)
            drift = 100 * (m["mask_map50_95_18"] - saved["metrics"]["mask_map50_95_18"])
            if set(current) != valset or abs(drift) > .10:
                raise RuntimeError(f"Failed to reproduce {key} mask endpoint or validation inventory: {drift:.3f} pp")
            endpoints[key] = {**m, "mask_endpoint_drift_pp": drift}
            runs[key] = current
            names = evaluated.names
    truth = {}
    for key in paths:
        g, boxes, *_ = runs["RGB"][key]
        other_g, other_boxes, *_ = runs[arm][key]
        if not (np.array_equal(g, other_g) and np.array_equal(boxes, other_boxes)):
            raise RuntimeError("Ground truth is not identical between the paired runs")
        truth.update({(key, i): int(c) for i, c in enumerate(g) if int(c) not in REUSED_OBJECT_CLASSES})
    subsets = {"18_class": truth, "prespecified_six_classes": {k: v for k, v in truth.items() if names[v] in FOCUS}}
    results = {}
    for conf, threshold in ((.25, .50), (.25, .75), (.10, .50), (.50, .50)):
        pred = {}
        for key in arms:
            pred[key] = {
                (p, i): c for p in paths
                for i, c in class_blind_match(*runs[key][p][2:], conf=conf, threshold=threshold).items()
                if (p, i) in truth
            }
        results[f"conf_{conf:.2f}_iou_{threshold:.2f}"] = {
            name: paired_objects(gt, {a: {k: v for k, v in pred[a].items() if k in gt} for a in arms},
                                 names, paths, arms=arms)
            for name, gt in subsets.items()
        }
    return {
        "protocol": protocol, "seed": SCHEDULE["seed"], "evaluation_split": "val",
        "evaluation_images": len(paths), "training_executed": False, "test_images_accessed": False,
        "checkpoint_sha256": {a: d["checkpoint_sha256"] for a, d in zip(arms, (record, result), strict=True)},
        "box_map50_95_21_delta_pp": 100 * (endpoints[arm]["box_map50_95_21"] -
                                         endpoints["RGB"]["box_map50_95_21"]),
        "mask_map50_95_18_delta_pp": 100 * (endpoints[arm]["mask_map50_95_18"] -
                                          endpoints["RGB"]["mask_map50_95_18"]),
        "endpoints": endpoints, "paired_classification": results, "status": "REVIEW_SINGLE_SEED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "rgb", "candidate", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--arm", choices=(RIVAL_ARM,), default=RIVAL_ARM)
    args = parser.parse_args()
    value = compare(args.development, args.rgb, args.candidate, args.device, arm=args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, value)
    primary = value["paired_classification"]["conf_0.25_iou_0.50"]
    print(json.dumps({"paired_classification": primary, "box_delta_pp": value["box_map50_95_21_delta_pp"],
                      "mask_delta_pp": value["mask_map50_95_18_delta_pp"],
                      "status": value["status"], "output": str(args.output),
                      "training": False, "test": False}, indent=2))


if __name__ == "__main__":
    main()
