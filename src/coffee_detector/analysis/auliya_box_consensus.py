"""One fixed, validation-only class-vote diagnostic for Auliya RGB (B to C)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import ultralytics
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import DetMetrics, box_iou

from coffee_detector.analysis.auliya_robusta_preflight import REUSED_OBJECT_CLASSES, _sha256
from coffee_detector.experiments.run_auliya_robusta_frequency import (
    BATCH, IMAGE_SIZE, PROTOCOL, ULTRALYTICS_VERSION, _development_yaml, _summarize_masks, _write_json,
)


FOCUS = 7
NEIGHBOR_IOU = .75
NEIGHBOR_LIMIT = 8
CANDIDATE_FLOOR = .001
OPERATING_CONF = .25


def vote_labels(boxes, probabilities, retained_idx, retained_cls):
    """Use other overlapping native anchors; permute labels, not boxes or scores."""
    anchors, inverse = retained_idx.unique(return_inverse=True)
    if not anchors.numel():
        return retained_cls
    max_confidence = probabilities.max(dim=1).values
    pool = torch.nonzero(max_confidence >= CANDIDATE_FLOOR).flatten()
    if not pool.numel():
        return retained_cls
    overlap = box_iou(boxes[anchors], boxes[pool])
    valid = (overlap >= NEIGHBOR_IOU) & (anchors[:, None] != pool)
    weights = (overlap * max_confidence[pool]).masked_fill(~valid, 0)
    # Stable order resolves equal neighboring weights by raw anchor index.
    rank = weights.argsort(dim=1, descending=True, stable=True)[:, :NEIGHBOR_LIMIT]
    weights = weights.gather(1, rank)
    total = weights.sum(dim=1, keepdim=True)
    vote = (weights[..., None] * probabilities[pool[rank]]).sum(1) / total.clamp_min(1e-12)
    own = probabilities[anchors]
    old = own.argmax(dim=1)
    new = torch.where(total.flatten() > 0, (own + vote).argmax(dim=1), old)
    old, new = old[inverse], new[inverse]
    # Top-k can include the same anchor in multiple class slots. Swap instead of duplicating a label.
    return torch.where(retained_cls == old, new, torch.where(retained_cls == new, old, retained_cls))


def matched_indices(ious, confidence):
    indices = np.flatnonzero(confidence >= OPERATING_CONF)
    overlap = ious[:, indices]
    if not overlap.size:
        return {}
    valid = overlap >= .5
    rows, cols = linear_sum_assignment(np.where(valid, len(overlap) + 1 + overlap, 0), maximize=True)
    return {int(i): int(indices[j]) for i, j in zip(rows, cols, strict=True) if valid[i, j]}


def audit(development, checkpoint, reference, output, device="cpu"):
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise ValueError(f"Expected Ultralytics {ULTRALYTICS_VERSION}")
    data = _development_yaml(development.resolve())
    saved = json.loads(reference.read_text(encoding="utf-8"))
    sha = _sha256(checkpoint)
    if (saved.get("protocol") != PROTOCOL or saved.get("arm") != "RGB" or saved.get("seed") != 42
            or saved.get("evaluation_split") != "val" or saved.get("test_images_accessed") is not False
            or saved.get("checkpoint_sha256") != sha or saved["run_contract"]["development_contract_sha256"]
            != _sha256(data.parent / "development_contract.json")):
        raise ValueError("Checkpoint and reference must match the original RGB B→C pilot")
    model = YOLO(str(checkpoint), task="segment")
    head = model.model.model[-1]
    if not head.end2end or head.nc != 21 or model.names[FOCUS] != "biji_hitam_sebagian":
        raise ValueError("Unexpected native head or class order")
    expected = {p.resolve() for p in (data.parent / "val/images").glob("*.jpg")}
    seen = set()
    counts = {k: Counter() for k in ("RGB", "consensus")}
    transitions = Counter()
    voted_metrics = DetMetrics(names=model.names)
    changed = [0, 0]

    class Compare(SegmentationValidator):
        def postprocess(self, preds):
            baseline = super().postprocess(preds)
            raw = preds[1]["one2one"]
            decoded = head._inference(raw).transpose(1, 2)
            probs = decoded[..., 4:4 + head.nc]
            conf, cls, idx = head.get_topk_index(probs, head.max_det)
            boxes = decoded[..., :4]
            self.voted = []
            for i, native in enumerate(baseline):
                keep = conf[i].flatten() > CANDIDATE_FLOOR
                ids, native_class = idx[i].flatten()[keep], cls[i].flatten()[keep]
                if (not torch.equal(native["bboxes"], boxes[i, ids])
                        or not torch.equal(native["conf"], conf[i].flatten()[keep])
                        or not torch.equal(native["cls"], native_class)):
                    raise RuntimeError("Decoded anchor indices do not reproduce native detections")
                labels = vote_labels(boxes[i], probs[i], ids, native_class.long()).to(native_class.dtype)
                differs = labels != native_class
                changed[0] += int(differs.sum())
                changed[1] += int((differs & (native["conf"] >= OPERATING_CONF)).sum())
                self.voted.append({**native, "cls": labels})
            return baseline

        def update_metrics(self, preds, batch):
            for i, (native, voted) in enumerate(zip(preds, self.voted, strict=True)):
                path = Path(batch["im_file"][i]).resolve()
                if path not in expected or path in seen:
                    raise RuntimeError("Unexpected or duplicate validation image")
                seen.add(path)
                truth = DetectionValidator._prepare_batch(self, i, batch)
                tcls = truth["cls"].cpu().numpy()
                ious = box_iou(truth["bboxes"], native["bboxes"]).cpu().numpy()
                confidence = native["conf"].cpu().numpy()
                matches = matched_indices(ious, confidence)
                above = confidence >= OPERATING_CONF
                outcomes = []
                for name, arm in (("RGB", native), ("consensus", voted)):
                    pred_cls = arm["cls"].cpu().numpy()
                    stats = DetectionValidator._process_batch(self, arm, truth)
                    if name == "consensus":
                        voted_metrics.update_stats({**stats, "target_cls": tcls,
                            "target_img": np.unique(tcls), "conf": confidence,
                            "pred_cls": pred_cls, "im_name": path.name})
                    focus = np.flatnonzero(tcls == FOCUS)
                    ok = np.array([g in matches and pred_cls[matches[g]] == FOCUS for g in focus], dtype=bool)
                    outcomes.append(ok)
                    c = counts[name]
                    c["target_gt"] += len(focus)
                    c["target_correct_on_matched_boxes"] += int(ok.sum())
                    # Match after applying the operating cutoff; a filtered box cannot claim a true positive.
                    eligible_iou = torch.as_tensor(ious[:, above], device=self.device)
                    tp50 = self.match_predictions(arm["cls"][above], truth["cls"], eligible_iou)[:, 0].cpu().numpy()
                    c["focus_fp_at_conf025_iou50"] += int(((pred_cls[above] == FOCUS) & ~tp50).sum())
                    c["all_fp_at_conf025_iou50"] += int((~tp50).sum())
                transitions["rescued"] += int((outcomes[1] & ~outcomes[0]).sum())
                transitions["regressed"] += int((outcomes[0] & ~outcomes[1]).sum())
            return super().update_metrics(preds, batch)

    with TemporaryDirectory(prefix="auliya-consensus-") as temp:
        baseline = model.val(validator=Compare, data=str(data), split="val", imgsz=IMAGE_SIZE,
                             batch=BATCH, device=device, workers=0, conf=CANDIDATE_FLOOR,
                             max_det=300, project=temp, plots=False, save_txt=False, save_json=False)
    drift = _summarize_masks(baseline.seg, baseline.names)["mask_map50_95_18"] - saved["metrics"]["mask_map50_95_18"]
    if seen != expected or abs(drift) > .001:
        raise RuntimeError(f"Historical RGB endpoint not reproduced: mask mAP drift {100 * drift:.4f} pp")
    voted_metrics.process(plot=False)

    def summarize(metric):
        ap = dict(zip(metric.ap_class_index.tolist(), metric.ap.tolist(), strict=True))
        clean = sorted(v for k, v in ap.items() if k not in REUSED_OBJECT_CLASSES)
        if len(clean) != 18:
            raise RuntimeError("Missing clean validation classes")
        return {"box_map50_95_18": float(np.mean(clean)), "bottom3_box_ap50_95_18": float(np.mean(clean[:3])),
                "focus_box_ap50_95": float(ap[FOCUS])}

    payload = {
        "format": "auliya.rgb_box_consensus.val_only.v1", "scope": "YOLO26n RGB pilot; B to C only",
        "checkpoint_sha256": sha, "images": len(seen), "training_executed": False,
        "test_images_accessed": False, "baseline_mask_calibration_drift_pp": 100 * drift,
        "changed_classes": {"all": changed[0], "conf025": changed[1]}, "target_transitions": dict(transitions),
        "RGB": {**summarize(baseline.box), **counts["RGB"]},
        "consensus": {**summarize(voted_metrics.box), **counts["consensus"]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("development", "checkpoint", "reference", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(audit(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
