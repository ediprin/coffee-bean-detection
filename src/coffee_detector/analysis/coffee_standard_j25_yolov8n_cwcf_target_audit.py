"""Validation-only paired target audit for YOLOv8n MATCHED vs CWCF1.

This audit does not train and does not open the locked test. It inspects the
six validation instances of Biji Hitam Pecah plus the local black/broken
confusion family. For CWCF1 it also inspects the auxiliary attribute evidence
at the same localized raw anchors.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from coffee_detector.analysis.coffee_fg_diagnostics import _letterbox_sample
from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout, parse_label
from coffee_detector.experiments.run_coffee_standard_j25_af2_direct import (
    SEED,
    _json,
    validate_j25_development,
)
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_cwcf import (
    PROTOCOL as CWCF1_PROTOCOL,
)
from coffee_detector.experiments.run_coffee_standard_j25_yolov8n_matched import (
    PROTOCOL as MATCHED_PROTOCOL,
)
from coffee_detector.experiments.run_faruq_v3_af2_direct import _sha256
from coffee_detector.j25_cwcf import (
    ChromaticWaveletDetectHead,
    build_j25_attribute_matrix,
    chromatic_wavelet_cue,
)
from coffee_detector.j25_cwcf.model import ATTRIBUTE_NAMES


PROTOCOL = "coffee-standard-j25-yolov8n-cwcf-target-audit-v1"
TARGET_CLASS = "Biji Hitam Pecah"
CONFUSION_FAMILY_NAMES = (
    "Biji Hitam Pecah",
    "Biji Hitam Penuh",
    "Biji Hitam Sebagian",
    "Biji Pecah",
)
RAW_COUNT = 500
IOU_THRESHOLD = 0.50
FINAL_CONFIDENCE = 0.001
NMS_IOU = 0.70
MAX_DET = 500


def _unwrap_head(model: torch.nn.Module):
    head = model.model[-1]
    while hasattr(head, "base_head"):
        head = head.base_head
    return head


def _component_id(path: Path) -> str:
    match = re.match(r"^([0-9a-f]{64})(?:_\d{2})?$", path.stem, flags=re.I)
    return match.group(1).lower() if match else path.stem


def _split_samples(data_root: Path, split: str):
    layout = discover_layout(data_root)
    image_root, label_root = layout.splits[split]
    rows = []
    for image in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image.relative_to(image_root)
        label = (label_root / relative).with_suffix(".txt")
        rows.append((image, parse_label(label, set(layout.names))))
    return layout, rows


def _class_counts(annotations, names: dict[int, str]) -> dict[str, int]:
    counts = Counter(item.class_id for item in annotations)
    return {
        names[index]: int(count)
        for index, count in sorted(counts.items())
        if count
    }


def _support_context(data_root: Path, target_id: int, names: dict[int, str]) -> dict:
    payload = {}
    for split in ("train", "val"):
        _, samples = _split_samples(data_root, split)
        images = []
        components = set()
        total_instances = 0
        for image, annotations in samples:
            target_count = sum(item.class_id == target_id for item in annotations)
            if not target_count:
                continue
            total = len(annotations)
            components.add(_component_id(image))
            total_instances += target_count
            images.append(
                {
                    "image": image.name,
                    "component_id": _component_id(image),
                    "target_instances": int(target_count),
                    "total_instances": int(total),
                    "target_purity": float(target_count / max(total, 1)),
                    "class_counts": _class_counts(annotations, names),
                }
            )
        payload[split] = {
            "images_with_target": len(images),
            "target_instances": int(total_instances),
            "source_components_with_target": len(components),
            "images": images,
        }
    return payload


def _topk(scores: torch.Tensor, names: dict[int, str], k: int = 5) -> list[dict]:
    k = min(k, int(scores.numel()))
    values, indices = scores.topk(k)
    return [
        {
            "class_id": int(index),
            "class_name": names[int(index)],
            "score": float(value),
        }
        for value, index in zip(values.detach().cpu(), indices.detach().cpu())
    ]


def _rank(scores: torch.Tensor, class_id: int) -> int:
    order = scores.argsort(descending=True)
    position = (order == int(class_id)).nonzero(as_tuple=False)
    return int(position[0, 0]) + 1


def _attribute_compatibility_2d(
    attribute_logits: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    if attribute_logits.ndim != 2:
        raise ValueError("attribute logits harus [N,A]")
    codes = matrix.to(
        device=attribute_logits.device, dtype=attribute_logits.dtype
    )
    positive = F.logsigmoid(attribute_logits)[:, None, :]
    negative = F.logsigmoid(-attribute_logits)[:, None, :]
    compatibility = (
        codes[None, :, :] * positive
        + (1.0 - codes[None, :, :]) * negative
    ).mean(dim=-1)
    return compatibility - compatibility.mean(dim=1, keepdim=True)


def _cwcf_attribute_logits(
    network: torch.nn.Module,
    head: ChromaticWaveletDetectHead,
    image: torch.Tensor,
) -> tuple[Any, torch.Tensor]:
    captured: dict[str, list[torch.Tensor]] = {}

    def pre_hook(_module, args):
        captured["features"] = [value.detach() for value in args[0]]

    handle = head.register_forward_pre_hook(pre_hook)
    try:
        output = network(image)
    finally:
        handle.remove()
    if "features" not in captured:
        raise RuntimeError("Feature pyramid CWCF tidak tertangkap")

    cue = chromatic_wavelet_cue(image, clip=head.config.cue_clip)
    attributes = []
    for index, feature in enumerate(captured["features"]):
        conditioned = head.adapters[index](feature, cue)
        attributes.append(
            head.attribute_heads[index](conditioned).view(
                image.shape[0], len(ATTRIBUTE_NAMES), -1
            )
        )
    logits = torch.cat(attributes, dim=-1).permute(0, 2, 1).contiguous()[0]
    return output, logits


def _forward_raw(
    network: torch.nn.Module,
    image: torch.Tensor,
    *,
    with_attributes: bool,
) -> dict[str, Any]:
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import xywh2xyxy

    wrapper = network.model[-1]
    if with_attributes:
        if not isinstance(wrapper, ChromaticWaveletDetectHead):
            raise TypeError("Checkpoint CWCF tidak memakai ChromaticWaveletDetectHead")
        output, attribute_logits = _cwcf_attribute_logits(
            network, wrapper, image
        )
    else:
        output = network(image)
        attribute_logits = None

    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("YOLOv8 checkpoint tidak mengembalikan (inference, raw)")
    inference, raw = output
    if not isinstance(raw, dict) or not {"boxes", "scores", "feats"} <= set(raw):
        raise TypeError("Raw YOLOv8 prediction schema tidak sesuai")

    base_head = _unwrap_head(network)
    decoded = base_head._get_decode_boxes(raw).transpose(1, 2)[0]
    raw_boxes = decoded if getattr(base_head, "xyxy", False) else xywh2xyxy(decoded)
    raw_scores = raw["scores"].sigmoid().transpose(1, 2)[0]
    confidence = raw_scores.max(dim=1).values
    count = min(RAW_COUNT, int(confidence.numel()))
    selected_indices = confidence.topk(count).indices

    final = non_max_suppression(
        inference,
        conf_thres=FINAL_CONFIDENCE,
        iou_thres=NMS_IOU,
        max_det=MAX_DET,
        nc=int(base_head.nc),
        end2end=False,
    )[0]

    return {
        "raw_boxes": raw_boxes,
        "raw_scores": raw_scores,
        "raw_confidence": confidence,
        "selected_indices": selected_indices,
        "final": final,
        "attribute_logits": attribute_logits,
    }


def _localized_candidates(
    predicted_boxes: torch.Tensor,
    target_box: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not len(predicted_boxes):
        empty = torch.empty(0, dtype=torch.long, device=target_box.device)
        return empty, target_box.new_empty((0,))
    ious = box_iou(predicted_boxes, target_box[None]).squeeze(1)
    indices = (ious >= threshold).nonzero(as_tuple=False).flatten()
    return indices, ious[indices]


def _raw_view_for_gt(
    raw: dict[str, Any],
    target_box: torch.Tensor,
    expected_class: int,
    names: dict[int, str],
    *,
    attribute_matrix: torch.Tensor | None,
) -> dict:
    selected = raw["selected_indices"]
    boxes = raw["raw_boxes"][selected]
    scores = raw["raw_scores"][selected]
    confidence = raw["raw_confidence"][selected]
    local, local_ious = _localized_candidates(boxes, target_box, IOU_THRESHOLD)
    result = {
        "accessible_top500": bool(local.numel()),
        "localized_candidates_top500": int(local.numel()),
    }
    if not local.numel():
        return result

    local_conf = confidence[local]
    best_local_offset = int(local_conf.argmax())
    local_index = int(local[best_local_offset])
    anchor_index = int(selected[local_index])
    vector = scores[local_index]
    result["best_conf_localized"] = {
        "anchor_index": anchor_index,
        "iou": float(local_ious[best_local_offset]),
        "predicted_class_id": int(vector.argmax()),
        "predicted_class_name": names[int(vector.argmax())],
        "max_score": float(vector.max()),
        "expected_score": float(vector[expected_class]),
        "expected_rank": _rank(vector, expected_class),
        "top5": _topk(vector, names),
    }

    target_scores = scores[local, expected_class]
    best_target_offset = int(target_scores.argmax())
    target_local_index = int(local[best_target_offset])
    target_anchor = int(selected[target_local_index])
    target_vector = scores[target_local_index]
    result["best_expected_score_localized"] = {
        "anchor_index": target_anchor,
        "iou": float(local_ious[best_target_offset]),
        "predicted_class_id": int(target_vector.argmax()),
        "predicted_class_name": names[int(target_vector.argmax())],
        "max_score": float(target_vector.max()),
        "expected_score": float(target_vector[expected_class]),
        "expected_rank": _rank(target_vector, expected_class),
        "top5": _topk(target_vector, names),
    }

    if raw["attribute_logits"] is not None:
        logits = raw["attribute_logits"][anchor_index]
        probs = logits.sigmoid()
        black_index = ATTRIBUTE_NAMES.index("black")
        broken_index = ATTRIBUTE_NAMES.index("broken")
        expected_code = attribute_matrix[expected_class]
        predicted_code = (probs >= 0.5).to(expected_code.dtype)
        compatibility = _attribute_compatibility_2d(
            logits[None], attribute_matrix
        )[0]
        result["best_conf_localized"]["attributes"] = {
            "black_probability": float(probs[black_index]),
            "broken_probability": float(probs[broken_index]),
            "black_binary": bool(probs[black_index] >= 0.5),
            "broken_binary": bool(probs[broken_index] >= 0.5),
            "attribute_binary_mismatches": int(
                predicted_code.ne(expected_code).sum()
            ),
            "compatibility_expected_rank": _rank(
                compatibility, expected_class
            ),
            "compatibility_top5": _topk(compatibility, names),
        }

        target_logits = raw["attribute_logits"][target_anchor]
        target_probs = target_logits.sigmoid()
        target_compatibility = _attribute_compatibility_2d(
            target_logits[None], attribute_matrix
        )[0]
        result["best_expected_score_localized"]["attributes"] = {
            "black_probability": float(target_probs[black_index]),
            "broken_probability": float(target_probs[broken_index]),
            "black_binary": bool(target_probs[black_index] >= 0.5),
            "broken_binary": bool(target_probs[broken_index] >= 0.5),
            "compatibility_expected_rank": _rank(
                target_compatibility, expected_class
            ),
            "compatibility_top5": _topk(target_compatibility, names),
        }
    return result


def _final_view_for_gt(
    final: torch.Tensor,
    target_box: torch.Tensor,
    expected_class: int,
    names: dict[int, str],
) -> dict:
    if not len(final):
        return {"matched": False}
    indices, ious = _localized_candidates(final[:, :4], target_box, IOU_THRESHOLD)
    if not indices.numel():
        return {"matched": False}
    confidence = final[indices, 4]
    offset = int(confidence.argmax())
    index = int(indices[offset])
    actual = int(final[index, 5])
    return {
        "matched": True,
        "iou": float(ious[offset]),
        "confidence": float(final[index, 4]),
        "predicted_class_id": actual,
        "predicted_class_name": names[actual],
        "correct_class": actual == expected_class,
    }


def _expected_pair(class_id: int, attribute_matrix: torch.Tensor) -> tuple[bool, bool]:
    black = ATTRIBUTE_NAMES.index("black")
    broken = ATTRIBUTE_NAMES.index("broken")
    return (
        bool(attribute_matrix[class_id, black] > 0.5),
        bool(attribute_matrix[class_id, broken] > 0.5),
    )


def _summarize_instances(rows: list[dict], target_id: int) -> dict:
    target = [row for row in rows if row["expected_class_id"] == target_id]
    raw_accessible = sum(row["raw"].get("accessible_top500", False) for row in target)
    raw_correct = sum(
        row["raw"].get("best_conf_localized", {}).get("predicted_class_id")
        == target_id
        for row in target
    )
    final_matched = sum(row["final"].get("matched", False) for row in target)
    final_correct = sum(row["final"].get("correct_class", False) for row in target)
    destinations = Counter(
        row["raw"].get("best_conf_localized", {}).get("predicted_class_name")
        for row in target
        if row["raw"].get("best_conf_localized")
        and row["raw"]["best_conf_localized"]["predicted_class_id"] != target_id
    )
    result = {
        "instances": len(target),
        "raw_top500_accessible": int(raw_accessible),
        "raw_best_conf_correct": int(raw_correct),
        "final_conf0001_matched": int(final_matched),
        "final_conf0001_correct": int(final_correct),
        "raw_wrong_destinations": {
            key: int(value) for key, value in destinations.items() if key
        },
        "mean_expected_leaf_rank_at_best_conf": float(
            np.mean(
                [
                    row["raw"]["best_conf_localized"]["expected_rank"]
                    for row in target
                    if row["raw"].get("best_conf_localized")
                ]
            )
        ),
        "mean_expected_leaf_score_at_best_conf": float(
            np.mean(
                [
                    row["raw"]["best_conf_localized"]["expected_score"]
                    for row in target
                    if row["raw"].get("best_conf_localized")
                ]
            )
        ),
    }
    attribute_rows = [
        row["raw"]["best_conf_localized"]["attributes"]
        for row in target
        if row["raw"].get("best_conf_localized", {}).get("attributes")
    ]
    if attribute_rows:
        result["cwcf_attribute_summary"] = {
            "black_positive": int(
                sum(row["black_binary"] for row in attribute_rows)
            ),
            "broken_positive": int(
                sum(row["broken_binary"] for row in attribute_rows)
            ),
            "black_and_broken_positive": int(
                sum(
                    row["black_binary"] and row["broken_binary"]
                    for row in attribute_rows
                )
            ),
            "mean_compatibility_expected_rank": float(
                np.mean(
                    [row["compatibility_expected_rank"] for row in attribute_rows]
                )
            ),
            "mean_attribute_binary_mismatches": float(
                np.mean(
                    [row["attribute_binary_mismatches"] for row in attribute_rows]
                )
            ),
        }
    return result


def _family_summary(
    rows: list[dict],
    family_ids: set[int],
    names: dict[int, str],
    attribute_matrix: torch.Tensor,
) -> dict:
    selected = [row for row in rows if row["expected_class_id"] in family_ids]
    confusion: dict[str, Counter] = defaultdict(Counter)
    pair_correct = Counter()
    pair_total = Counter()
    for row in selected:
        expected = row["expected_class_name"]
        best = row["raw"].get("best_conf_localized")
        if best:
            confusion[expected][best["predicted_class_name"]] += 1
            attributes = best.get("attributes")
            if attributes:
                expected_pair = _expected_pair(
                    row["expected_class_id"], attribute_matrix
                )
                actual_pair = (
                    attributes["black_binary"],
                    attributes["broken_binary"],
                )
                pair_total[expected] += 1
                pair_correct[expected] += int(actual_pair == expected_pair)
    return {
        "instances": len(selected),
        "raw_confusion": {
            expected: {
                actual: int(count)
                for actual, count in sorted(destinations.items())
            }
            for expected, destinations in sorted(confusion.items())
        },
        "attribute_black_broken_pair_accuracy": {
            expected: {
                "correct": int(pair_correct[expected]),
                "total": int(pair_total[expected]),
                "accuracy": float(
                    pair_correct[expected] / max(pair_total[expected], 1)
                ),
            }
            for expected in sorted(pair_total)
        },
    }


def _audit_model(
    checkpoint: Path,
    data_root: Path,
    *,
    names: dict[int, str],
    target_id: int,
    family_ids: set[int],
    device: torch.device,
    with_attributes: bool,
) -> dict:
    from ultralytics import YOLO

    network = YOLO(str(checkpoint)).model.to(device).eval()
    checkpoint_classes = int(_unwrap_head(network).nc)
    if checkpoint_classes != len(names):
        raise RuntimeError("Jumlah kelas checkpoint tidak cocok dengan dataset")
    matrix = build_j25_attribute_matrix().to(device)

    _, samples = _split_samples(data_root, "val")
    target_images = [
        (image, annotations)
        for image, annotations in samples
        if any(item.class_id == target_id for item in annotations)
    ]
    rows = []
    with torch.inference_mode():
        for image_path, annotations in target_images:
            image, boxes, labels, original_size = _letterbox_sample(
                image_path, annotations, 640, device
            )
            raw = _forward_raw(
                network, image, with_attributes=with_attributes
            )
            for gt_index, (box, label) in enumerate(zip(boxes, labels)):
                class_id = int(label)
                if class_id not in family_ids:
                    continue
                rows.append(
                    {
                        "image": image_path.name,
                        "original_size": list(original_size),
                        "gt_index": int(gt_index),
                        "expected_class_id": class_id,
                        "expected_class_name": names[class_id],
                        "raw": _raw_view_for_gt(
                            raw,
                            box,
                            class_id,
                            names,
                            attribute_matrix=matrix if with_attributes else None,
                        ),
                        "final": _final_view_for_gt(
                            raw["final"], box, class_id, names
                        ),
                    }
                )

    return {
        "checkpoint": str(checkpoint),
        "target_summary": _summarize_instances(rows, target_id),
        "confusion_family_summary": _family_summary(
            rows, family_ids, names, matrix
        ),
        "instances": rows,
    }


def _paired_target_comparison(
    matched: dict,
    cwcf: dict,
    target_id: int,
) -> list[dict]:
    def keyed(model):
        return {
            (row["image"], row["gt_index"]): row
            for row in model["instances"]
            if row["expected_class_id"] == target_id
        }

    left = keyed(matched)
    right = keyed(cwcf)
    if set(left) != set(right):
        raise RuntimeError("Target GT pairing MATCHED/CWCF1 berbeda")

    paired = []
    for key in sorted(left):
        base = left[key]
        candidate = right[key]
        base_raw = base["raw"].get("best_conf_localized", {})
        candidate_raw = candidate["raw"].get("best_conf_localized", {})
        attrs = candidate_raw.get("attributes")
        paired.append(
            {
                "image": key[0],
                "gt_index": key[1],
                "matched_predicted_class": base_raw.get("predicted_class_name"),
                "cwcf1_predicted_class": candidate_raw.get("predicted_class_name"),
                "matched_target_score": base_raw.get("expected_score"),
                "cwcf1_target_score": candidate_raw.get("expected_score"),
                "target_score_delta": (
                    candidate_raw.get("expected_score") - base_raw.get("expected_score")
                    if "expected_score" in candidate_raw
                    and "expected_score" in base_raw
                    else None
                ),
                "matched_target_rank": base_raw.get("expected_rank"),
                "cwcf1_target_rank": candidate_raw.get("expected_rank"),
                "cwcf1_black_probability":
                    None if attrs is None else attrs["black_probability"],
                "cwcf1_broken_probability":
                    None if attrs is None else attrs["broken_probability"],
                "cwcf1_compatibility_target_rank":
                    None if attrs is None else attrs["compatibility_expected_rank"],
                "matched_final_class":
                    base["final"].get("predicted_class_name"),
                "cwcf1_final_class":
                    candidate["final"].get("predicted_class_name"),
            }
        )
    return paired


def run_audit(
    data_root: str | Path,
    development_contract: str | Path,
    provenance_summary: str | Path,
    matched_checkpoint: str | Path,
    matched_result: str | Path,
    cwcf1_checkpoint: str | Path,
    cwcf1_result: str | Path,
    output: str | Path,
    *,
    device: str = "0",
    authorize_diagnostic: bool = False,
) -> dict:
    if not authorize_diagnostic:
        raise RuntimeError("Audit harus diotorisasi eksplisit")

    root = Path(data_root).expanduser().resolve()
    dataset = validate_j25_development(
        root, development_contract, provenance_summary
    )
    if not dataset["data_format"].endswith("train_siblings.v2"):
        raise RuntimeError("Audit hanya untuk J25 train-siblings v2")

    checkpoints = {
        "V8N_MATCHED": Path(matched_checkpoint).expanduser().resolve(),
        "V8N_CWCF1": Path(cwcf1_checkpoint).expanduser().resolve(),
    }
    results = {
        "V8N_MATCHED": _json(matched_result, "V8N_MATCHED result"),
        "V8N_CWCF1": _json(cwcf1_result, "V8N_CWCF1 result"),
    }
    expected_protocols = {
        "V8N_MATCHED": MATCHED_PROTOCOL,
        "V8N_CWCF1": CWCF1_PROTOCOL,
    }
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = results[name]
        if (
            result.get("protocol") != expected_protocols[name]
            or result.get("seed") != SEED
            or result.get("test_images_accessed") is not False
        ):
            raise RuntimeError(f"Result contract salah: {name}")
        if result.get("checkpoint_sha256") != _sha256(checkpoint):
            raise RuntimeError(f"Checkpoint SHA berbeda: {name}")

    layout = discover_layout(root)
    target_ids = [
        index for index, name in layout.names.items() if name == TARGET_CLASS
    ]
    if len(target_ids) != 1:
        raise RuntimeError("Target class ambigu/tidak ditemukan")
    target_id = target_ids[0]
    family_ids = {
        index
        for index, name in layout.names.items()
        if name in CONFUSION_FAMILY_NAMES
    }
    if len(family_ids) != len(CONFUSION_FAMILY_NAMES):
        raise RuntimeError("Confusion family tidak lengkap")

    torch_device = torch.device(
        f"cuda:{device}" if str(device).isdigit() else device
    )
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diminta tetapi tidak tersedia")

    support = _support_context(root, target_id, layout.names)
    matched = _audit_model(
        checkpoints["V8N_MATCHED"],
        root,
        names=layout.names,
        target_id=target_id,
        family_ids=family_ids,
        device=torch_device,
        with_attributes=False,
    )
    cwcf = _audit_model(
        checkpoints["V8N_CWCF1"],
        root,
        names=layout.names,
        target_id=target_id,
        family_ids=family_ids,
        device=torch_device,
        with_attributes=True,
    )
    paired = _paired_target_comparison(matched, cwcf, target_id)

    payload = {
        "format": "coffee_detector.coffee_standard_j25.yolov8n_cwcf_target_audit.v1",
        "protocol": PROTOCOL,
        "seed": SEED,
        "target_class": TARGET_CLASS,
        "target_class_id": target_id,
        "confusion_family": list(CONFUSION_FAMILY_NAMES),
        "support_context": support,
        "models": {
            "V8N_MATCHED": matched,
            "V8N_CWCF1": cwcf,
        },
        "paired_target_comparison": paired,
        "settings": {
            "image_size": 640,
            "raw_candidates": RAW_COUNT,
            "iou_threshold": IOU_THRESHOLD,
            "final_confidence": FINAL_CONFIDENCE,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
        },
        "training_executed": False,
        "test_opened": False,
    }

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    concise = {
        "support_context": support,
        "V8N_MATCHED": matched["target_summary"],
        "V8N_CWCF1": cwcf["target_summary"],
        "family_MATCHED": matched["confusion_family_summary"],
        "family_CWCF1": cwcf["confusion_family_summary"],
        "paired_target_comparison": paired,
        "test_opened": False,
    }
    print(json.dumps(concise, indent=2, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--development-contract", required=True)
    parser.add_argument("--provenance-summary", required=True)
    parser.add_argument("--matched-checkpoint", required=True)
    parser.add_argument("--matched-result", required=True)
    parser.add_argument("--cwcf1-checkpoint", required=True)
    parser.add_argument("--cwcf1-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--authorize-diagnostic", action="store_true")
    args = parser.parse_args()

    run_audit(
        args.data_root,
        args.development_contract,
        args.provenance_summary,
        args.matched_checkpoint,
        args.matched_result,
        args.cwcf1_checkpoint,
        args.cwcf1_result,
        args.output,
        device=args.device,
        authorize_diagnostic=args.authorize_diagnostic,
    )


if __name__ == "__main__":
    main()
