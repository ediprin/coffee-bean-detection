"""Validation-only object-level complementarity audit for STB1 versus CMC0.

This audit is deliberately post-training. It does not retrain, tune, or access test.
Each final prediction set is matched to validation GT boxes class-agnostically at
IoU >= 0.50 in descending confidence order. The same GT target is then compared
across CMC0 and STB1 to measure directional rescue, shared-error overlap,
classification-only rescue on jointly matched targets, oracle headroom, and
confusion-pair rescue.

Important: these object-level accuracy diagnostics are not mAP and must not be
reported as replacements for the frozen Macro/Bottom-3/Worst AP metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

from coffee_detector.dataset import IMAGE_SUFFIXES, discover_layout, parse_label

SEEDS = (42, 123, 2026)
MATCH_IOU = 0.50
CONFIDENCE_THRESHOLD = 0.25
PREDICTION_NMS_IOU = 0.70
MAX_DET = 500
EXPECTED_PROTOCOL = "faruq-v3-stb-capacity-paired-confirmation-v1"


def _validate_layout(data_root: str | Path):
    layout = discover_layout(data_root)
    if "val" not in layout.splits:
        raise RuntimeError("Validation split tidak tersedia")
    if "test" in layout.splits or (layout.root / "test").exists():
        raise RuntimeError("Audit complementarity menolak dataset yang mengekspos test")
    return layout


def _validate_paired_summary(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Paired summary tidak ditemukan: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("protocol") != EXPECTED_PROTOCOL:
        raise RuntimeError(f"Protocol paired summary tidak kompatibel: {payload.get('protocol')}")
    if tuple(int(seed) for seed in payload.get("seeds", [])) != SEEDS:
        raise RuntimeError(f"Paired summary harus berisi seeds {SEEDS}")
    if payload.get("evaluation_split") != "val":
        raise RuntimeError("Paired summary bukan validation-only")
    if payload.get("test_images_accessed") is not False or payload.get("test_opened") is not False:
        raise RuntimeError("Paired summary menunjukkan akses test; audit dihentikan")
    return payload


def _checkpoint_seed(path: str | Path) -> int:
    from ultralytics.utils.patches import torch_load

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {source}")
    payload = torch_load(source, map_location="cpu")
    args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(args, dict) or "seed" not in args:
        raise RuntimeError(f"Checkpoint tidak merekam train_args.seed: {source}")
    return int(args["seed"])


def _val_samples(layout):
    image_root, label_root = layout.splits["val"]
    samples = []
    for image_path in sorted(
        path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    ):
        relative = image_path.relative_to(image_root)
        label_path = (label_root / relative).with_suffix(".txt")
        samples.append((relative.as_posix(), image_path, parse_label(label_path, set(layout.names))))
    if not samples:
        raise RuntimeError("Validation split kosong")
    return samples


def _target_tensors(image_path: Path, annotations, device: torch.device):
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Gambar tidak dapat dibaca: {image_path}")
    height, width = image.shape[:2]
    boxes, labels = [], []
    for item in annotations:
        boxes.append(
            [
                (item.x_center - item.width * 0.5) * width,
                (item.y_center - item.height * 0.5) * height,
                (item.x_center + item.width * 0.5) * width,
                (item.y_center + item.height * 0.5) * height,
            ]
        )
        labels.append(int(item.class_id))
    return (
        torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def _confidence_ordered_match(pred_boxes, pred_conf, target_boxes, threshold=MATCH_IOU):
    if not len(pred_boxes) or not len(target_boxes):
        return []
    matrix = box_iou(pred_boxes, target_boxes)
    available = torch.ones(len(target_boxes), dtype=torch.bool, device=target_boxes.device)
    matches = []
    for pred_index in pred_conf.argsort(descending=True).tolist():
        candidate = matrix[pred_index].clone()
        candidate[~available] = -1
        value, target_index = candidate.max(dim=0)
        if float(value) < threshold:
            continue
        target_index = int(target_index)
        available[target_index] = False
        matches.append((int(pred_index), target_index, float(value)))
    return matches


def _model_events(
    checkpoint: str | Path,
    data_root: str | Path,
    *,
    seed: int,
    model_name: str,
    device: str,
):
    from ultralytics import YOLO

    checkpoint = Path(checkpoint).expanduser().resolve()
    recorded_seed = _checkpoint_seed(checkpoint)
    if recorded_seed != seed:
        raise RuntimeError(
            f"Seed checkpoint mismatch untuk {model_name}: diminta {seed}, checkpoint={recorded_seed}"
        )

    layout = _validate_layout(data_root)
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")

    model = YOLO(str(checkpoint))
    rows = {}
    samples = _val_samples(layout)
    for image_index, (image_key, image_path, annotations) in enumerate(samples, 1):
        target_boxes, target_labels = _target_tensors(image_path, annotations, torch_device)
        result = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            iou=PREDICTION_NMS_IOU,
            max_det=MAX_DET,
            device=device,
            verbose=False,
        )[0]
        if result.boxes is None:
            pred_boxes = target_boxes.new_zeros((0, 4))
            pred_conf = target_boxes.new_zeros((0,))
            pred_labels = target_labels.new_zeros((0,), dtype=torch.long)
        else:
            pred_boxes = result.boxes.xyxy.to(torch_device)
            pred_conf = result.boxes.conf.to(torch_device)
            pred_labels = result.boxes.cls.long().to(torch_device)

        accessibility = (
            box_iou(pred_boxes, target_boxes).max(dim=0).values >= MATCH_IOU
            if len(pred_boxes) and len(target_boxes)
            else torch.zeros(len(target_boxes), dtype=torch.bool, device=torch_device)
        )
        match_by_target = {
            target_index: (pred_index, iou)
            for pred_index, target_index, iou in _confidence_ordered_match(
                pred_boxes, pred_conf, target_boxes
            )
        }
        for target_index, expected_tensor in enumerate(target_labels):
            expected = int(expected_tensor)
            key = f"{image_key}::gt{target_index}"
            event = {
                "target_key": key,
                "image": image_key,
                "target_index": int(target_index),
                "gt_class_id": expected,
                "gt_class_name": layout.names[expected],
                "accessible": bool(accessibility[target_index]),
                "matched": False,
                "pred_class_id": None,
                "pred_class_name": None,
                "confidence": None,
                "iou": None,
                "correct": False,
            }
            if target_index in match_by_target:
                pred_index, iou = match_by_target[target_index]
                actual = int(pred_labels[pred_index])
                event.update(
                    matched=True,
                    pred_class_id=actual,
                    pred_class_name=layout.names[actual],
                    confidence=float(pred_conf[pred_index]),
                    iou=float(iou),
                    correct=actual == expected,
                )
            rows[key] = event

        if image_index % 100 == 0 or image_index == len(samples):
            print(
                f"{model_name} seed {seed}: {image_index}/{len(samples)} validation images",
                flush=True,
            )
    return rows, layout.names


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _pair_summary(cmc0_rows: dict, stb_rows: dict, names: dict[int, str]) -> tuple[dict, list[dict]]:
    if set(cmc0_rows) != set(stb_rows):
        missing_left = sorted(set(stb_rows) - set(cmc0_rows))[:5]
        missing_right = sorted(set(cmc0_rows) - set(stb_rows))[:5]
        raise RuntimeError(
            f"Target universe berbeda. Missing CMC0={missing_left}, missing STB={missing_right}"
        )

    counts = Counter()
    per_class = defaultdict(Counter)
    confusion_rescue_cmc0_to_stb = Counter()
    confusion_rescue_stb_to_cmc0 = Counter()
    object_rows = []

    for key in sorted(cmc0_rows):
        a = cmc0_rows[key]
        b = stb_rows[key]
        if a["gt_class_id"] != b["gt_class_id"]:
            raise RuntimeError(f"GT class mismatch untuk target {key}")
        class_name = a["gt_class_name"]
        a_correct, b_correct = bool(a["correct"]), bool(b["correct"])
        a_matched, b_matched = bool(a["matched"]), bool(b["matched"])

        counts["targets"] += 1
        per_class[class_name]["targets"] += 1
        counts["cmc0_correct"] += int(a_correct)
        counts["stb_correct"] += int(b_correct)
        per_class[class_name]["cmc0_correct"] += int(a_correct)
        per_class[class_name]["stb_correct"] += int(b_correct)
        counts["both_correct"] += int(a_correct and b_correct)
        counts["cmc0_only_correct"] += int(a_correct and not b_correct)
        counts["stb_only_correct"] += int(b_correct and not a_correct)
        counts["neither_correct"] += int(not a_correct and not b_correct)
        counts["cmc0_matched"] += int(a_matched)
        counts["stb_matched"] += int(b_matched)
        counts["jointly_matched"] += int(a_matched and b_matched)
        counts["cmc0_accessible"] += int(a["accessible"])
        counts["stb_accessible"] += int(b["accessible"])

        if not a_correct and b_correct:
            counts["cmc0_to_stb_overall_rescue"] += 1
            per_class[class_name]["cmc0_to_stb_overall_rescue"] += 1
            if a_matched and a["pred_class_name"] != class_name:
                confusion_rescue_cmc0_to_stb[(class_name, a["pred_class_name"])] += 1
        if not b_correct and a_correct:
            counts["stb_to_cmc0_overall_rescue"] += 1
            per_class[class_name]["stb_to_cmc0_overall_rescue"] += 1
            if b_matched and b["pred_class_name"] != class_name:
                confusion_rescue_stb_to_cmc0[(class_name, b["pred_class_name"])] += 1

        if a_matched and b_matched:
            counts["classification_only_targets"] += 1
            if (not a_correct) and b_correct:
                counts["cmc0_to_stb_classification_rescue"] += 1
            if (not b_correct) and a_correct:
                counts["stb_to_cmc0_classification_rescue"] += 1

        object_rows.append(
            {
                "target_key": key,
                "image": a["image"],
                "target_index": a["target_index"],
                "gt_class_id": a["gt_class_id"],
                "gt_class_name": class_name,
                "cmc0_accessible": a["accessible"],
                "cmc0_matched": a_matched,
                "cmc0_pred_class": a["pred_class_name"],
                "cmc0_confidence": a["confidence"],
                "cmc0_iou": a["iou"],
                "cmc0_correct": a_correct,
                "stb_accessible": b["accessible"],
                "stb_matched": b_matched,
                "stb_pred_class": b["pred_class_name"],
                "stb_confidence": b["confidence"],
                "stb_iou": b["iou"],
                "stb_correct": b_correct,
                "cmc0_to_stb_rescue": (not a_correct) and b_correct,
                "stb_to_cmc0_rescue": (not b_correct) and a_correct,
            }
        )

    total = counts["targets"]
    cmc0_errors = total - counts["cmc0_correct"]
    stb_errors = total - counts["stb_correct"]
    error_intersection = counts["neither_correct"]
    error_union = total - counts["both_correct"]
    oracle_correct = total - counts["neither_correct"]
    cmc0_accuracy = _safe_div(counts["cmc0_correct"], total)
    stb_accuracy = _safe_div(counts["stb_correct"], total)
    oracle_accuracy = _safe_div(oracle_correct, total)

    per_class_rows = {}
    for class_name in names.values():
        row = per_class[class_name]
        targets = row["targets"]
        cmc0_acc = _safe_div(row["cmc0_correct"], targets)
        stb_acc = _safe_div(row["stb_correct"], targets)
        oracle = _safe_div(
            row["cmc0_correct"] + row["stb_only_correct"] if "stb_only_correct" in row else 0,
            targets,
        )
        # Recompute oracle from target rows below to avoid relying on an unstored counter.
        class_objects = [item for item in object_rows if item["gt_class_name"] == class_name]
        oracle = _safe_div(sum(item["cmc0_correct"] or item["stb_correct"] for item in class_objects), targets)
        per_class_rows[class_name] = {
            "targets": int(targets),
            "cmc0_accuracy_iou50": cmc0_acc,
            "stb_accuracy_iou50": stb_acc,
            "stb_minus_cmc0_accuracy": stb_acc - cmc0_acc,
            "oracle_accuracy_iou50": oracle,
            "oracle_gain_over_best": oracle - max(cmc0_acc, stb_acc),
            "cmc0_to_stb_rescue": int(row["cmc0_to_stb_overall_rescue"]),
            "stb_to_cmc0_rescue": int(row["stb_to_cmc0_overall_rescue"]),
        }

    summary = {
        "targets": int(total),
        "cmc0": {
            "correct": int(counts["cmc0_correct"]),
            "accuracy_iou50": cmc0_accuracy,
            "matched_recall_iou50": _safe_div(counts["cmc0_matched"], total),
            "accessibility_iou50": _safe_div(counts["cmc0_accessible"], total),
            "errors": int(cmc0_errors),
        },
        "stb1": {
            "correct": int(counts["stb_correct"]),
            "accuracy_iou50": stb_accuracy,
            "matched_recall_iou50": _safe_div(counts["stb_matched"], total),
            "accessibility_iou50": _safe_div(counts["stb_accessible"], total),
            "errors": int(stb_errors),
        },
        "stb_minus_cmc0_accuracy": stb_accuracy - cmc0_accuracy,
        "contingency": {
            "both_correct": int(counts["both_correct"]),
            "cmc0_only_correct": int(counts["cmc0_only_correct"]),
            "stb_only_correct": int(counts["stb_only_correct"]),
            "neither_correct": int(counts["neither_correct"]),
        },
        "rescue": {
            "cmc0_to_stb_count": int(counts["cmc0_to_stb_overall_rescue"]),
            "cmc0_to_stb_rate_given_cmc0_error": _safe_div(
                counts["cmc0_to_stb_overall_rescue"], cmc0_errors
            ),
            "stb_to_cmc0_count": int(counts["stb_to_cmc0_overall_rescue"]),
            "stb_to_cmc0_rate_given_stb_error": _safe_div(
                counts["stb_to_cmc0_overall_rescue"], stb_errors
            ),
            "classification_only_jointly_matched": int(counts["classification_only_targets"]),
            "cmc0_to_stb_classification_rescue": int(
                counts["cmc0_to_stb_classification_rescue"]
            ),
            "stb_to_cmc0_classification_rescue": int(
                counts["stb_to_cmc0_classification_rescue"]
            ),
        },
        "error_overlap": {
            "intersection": int(error_intersection),
            "union": int(error_union),
            "jaccard": _safe_div(error_intersection, error_union),
        },
        "oracle": {
            "correct": int(oracle_correct),
            "accuracy_iou50": oracle_accuracy,
            "gain_over_best_model": oracle_accuracy - max(cmc0_accuracy, stb_accuracy),
        },
        "top_confusion_pair_rescues": {
            "cmc0_wrong_stb_correct": [
                {"gt": gt, "cmc0_pred": pred, "count": int(count)}
                for (gt, pred), count in confusion_rescue_cmc0_to_stb.most_common(20)
            ],
            "stb_wrong_cmc0_correct": [
                {"gt": gt, "stb_pred": pred, "count": int(count)}
                for (gt, pred), count in confusion_rescue_stb_to_cmc0.most_common(20)
            ],
        },
        "per_class": per_class_rows,
    }
    return summary, object_rows


def _aggregate(seed_summaries: dict[str, dict]) -> dict:
    metric_paths = {
        "stb_minus_cmc0_accuracy": lambda row: row["stb_minus_cmc0_accuracy"],
        "cmc0_to_stb_rescue_rate": lambda row: row["rescue"]["cmc0_to_stb_rate_given_cmc0_error"],
        "stb_to_cmc0_rescue_rate": lambda row: row["rescue"]["stb_to_cmc0_rate_given_stb_error"],
        "error_jaccard": lambda row: row["error_overlap"]["jaccard"],
        "oracle_gain_over_best": lambda row: row["oracle"]["gain_over_best_model"],
    }
    aggregate = {}
    for name, getter in metric_paths.items():
        values = [float(getter(seed_summaries[str(seed)])) for seed in SEEDS]
        aggregate[name] = {
            "mean": mean(values),
            "std": pstdev(values),
            "values": dict(zip((str(seed) for seed in SEEDS), values)),
        }
    aggregate["stb_accuracy_improved_seeds"] = sum(
        seed_summaries[str(seed)]["stb_minus_cmc0_accuracy"] > 0 for seed in SEEDS
    )
    aggregate["interpretation_guardrails"] = {
        "object_accuracy_is_not_map": True,
        "high_error_jaccard_suggests_redundancy_but_is_not_feature_similarity": True,
        "oracle_gain_is_an_upper_bound_not_a_real_fusion_result": True,
        "no_entropy_or_top2_margin_in_v1": True,
    }
    return aggregate


def _write_object_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_stb_cmc0_complementarity(
    cmc0_checkpoints: dict[int, str | Path],
    stb_checkpoints: dict[int, str | Path],
    data_root: str | Path,
    paired_summary: str | Path,
    output_root: str | Path,
    *,
    device: str = "0",
) -> dict:
    _validate_layout(data_root)
    frozen = _validate_paired_summary(paired_summary)
    if tuple(sorted(cmc0_checkpoints)) != SEEDS or tuple(sorted(stb_checkpoints)) != SEEDS:
        raise RuntimeError(f"Audit dikunci ke seeds {SEEDS}")

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    seed_summaries = {}
    all_rows = []

    for seed in SEEDS:
        print(f"\n=== COMPLEMENTARITY AUDIT | SEED {seed} ===", flush=True)
        cmc0_rows, names = _model_events(
            cmc0_checkpoints[seed], data_root, seed=seed, model_name="CMC0", device=device
        )
        stb_rows, stb_names = _model_events(
            stb_checkpoints[seed], data_root, seed=seed, model_name="STB1", device=device
        )
        if names != stb_names:
            raise RuntimeError("Class-name mapping berbeda antara CMC0 dan STB1")
        summary, rows = _pair_summary(cmc0_rows, stb_rows, names)
        for row in rows:
            row["seed"] = seed
        seed_summaries[str(seed)] = summary
        all_rows.extend(rows)
        _write_object_csv(rows, output_root / f"stb_cmc0_objects_seed{seed}.csv")

    payload = {
        "protocol": "faruq-v3-stb-cmc0-object-complementarity-v1",
        "source_paired_protocol": frozen["protocol"],
        "source_paired_decision": frozen.get("decision"),
        "training_executed": False,
        "evaluation_split": "val",
        "test_images_accessed": False,
        "test_opened": False,
        "seeds": list(SEEDS),
        "matching": {
            "class_agnostic": True,
            "iou_threshold": MATCH_IOU,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "prediction_nms_iou": PREDICTION_NMS_IOU,
            "order": "descending_prediction_confidence",
        },
        "definitions": {
            "overall_correct": "GT has a matched final detection at IoU>=0.50 with the correct class",
            "classification_only_rescue": "both models matched the same GT at IoU>=0.50; one classified it wrong and the other correctly",
            "oracle": "upper bound that counts a GT correct if either model is correct; not an implemented fusion",
            "error_jaccard": "Jaccard overlap of GT targets not correctly classified/detected by each model",
        },
        "aggregate": _aggregate(seed_summaries),
        "per_seed": seed_summaries,
    }
    summary_path = output_root / "stb_cmc0_complementarity.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_object_csv(all_rows, output_root / "stb_cmc0_objects_all_seeds.csv")
    payload["summary"] = str(summary_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only object-level complementarity audit for STB1 versus CMC0."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--paired-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cmc0-seed42", required=True)
    parser.add_argument("--cmc0-seed123", required=True)
    parser.add_argument("--cmc0-seed2026", required=True)
    parser.add_argument("--stb-seed42", required=True)
    parser.add_argument("--stb-seed123", required=True)
    parser.add_argument("--stb-seed2026", required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    result = run_stb_cmc0_complementarity(
        {
            42: args.cmc0_seed42,
            123: args.cmc0_seed123,
            2026: args.cmc0_seed2026,
        },
        {
            42: args.stb_seed42,
            123: args.stb_seed123,
            2026: args.stb_seed2026,
        },
        args.data_root,
        args.paired_summary,
        args.output_root,
        device=args.device,
    )
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    print("SAVED:", result["summary"])


if __name__ == "__main__":
    main()
