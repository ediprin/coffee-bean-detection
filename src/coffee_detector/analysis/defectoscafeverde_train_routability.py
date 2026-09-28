"""Train-only deployability audit for DefectosCafeVerde expert routing."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torchvision.ops import box_iou

from coffee_detector.analysis.coffee_fg_diagnostics import (
    _confidence_ordered_match,
    _greedy_match,
    _letterbox_sample,
    _raw_branches,
    _split_samples,
    _unwrap_head,
)
from coffee_detector.analysis.defectoscafeverde_error_complementarity import (
    MODEL_ORDER,
    REFERENCE,
    validate_result_contracts,
)


def _appearance_features(image: torch.Tensor, box: torch.Tensor) -> list[float]:
    """Return inference-available global and anchor-crop appearance statistics."""

    image = image[:, [2, 1, 0]]  # OpenCV BGR tensor -> RGB statistics.
    _, _, height, width = image.shape
    x1, y1, x2, y2 = box.detach().round().long().tolist()
    x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
    crop = image[:, :, y1:y2, x1:x2]
    if crop.numel() == 0:
        crop = image

    def describe(tensor: torch.Tensor) -> list[float]:
        rgb_mean = tensor.mean(dim=(0, 2, 3))
        rgb_std = tensor.std(dim=(0, 2, 3), unbiased=False)
        luminance = (
            0.2126 * tensor[:, 0] + 0.7152 * tensor[:, 1] + 0.0722 * tensor[:, 2]
        )
        lum_mean = luminance.mean()
        lum_std = luminance.std(unbiased=False)
        grad_x = (
            (luminance[:, :, 1:] - luminance[:, :, :-1]).abs().mean()
            if luminance.shape[-1] > 1
            else luminance.new_zeros(())
        )
        grad_y = (
            (luminance[:, 1:, :] - luminance[:, :-1, :]).abs().mean()
            if luminance.shape[-2] > 1
            else luminance.new_zeros(())
        )
        return [
            *rgb_mean.detach().cpu().tolist(),
            *rgb_std.detach().cpu().tolist(),
            float(lum_mean),
            float(lum_std),
            float(grad_x),
            float(grad_y),
        ]

    return describe(image) + describe(crop)


def _kept_final(network: torch.nn.Module, image: torch.Tensor, confidence: float, max_det: int):
    final, _, _ = _raw_branches(network, image, max_det)
    kept = final[final[:, 4] >= confidence]
    return kept[:, :4], kept[:, 5].long(), kept[:, 4]


def _associate_to_anchors(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
    anchors: torch.Tensor,
    threshold: float,
) -> dict[int, dict[str, Any]]:
    matches = _greedy_match(boxes, anchors, threshold)
    return {
        int(anchor_index): {
            "associated": True,
            "prediction_index": int(prediction_index),
            "predicted_class_id": int(labels[prediction_index]),
            "confidence": float(confidences[prediction_index]),
            "anchor_iou": float(iou),
            "box": boxes[prediction_index],
        }
        for prediction_index, anchor_index, iou in matches
    }


def _blank_expert() -> dict[str, Any]:
    return {
        "associated": False,
        "predicted_class_id": None,
        "confidence": 0.0,
        "anchor_iou": 0.0,
        "correct": False,
    }


def build_router_features(record: Mapping[str, Any], subset: tuple[str, ...], class_count: int) -> np.ndarray:
    """Build a feature vector containing no ground-truth fields."""

    values = [float(value) for value in record["appearance"]]
    votes = np.zeros(class_count, dtype=np.float64)
    predicted: dict[str, int | None] = {}
    for model in subset:
        row = record["models"][model]
        class_id = row["predicted_class_id"] if row["associated"] else None
        predicted[model] = class_id
        values.extend(
            [
                float(bool(row["associated"])),
                float(row["confidence"]),
                float(row["anchor_iou"]),
            ]
        )
        one_hot = np.zeros(class_count, dtype=np.float64)
        if class_id is not None:
            one_hot[int(class_id)] = 1.0
            votes[int(class_id)] += 1.0
        values.extend(one_hot.tolist())
    d0_class = predicted.get(REFERENCE)
    values.extend(
        [float(predicted[model] is not None and predicted[model] == d0_class) for model in subset[1:]]
    )
    values.extend((votes / max(len(subset), 1)).tolist())
    return np.asarray(values, dtype=np.float64)


def _preferred_label(record: Mapping[str, Any], subset: tuple[str, ...]) -> int:
    if record["models"][REFERENCE]["correct"]:
        return 0
    correct = [
        (float(record["models"][model]["confidence"]), -index, index)
        for index, model in enumerate(subset)
        if record["models"][model]["correct"]
    ]
    return max(correct)[2] if correct else 0


def _stable_fold(image: str, folds: int) -> int:
    name = Path(image).name.encode("utf-8")
    return int.from_bytes(hashlib.sha256(name).digest()[:8], "big") % folds


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    classes: int,
    alpha: float,
) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (x_train - mean) / scale
    test = (x_test - mean) / scale
    train = np.column_stack((np.ones(len(train)), train))
    test = np.column_stack((np.ones(len(test)), test))
    targets = np.eye(classes, dtype=np.float64)[y_train]
    penalty = np.eye(train.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ targets)
    return (test @ weights).argmax(axis=1)


def select_minimal_subset(records: list[dict], model_order: tuple[str, ...] = MODEL_ORDER) -> dict:
    """Select the smallest D0-containing subset retaining 90% of full oracle gain."""

    total = len(records)
    correct = {
        model: np.asarray([bool(row["models"][model]["correct"]) for row in records])
        for model in model_order
    }
    d0_correct = int(correct[REFERENCE].sum())
    full_correct = int(np.logical_or.reduce([correct[m] for m in model_order]).sum())
    full_gain = full_correct - d0_correct
    required = int(np.ceil(0.90 * full_gain)) if full_gain > 0 else 0
    candidates = []
    for size in range(1, len(model_order) + 1):
        for tail in itertools.combinations(model_order[1:], size - 1):
            subset = (REFERENCE, *tail)
            union = int(np.logical_or.reduce([correct[m] for m in subset]).sum())
            gain = union - d0_correct
            candidates.append((len(subset), -gain, tuple(model_order.index(m) for m in subset), subset, union))
        eligible = [row for row in candidates if row[0] == size and -row[1] >= required]
        if eligible:
            chosen = min(eligible)
            break
    else:
        chosen = min(candidates)
    subset, subset_correct = chosen[3], chosen[4]
    return {
        "subset": list(subset),
        "d0_correct": d0_correct,
        "full_oracle_correct": full_correct,
        "full_oracle_gain_objects": full_gain,
        "full_oracle_gain": full_gain / max(total, 1),
        "required_gain_objects": required,
        "subset_oracle_correct": subset_correct,
        "subset_oracle_gain_objects": subset_correct - d0_correct,
        "subset_oracle_gain": (subset_correct - d0_correct) / max(total, 1),
        "retained_full_gain_fraction": (subset_correct - d0_correct) / max(full_gain, 1),
    }


def evaluate_router(records: list[dict], subset: tuple[str, ...], class_count: int, folds: int = 5, alpha: float = 10.0) -> dict:
    anchored_indices = [index for index, row in enumerate(records) if row["anchor_available"]]
    if not anchored_indices:
        raise RuntimeError("Tidak ada D0 anchor untuk audit router")
    fold_ids = np.asarray([_stable_fold(records[index]["image"], folds) for index in anchored_indices])
    selected_models: list[str | None] = [None] * len(anchored_indices)
    confidence_models: list[str | None] = [None] * len(anchored_indices)
    fold_subsets = {}
    feature_counts = {}
    for fold in range(folds):
        train = fold_ids != fold
        test = fold_ids == fold
        if not train.any() or not test.any():
            raise RuntimeError(f"Fold kosong: {fold}")
        train_records = [records[anchored_indices[index]] for index in np.flatnonzero(train)]
        fold_subset = tuple(select_minimal_subset(train_records)["subset"])
        fold_subsets[str(fold)] = list(fold_subset)
        x_train = np.stack([build_router_features(record, fold_subset, class_count) for record in train_records])
        test_records = [records[anchored_indices[index]] for index in np.flatnonzero(test)]
        x_test = np.stack([build_router_features(record, fold_subset, class_count) for record in test_records])
        y_train = np.asarray([_preferred_label(record, fold_subset) for record in train_records], dtype=np.int64)
        predictions = _ridge_predict(x_train, y_train, x_test, len(fold_subset), alpha)
        feature_counts[str(fold)] = int(x_train.shape[1])
        for local, anchored_local_index in enumerate(np.flatnonzero(test)):
            record = records[anchored_indices[anchored_local_index]]
            selected_models[anchored_local_index] = fold_subset[int(predictions[local])]
            confidence_models[anchored_local_index] = max(
                fold_subset, key=lambda item: float(record["models"][item]["confidence"])
            )
    router_correct = 0
    confidence_correct = 0
    choices = Counter()
    for local, record_index in enumerate(anchored_indices):
        record = records[record_index]
        model = selected_models[local]
        if model is None or confidence_models[local] is None:
            raise RuntimeError("Prediksi OOF tidak lengkap")
        choices[model] += 1
        router_correct += int(record["models"][model]["correct"])
        confidence_correct += int(record["models"][confidence_models[local]]["correct"])
    total = len(records)
    d0_correct = sum(bool(row["models"][REFERENCE]["correct"]) for row in records)
    standalone = {
        model: sum(bool(row["models"][model]["correct"]) for row in records)
        for model in MODEL_ORDER
    }
    subset_oracle = sum(any(row["models"][m]["correct"] for m in subset) for row in records)
    oracle_gain = subset_oracle - d0_correct
    global_labels = np.asarray([_preferred_label(records[index], subset) for index in anchored_indices])
    return {
        "folds": folds,
        "ridge_alpha": alpha,
        "groups": len({row["image"] for row in records}),
        "anchored_targets": len(anchored_indices),
        "unanchored_targets": total - len(anchored_indices),
        "fold_subsets": fold_subsets,
        "feature_count_by_fold": feature_counts,
        "full_train_subset_label_counts": {
            subset[index]: int((global_labels == index).sum()) for index in range(len(subset))
        },
        "selection_counts": dict(choices),
        "d0_correct": d0_correct,
        "standalone_correct": standalone,
        "best_standalone_correct": max(standalone.values()),
        "max_confidence_correct": confidence_correct,
        "router_correct": router_correct,
        "subset_oracle_correct": subset_oracle,
        "d0_recall": d0_correct / total,
        "best_standalone_recall": max(standalone.values()) / total,
        "max_confidence_recall": confidence_correct / total,
        "router_recall": router_correct / total,
        "subset_oracle_recall": subset_oracle / total,
        "router_gain_over_d0": (router_correct - d0_correct) / total,
        "captured_oracle_gain_fraction": (router_correct - d0_correct) / max(oracle_gain, 1),
    }


def summarize_routability(records: list[dict], class_count: int) -> dict:
    selection = select_minimal_subset(records)
    subset = tuple(selection["subset"])
    router = evaluate_router(records, subset, class_count)
    criteria = {
        "associated_full_oracle_gain_at_least_1_point": selection["full_oracle_gain"] >= 0.01,
        "minimal_subset_retains_at_least_90_percent_full_gain": selection["retained_full_gain_fraction"] >= 0.90,
        "oof_router_gain_at_least_0_5_point": router["router_gain_over_d0"] >= 0.005,
        "oof_router_captures_at_least_25_percent_oracle_gain": router["captured_oracle_gain_fraction"] >= 0.25,
        "oof_router_not_below_best_standalone": router["router_correct"] >= router["best_standalone_correct"],
    }
    passed = all(criteria.values())
    return {
        "minimal_subset": selection,
        "oof_router": router,
        "criteria": criteria,
        "decision": "AUTHORIZE_SEPARATE_ROUTED_ARCHITECTURE_SCREEN" if passed else "STOP_ROUTED_FUSION",
        "next": "FREEZE_SINGLE_SEED_ARCHITECTURE_PROTOCOL" if passed else "RETAIN_D0DIRECT",
    }


def run_train_routability_audit(
    data_root: str | Path,
    result_paths: Mapping[str, str | Path],
    checkpoint_paths: Mapping[str, str | Path],
    output: str | Path,
    *,
    records_output: str | Path | None = None,
    device: str = "cpu",
    image_size: int = 640,
    final_confidence: float = 0.001,
    association_iou: float = 0.30,
    correctness_iou: float = 0.50,
    max_det: int = 500,
) -> dict:
    contract = validate_result_contracts(result_paths, checkpoint_paths)
    layout, samples = _split_samples(data_root, "train")
    if "test" in layout.splits or (Path(data_root) / "test").exists():
        raise RuntimeError("Test tidak boleh tersedia pada audit train-only")
    names = {int(index): str(name) for index, name in layout.names.items()}
    if len(names) != 12:
        raise RuntimeError(f"Ontologi harus 12 kelas, ditemukan {len(names)}")
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA diminta tetapi tidak tersedia: {torch_device}")

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    record_path = Path(records_output).expanduser().resolve() if records_output else destination.with_name(destination.stem + "_targets.jsonl")
    resume_contract = {
        "format": "coffee_detector.defectoscafeverde.train_routability.resume.v1",
        "dataset_audit_sha256": contract["dataset_audit_sha256"],
        "checkpoint_sha256": {
            model: contract["models"][model]["checkpoint_sha256"] for model in MODEL_ORDER
        },
        "settings": {
            "image_size": image_size,
            "final_confidence": final_confidence,
            "association_iou": association_iou,
            "correctness_iou": correctness_iou,
            "max_det": max_det,
        },
    }
    resume_marker = record_path.with_suffix(record_path.suffix + ".contract.json")
    if record_path.exists():
        if not resume_marker.is_file():
            raise RuntimeError(f"Record resume tanpa kontrak: {record_path}")
        observed_contract = json.loads(resume_marker.read_text(encoding="utf-8"))
        if observed_contract != resume_contract:
            raise RuntimeError("Kontrak resume routability berbeda; output lama tidak boleh dicampur")
        records = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text("", encoding="utf-8")
        resume_marker.write_text(json.dumps(resume_contract, indent=2) + "\n", encoding="utf-8")
        records = []
    expected_counts = {str(image_path): len(annotations) for image_path, annotations in samples}
    observed_counts = Counter(row["image"] for row in records)
    invalid_resume = {
        image: count for image, count in observed_counts.items()
        if image not in expected_counts or count != expected_counts[image]
    }
    if invalid_resume:
        raise RuntimeError(f"Record resume parsial/tidak cocok: {invalid_resume}")
    completed_images = set(observed_counts)
    pending_samples = [sample for sample in samples if str(sample[0]) not in completed_images]
    print(
        f"RESUME ROUTABILITY: {len(completed_images)}/{len(samples)} images selesai; "
        f"{len(pending_samples)} tersisa",
        flush=True,
    )

    networks = {}
    if pending_samples:
        from ultralytics import YOLO

        for model in MODEL_ORDER:
            network = YOLO(str(Path(checkpoint_paths[model]).resolve())).model.to(torch_device).eval()
            if int(_unwrap_head(network).nc) != len(names):
                raise RuntimeError(f"Jumlah kelas checkpoint salah: {model}")
            networks[model] = network

    with torch.inference_mode(), record_path.open("a", encoding="utf-8") as record_stream:
        for image_index, (image_path, annotations) in enumerate(pending_samples, 1):
            image, target_boxes, target_labels, _ = _letterbox_sample(image_path, annotations, image_size, torch_device)
            predictions = {
                model: _kept_final(networks[model], image, final_confidence, max_det)
                for model in MODEL_ORDER
            }
            d0_boxes, d0_labels, d0_confidences = predictions[REFERENCE]
            target_matches = _confidence_ordered_match(d0_boxes, d0_confidences, target_boxes, correctness_iou)
            target_to_anchor = {int(target): int(prediction) for prediction, target, _ in target_matches}
            anchor_indices = sorted(set(target_to_anchor.values()))
            anchor_boxes = d0_boxes[anchor_indices] if anchor_indices else d0_boxes.new_zeros((0, 4))
            global_to_local = {global_index: local for local, global_index in enumerate(anchor_indices)}
            associated: dict[str, dict[int, dict[str, Any]]] = {}
            for model in MODEL_ORDER[1:]:
                boxes, labels, confidences = predictions[model]
                local = _associate_to_anchors(boxes, labels, confidences, anchor_boxes, association_iou)
                associated[model] = {anchor_indices[index]: row for index, row in local.items()}

            image_records = []
            for target_index, annotation in enumerate(annotations):
                target_class = int(annotation.class_id)
                anchor_index = target_to_anchor.get(target_index)
                record = {
                    "image": str(image_path),
                    "target_index": target_index,
                    "class_id": target_class,
                    "class_name": names[target_class],
                    "anchor_available": anchor_index is not None,
                    "appearance": [],
                    "models": {},
                }
                if anchor_index is None:
                    record["models"] = {model: _blank_expert() for model in MODEL_ORDER}
                    image_records.append(record)
                    continue
                anchor_box = d0_boxes[anchor_index]
                record["appearance"] = _appearance_features(image, anchor_box)
                d0_iou = float(box_iou(anchor_box[None], target_boxes[target_index][None])[0, 0])
                record["models"][REFERENCE] = {
                    "associated": True,
                    "predicted_class_id": int(d0_labels[anchor_index]),
                    "confidence": float(d0_confidences[anchor_index]),
                    "anchor_iou": 1.0,
                    "correct": d0_iou >= correctness_iou and int(d0_labels[anchor_index]) == target_class,
                }
                for model in MODEL_ORDER[1:]:
                    row = associated[model].get(anchor_index)
                    if row is None:
                        record["models"][model] = _blank_expert()
                        continue
                    candidate_iou = float(box_iou(row["box"][None], target_boxes[target_index][None])[0, 0])
                    record["models"][model] = {
                        "associated": True,
                        "predicted_class_id": row["predicted_class_id"],
                        "confidence": row["confidence"],
                        "anchor_iou": row["anchor_iou"],
                        "correct": candidate_iou >= correctness_iou and row["predicted_class_id"] == target_class,
                    }
                image_records.append(record)
            for record in image_records:
                record_stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            record_stream.flush()
            records.extend(image_records)
            completed = len(completed_images) + image_index
            if completed % 100 == 0 or image_index == len(pending_samples):
                print(f"ROUTABILITY TRAIN: {completed}/{len(samples)} images", flush=True)

    summary = summarize_routability(records, len(names))
    payload = {
        "format": "coffee_detector.defectoscafeverde.train_routability.v1",
        "scope": "grouped_train_only_oof_deployable_feature_audit",
        "contract": contract,
        "settings": {
            "image_size": image_size,
            "final_confidence": final_confidence,
            "association_iou": association_iou,
            "correctness_iou": correctness_iou,
            "max_det": max_det,
        },
        "images": len(samples),
        "targets": len(records),
        "classes": names,
        **summary,
        "claim_boundary": "train-only grouped OOF routability audit; no validation or test superiority claim",
        "detector_training_executed": False,
        "router_fitting_executed": True,
        "validation_images_accessed": False,
        "test_images_accessed": False,
    }
    payload["target_records"] = str(record_path)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="DefectosCafeVerde train-only expert routability audit")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records-output")
    parser.add_argument("--device", default="cpu")
    for model in MODEL_ORDER:
        parser.add_argument(f"--{model.lower()}-checkpoint", required=True)
        parser.add_argument(f"--{model.lower()}-result", required=True)
    args = parser.parse_args()
    checkpoints = {model: getattr(args, f"{model.lower()}_checkpoint") for model in MODEL_ORDER}
    results = {model: getattr(args, f"{model.lower()}_result") for model in MODEL_ORDER}
    payload = run_train_routability_audit(
        args.data_root, results, checkpoints, args.output,
        records_output=args.records_output, device=args.device,
    )
    print(json.dumps({key: payload[key] for key in ("minimal_subset", "oof_router", "criteria", "decision", "next")}, indent=2))


if __name__ == "__main__":
    main()

